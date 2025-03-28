# Load every pkl file, load the image and see if there is a grasp in the image
import argparse
import os
import random
from pathlib import Path

import cv2
import hamer
import numpy as np
import torch
from detectron2.config import LazyConfig
from dex_grasp.utils.load_datasets import load_img_and_traj
from hamer.configs import CACHE_DIR_HAMER
from hamer.datasets.vitdet_dataset import DEFAULT_MEAN, DEFAULT_STD, ViTDetDataset
from hamer.models import DEFAULT_CHECKPOINT, HAMER, download_models, load_hamer
from hamer.utils import recursive_to
from hamer.utils.renderer import Renderer, cam_crop_to_full
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from vitpose_model import ViTPoseModel

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def load_model(save_dir):
    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)

    # Setup HaMeR model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    # Load detector
    cfg_path = (
        Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
    )
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    return model, model_cfg, device, detector, cpm, renderer


def get_detection(
    img_cv2, save_dir, img_fn, model, model_cfg, device, detector, cpm, renderer
):
    # Now apply hammer model to detect hand
    # Detect humans in image

    cv2.imwrite(os.path.join(save_dir, f"{img_fn}_img.png"), img_cv2)

    det_out = detector(img_cv2)
    img = img_cv2.copy()[:, :, ::-1]

    det_instances = det_out["instances"]
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()

    # Detect human keypoints for each person
    vitposes_out = cpm.predict_pose(
        img,
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
    )

    # print(vitposes_out)

    bboxes = []
    is_right = []

    # Use hands based on hand keypoint detections
    for vitposes in vitposes_out:
        left_hand_keyp = vitposes["keypoints"][-42:-21]
        right_hand_keyp = vitposes["keypoints"][-21:]

        # Rejecting not confident detections
        keyp = left_hand_keyp
        valid = keyp[:, 2] > 0.5
        if sum(valid) > 3:
            bbox = [
                keyp[valid, 0].min(),
                keyp[valid, 1].min(),
                keyp[valid, 0].max(),
                keyp[valid, 1].max(),
            ]
            bboxes.append(bbox)
            is_right.append(0)
        keyp = right_hand_keyp
        valid = keyp[:, 2] > 0.5
        if sum(valid) > 3:
            bbox = [
                keyp[valid, 0].min(),
                keyp[valid, 1].min(),
                keyp[valid, 0].max(),
                keyp[valid, 1].max(),
            ]
            bboxes.append(bbox)
            is_right.append(1)

    if len(bboxes) == 0:
        return None

    boxes = np.stack(bboxes)
    right = np.stack(is_right)

    # Run reconstruction on all detected hands
    dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=2.0)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=0
    )

    all_verts = []
    all_cam_t = []
    all_right = []

    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        multiplier = 2 * batch["right"] - 1
        pred_cam = out["pred_cam"]
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        multiplier = 2 * batch["right"] - 1
        scaled_focal_length = (
            model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        )
        pred_cam_t_full = (
            cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length
            )
            .detach()
            .cpu()
            .numpy()
        )

        # Render the result
        batch_size = batch["img"].shape[0]
        for n in range(batch_size):
            # Get filename from path img_path
            # img_fn, _ = os.path.splitext(os.path.basename(img_path))
            person_id = int(batch["personid"][n])
            white_img = (
                torch.ones_like(batch["img"][n]).cpu()
                - DEFAULT_MEAN[:, None, None] / 255
            ) / (DEFAULT_STD[:, None, None] / 255)
            input_patch = batch["img"][n].cpu() * (DEFAULT_STD[:, None, None] / 255) + (
                DEFAULT_MEAN[:, None, None] / 255
            )
            input_patch = input_patch.permute(1, 2, 0).numpy()

            regression_img = renderer(
                out["pred_vertices"][n].detach().cpu().numpy(),
                out["pred_cam_t"][n].detach().cpu().numpy(),
                batch["img"][n],
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )

            final_img = np.concatenate([input_patch, regression_img], axis=1)

            cv2.imwrite(
                os.path.join(save_dir, f"{img_fn}_{person_id}.png"),
                255 * final_img[:, :, ::-1],
            )

            # Add all verts and cams to list
            verts = out["pred_vertices"][n].detach().cpu().numpy()
            is_right = batch["right"][n].cpu().numpy()
            verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
            cam_t = pred_cam_t_full[n]
            all_verts.append(verts)
            all_cam_t.append(cam_t)
            all_right.append(is_right)

    # Render front view
    if len(all_verts) > 0:
        misc_args = dict(
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
            focal_length=scaled_focal_length,
        )
        cam_view = renderer.render_rgba_multiple(
            all_verts,
            cam_t=all_cam_t,
            render_res=img_size[n],
            is_right=all_right,
            **misc_args,
        )

        # Overlay image
        input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
        input_img = np.concatenate(
            [input_img, np.ones_like(input_img[:, :, :1])], axis=2
        )  # Add alpha channel
        input_img_overlay = (
            input_img[:, :, :3] * (1 - cam_view[:, :, 3:])
            + cam_view[:, :, :3] * cam_view[:, :, 3:]
        )

        cv2.imwrite(
            os.path.join(save_dir, f"{img_fn}_all.jpg"),
            255 * input_img_overlay[:, :, ::-1],
        )


def dump_demo_hand_detections(img_folder, save_dir):
    # Get all demo images ends with .jpg or .png
    img_paths = [
        img for end in ["*.jpg", "*.png"] for img in Path(img_folder).glob(end)
    ]

    model, model_cfg, device, detector, cpm, renderer = load_model(save_dir)

    # Iterate over all images in folder
    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))
        img_fn, _ = os.path.splitext(os.path.basename(img_path))
        get_detection(
            img_cv2,
            save_dir,
            img_fn=img_fn,
            model=model,
            model_cfg=model_cfg,
            device=device,
            detector=detector,
            cpm=cpm,
            renderer=renderer,
        )
        print(img_fn)


def dump_hand_detections(pkl_dir, save_dir, num_files=20):
    pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith(".pkl")]

    # Randomly select 20 files
    selected_files = random.sample(pkl_files, num_files)

    model, model_cfg, device, detector, cpm, renderer = load_model(save_dir)

    # Make output directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    # Process each selected file
    for i, pkl_file in enumerate(selected_files):
        file_path = os.path.join(pkl_dir, pkl_file)
        print(f"Processing file {i+1}/{num_files}: {pkl_file}")

        # Load and visualize
        img_cv2, traj, obj = load_img_and_traj(file_path)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        get_detection(
            img_cv2,
            save_dir,
            img_fn=f"img_{i+1}",
            model=model,
            model_cfg=model_cfg,
            device=device,
            detector=detector,
            cpm=cpm,
            renderer=renderer,
        )


if __name__ == "__main__":
    # dump_demo_hand_detections(
    #     "/home/irmak/Workspace/nyu-big-data-and-ml/project/submodules/hamer/example_data",
    #     "/home/irmak/Workspace/nyu-big-data-and-ml/project/submodules/hamer/hamer_demo_hand_detections",
    # )
    random.seed(42)
    dump_hand_detections(
        pkl_dir="/data_ssd/irmak/deft-data-all/ego4d-r3m/labels_obj_bbox",
        save_dir="/home/irmak/Workspace/nyu-big-data-and-ml/project/dex_grasp/utils/ego4d-r3m_hand_detections",
    )
