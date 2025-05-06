# A dataset class to return mean contact point, grasp rotation and the handpose
import glob
import os
import pickle

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor


def center_crop_square(image):
    if isinstance(image, torch.Tensor):
        _, h, w = image.shape  # (C, H, W)
    elif isinstance(image, np.ndarray):
        h, w = image.shape[:2]
    else:
        w, h = image.size  # PIL Image: (W, H)

    crop_size = min(h, w)
    return F.center_crop(image, output_size=[crop_size, crop_size])


class GraspDataset(Dataset):
    def __init__(self, pkl_dir, return_cropped_image=False, transform_contact=False):
        self.pkl_files = glob.glob(os.path.join(pkl_dir, "*.pkl"))
        self.return_cropped_image = return_cropped_image
        self.transform_contact = transform_contact

        # Initialize the processor with a specific model
        # NOTE: We will be using Dinov2-base as our image processor / and encoder
        # self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        if return_cropped_image:
            self.transform = (
                transforms.Compose(  # NOTE: This is basically used for debugging!
                    [transforms.ToTensor(), transforms.Resize((224, 224))]
                )
            )
        else:
            self.transform = (
                transforms.Compose(  # NOTE: This is basically used for debugging!
                    [
                        transforms.ToTensor(),
                        transforms.Lambda(center_crop_square),
                        transforms.Resize((224, 224)),
                    ]
                )
            )

    def _transform_contact_point(self, contact_point, org_img, bbox):
        if isinstance(org_img, torch.Tensor):
            _, h, w = org_img.shape  # (C, H, W)
        elif isinstance(org_img, np.ndarray):
            h, w = org_img.shape[:2]

        if not self.return_cropped_image:
            crop_h, crop_w = min(h, w), min(h, w)
        else:  # NOTE: if the image is cropped then we should not really thinking of cropped the image from the middle!
            crop_h, crop_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        resize_size = (224, 224)

        """
        contact_point: (batch_size, 2) torch tensor
        Returns transformed points: (batch_size, 2) torch tensor
        """
        orig_h, orig_w = h, w
        resize_h, resize_w = resize_size

        if not self.return_cropped_image:
            crop_x_start = (orig_w - crop_w) / 2
            crop_y_start = (orig_h - crop_h) / 2
        else:
            crop_x_start = bbox[0]
            crop_y_start = bbox[1]

        scale_x = resize_w / crop_w
        scale_y = resize_h / crop_h

        y = contact_point[:, 1]
        x = contact_point[:, 0]

        x_cropped = x - crop_x_start
        y_cropped = y - crop_y_start

        print(f"x_cropped: {x_cropped}, y_cropped: {y_cropped}")

        x_resized = x_cropped * scale_x
        y_resized = y_cropped * scale_y

        transformed_points = torch.stack([x_resized, y_resized], dim=-1)

        # print(f"cropped_img.shape: {cropped_img.shape}, org_img.shape: {org_img.shape}")
        # print(
        #     f"transformed_points: {transformed_points}, org_contact_point: {contact_point}"
        # )

        return transformed_points

    def _crop_image(self, org_image, bbox):
        if self.return_cropped_image:
            y1, x1, y2, x2 = [int(x) for x in bbox]
            # bbox_offset = 20
            # y1, x1, y2, x2 = (
            #     int(y1) - bbox_offset,
            #     int(x1) - bbox_offset,
            #     int(y2) + bbox_offset,
            #     int(x2) + bbox_offset,
            # )
            if y1 < 0:
                y1 = 0
            if x1 < 0:
                x1 = 0
            if x2 > org_image.shape[0]:
                x2 = org_image.shape[0]
            if y2 > org_image.shape[1]:
                y2 = org_image.shape[1]

            # Crop the image using the bounding box coordinates
            cropped_image = org_image[x1:x2, y1:y2]
            return cropped_image, torch.Tensor([y1, x1, y2, x2])
        else:
            return org_image, bbox

    def load_pkl(self, pkl_file_path):
        with open(pkl_file_path, "rb") as f:
            data = pickle.load(f)

        # NOTE: For now we're priotizing the right hand
        if "right_hand" in data["contact_grasp"][0]:
            hand_pose = data["contact_grasp"][0]["right_hand"]["pred_hand_pose"]
        elif "left_hand" in data["contact_grasp"][0]:
            hand_pose = data["contact_grasp"][0]["left_hand"]["pred_hand_pose"]

        if len(hand_pose.shape) == 1:
            hand_pose = np.expand_dims(hand_pose, axis=0)

        org_image = data["img"]
        bbox = torch.FloatTensor(data["contact_object_det"]["bbox"])

        # print(data["contact_object_det"].keys(), bbox.shape)
        if "label" in data["contact_object_det"]:
            object_label = data["contact_object_det"]["label"]
        else:
            object_label = data["contact_object_det"]["class"]
        cropped_image, cropped_bbox = self._crop_image(org_image, bbox)

        # Get contact points
        contact_points = torch.FloatTensor(data["contact"])[:5]
        if len(contact_points) < 5:
            # Append the last element until we have 5 points
            last_point = contact_points[-1].unsqueeze(0)
            while len(contact_points) < 5:
                contact_points = torch.cat([contact_points, last_point], dim=0)

        print(f"pre transform contact_points: {contact_points}, bbox: {bbox}")
        if self.transform_contact:
            # try:
            contact_points = self._transform_contact_point(
                contact_points, org_img=org_image, bbox=bbox
            )
            bbox[:2] = self._transform_contact_point(
                cropped_bbox[:2].unsqueeze(0), org_img=org_image, bbox=cropped_bbox
            )[0]
            bbox[2:] = self._transform_contact_point(
                cropped_bbox[2:].unsqueeze(0), org_img=org_image, bbox=cropped_bbox
            )[0]
            # except Exception as e:
            #     print(e)
            #     print(bbox, org_image.shape, cropped_image.shape)

        print(f"post transform contact_points: {contact_points}, bbox: {bbox}")
        # NOTE: Not sure if i want to return the original image as well
        image = self.transform(cropped_image).clamp(
            0, 1
        )  # Should clamp it to input to clip preprocessor

        hand_pose = torch.FloatTensor(hand_pose)[0]

        return (
            image,
            contact_points,
            hand_pose[:3],
            hand_pose[3:],
            object_label,
            bbox,
        )

    def __getitem__(self, idx):
        pkl_path = self.pkl_files[idx]
        pkl_data = self.load_pkl(pkl_path)
        return pkl_data

    def __len__(self):
        return len(self.pkl_files)


class DexGraspEvalDataset(Dataset):
    def __init__(self, pkl_dir):
        self.pkl_files = glob.glob(os.path.join(pkl_dir, "*.pkl"))
        self.transform = (
            transforms.Compose(  # NOTE: This is basically used for debugging!
                [
                    transforms.ToTensor(),
                    transforms.Lambda(center_crop_square),
                    transforms.Resize((224, 224)),
                ]
            )
        )

    def __len__(self):
        return len(self.pkl_files)

    def load_pkl(self, pkl_file_path):
        with open(pkl_file_path, "rb") as f:
            data = pickle.load(f)

        # NOTE: Not sure if i want to return the original image as well
        image = self.transform(data["img"]).clamp(
            0, 1
        )  # Should clamp it to input to clip preprocessor
        text_prompt = data["text"]

        return (
            image,
            text_prompt,
        )

    def __getitem__(self, idx):
        pkl_path = self.pkl_files[idx]
        pkl_data = self.load_pkl(pkl_path)
        return pkl_data


if __name__ == "__main__":
    dataset = GraspDataset(
        pkl_dir="/data_ssd/irmak/deft-data-all/ego4d-r3m/labels_obj_bbox",
        return_cropped_image=False,
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(len(dataset))
    batch = next(iter(dataloader))

    # Visualize the image
    img = batch[0]
    task_description = batch[4]
    gt_mu = batch[1][:, :, :2]
    gt_grasp_rotation = batch[2]
    gt_grasp_pose = batch[3]
    print(img.shape)
    # Convert to numpy and ensure correct shape
    image_np = img.numpy()
    # If image is in range [0,1], scale to [0,255]
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)
    # Ensure correct shape for PIL
    if len(image_np.shape) == 3 and image_np.shape[0] == 3:  # CHW format
        image_np = np.transpose(image_np, (1, 2, 0))  # Convert to HWC
    image_pil = Image.fromarray(image_np)
    image_pil.save(f"debug_image_{task_description}.png")
    print(task_description)
    # import pdb

    # pdb.set_trace()
