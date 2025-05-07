# TODO: Implement the OOD evaluation - this will take some example images and check if the model is able to detect poses

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from dex_grasp.dataset.get_dataloaders import get_eval_dataloader
from dex_grasp.evaluation.dataset_eval import Evaluator
from dex_grasp.models.lang_sam import LangSAM
from dex_grasp.utils.config import TrainingConfig
from dex_grasp.utils.visualization import fig_to_img, vis_dino_boxes, vis_sam_mask
from PIL import Image
from torchvision.transforms import ToPILImage, transforms


class OOD_Evaluator(Evaluator):

    def __init__(self, ood_dset_dir, **kwargs):
        self.ood_dset_dir = ood_dset_dir
        super().__init__(**kwargs)

        self.image_transform = transforms.Compose(
            [
                # transforms.ToTensor(),
                transforms.Resize((224, 224)),
                # transforms.ToPILImage(),
            ]
        )

        self._load_cfg()
        if self.cfg.crop_image:
            self.langsam = LangSAM()

    def _load_data(self):
        self.test_loader = get_eval_dataloader(batch_size=32, num_workers=4)

    def _load_cfg(self):
        self.cfg = TrainingConfig()
        self.cfg.load_config(self.checkpoint_dir)
        print(self.cfg)

    def crop_batch(self, img, task_description):
        cropped_imgs = []
        bboxes = []
        for i in range(len(img)):
            cropped_img, bbox = self.crop_image(img[i], task_description[i], i)
            cropped_imgs.append(cropped_img)
            bboxes.append(bbox)
        cropped_imgs = torch.stack(cropped_imgs)
        return cropped_imgs, img, bboxes

    def crop_image(self, img, task_description, test_id):
        print(
            f"crop_image - > img.shape: {img.shape} - task_description: {task_description}"
        )

        # img = img.cpu()
        img_pil = ToPILImage()(img)
        masks, boxes, phrases, logits, embeddings = self.langsam.predict(
            image_pil=img_pil, text_prompt=task_description
        )
        # import pdb

        # pdb.set_trace()

        if len(boxes) == 0:
            img_cropped = img
            x_min, y_min, x_max, y_max = 0, 0, 224, 224
        else:
            bbox = boxes[0]
            x_min, y_min, x_max, y_max = (
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[2]),
                int(bbox[3]),
            )

            # print(f"bbox: {bbox}, box_width: {box_width}, box_height: {box_height}")
            # print(f"img.shape: {img.shape}")
            img_cropped = img[:, y_min:y_max, x_min:x_max]
            # fig, ax = plt.subplots(nrows=1, ncols=3)
            # ax[0] = vis_dino_boxes(ax[0], img.permute(1, 2, 0), boxes, logits)
            # ax[1] = vis_sam_mask(ax[1], masks[0])
            # ax[2].imshow(img_cropped.permute(1, 2, 0))
            # plt.savefig(f"{self.dump_dir}/ex_sam_{test_id}.png")
            # plt.close(fig)

            img_cropped = self.image_transform(img_cropped)
            # print(f"img_cropped.shape: {img_cropped.shape}")

        return img_cropped, [x_min, y_min, x_max, y_max]

    def detransform_contact(self, contact, bbox):
        # bbox: (x1,y1,x2,y2)  -> x: width, y: height
        # contact: (y,x) -> y: height, x: width
        # this method will get the relative point of the contact wrt the crop
        # then, convert that to its original location
        # Tx, Ty -> x1+Tx', y1+Ty' oluyor aslinda

        # contact is with respect to the resized image (224,224)
        # should move that back to how it would be within bbox
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        x_scale = 224 / bbox_width
        y_scale = 224 / bbox_height

        contact_x = contact[:, 0] / x_scale
        contact_y = contact[:, 1] / y_scale
        # print(f"contact_x: {contact_x}, contact_y: {contact_y}")

        # import pdb

        # pdb.set_trace()
        scaled_contact = torch.stack([contact_x, contact_y], dim=1)
        transformed_contact = scaled_contact + torch.Tensor([bbox[0], bbox[1]])

        print(f"bbox: {bbox}")
        print(f"contact: {contact}")
        print(f"transformed_contact: {transformed_contact}")
        return transformed_contact

    def evaluate(self):
        for batch in self.test_loader:

            img = batch[0]
            cropped_imgs, og_imgs, bboxes = self.crop_batch(img, batch[1])
            print(
                f"cropped_imgs.shape: {cropped_imgs.shape}, og_imgs.shape: {og_imgs.shape}"
            )
            cropped_imgs = cropped_imgs.to(self.device)
            task_description = batch[1]

            pred_mu, _, pred_grasp_rotation, pred_grasp_pose = self.forward(
                cropped_imgs, task_description
            )
            pred_joint_pose = self.get_gt_mano_output(
                pred_grasp_rotation, pred_grasp_pose
            ).cpu()

            for test_id in range(len(img)):

                # if self.cfg.crop_image:
                #     img[test_id] = self.crop_image(
                #         img[test_id], task_description[test_id], test_id
                #     )

                transformed_contact = self.detransform_contact(
                    pred_mu[test_id], bboxes[test_id]
                )
                contact_img = self.plot_contact(
                    img=og_imgs[test_id],
                    pred_contact=transformed_contact,
                    task_description=task_description[test_id],
                )
                fig, ax = self.plot_grasp(pred_grasp=pred_joint_pose[test_id])
                fig, ax = self.plot_rot(
                    fig,
                    ax,
                    pred_rot=pred_grasp_rotation[test_id].cpu(),
                    origin=pred_joint_pose[test_id][0],
                    task_description=task_description[test_id],
                )

                # Example matplotlib figure
                fig_img = fig_to_img(fig)
                plt.close(fig)

                # Example OpenCV image
                cv_img = contact_img
                # Resize images to the same height
                height = min(fig_img.shape[0], cv_img.shape[0])
                fig_img_resized = cv2.resize(
                    fig_img, (int(fig_img.shape[1] * height / fig_img.shape[0]), height)
                )
                cv_img_resized = cv2.resize(
                    cv_img, (int(cv_img.shape[1] * height / cv_img.shape[0]), height)
                )

                # Stack horizontally
                combined = np.hstack((fig_img_resized, cv_img_resized))

                # Save result
                cv2.imwrite(
                    f"{self.dump_dir}/combined_output_{'_'.join(task_description[test_id].split(' '))}.jpg",
                    combined,
                )

                # break

            break


if __name__ == "__main__":
    import time

    timestamp = time.time()
    time_local = time.localtime(timestamp)
    string_local = time.strftime("%m-%d-%H-%M-%S", time_local)
    evaluator = OOD_Evaluator(
        ood_dset_dir="/home/irmak/Workspace/nyu-big-data-and-ml/project/dex_grasp_eval_dataset",
        checkpoint_path="/home/irmak/Workspace/nyu-big-data-and-ml/project/checkpoints/grasp_dex_05-05_23:43:32/model_best.pth",
        dump_dir=f"eval_results/ood_evals/{string_local}",
        device="cuda",
        use_clip=False,
    )
    evaluator.evaluate()
