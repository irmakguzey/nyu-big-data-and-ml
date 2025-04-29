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
        self.transform = (
            transforms.Compose(  # NOTE: This is basically used for debugging!
                [
                    transforms.ToTensor(),
                    transforms.Lambda(center_crop_square),
                    transforms.Resize((224, 224)),
                ]
            )
        )

    def _transform_contact_point(self, contact_point, img):
        if isinstance(img, torch.Tensor):
            _, h, w = img.shape  # (C, H, W)
        elif isinstance(img, np.ndarray):
            h, w = img.shape[:2]

        crop_size = min(h, w)
        resize_size = (224, 224)

        """
        contact_point: (batch_size, 2) torch tensor
        Returns transformed points: (batch_size, 2) torch tensor
        """
        orig_h, orig_w = h, w
        crop_h, crop_w = crop_size, crop_size
        resize_h, resize_w = resize_size

        crop_x_start = (orig_w - crop_w) / 2
        crop_y_start = (orig_h - crop_h) / 2

        scale_x = resize_w / crop_w
        scale_y = resize_h / crop_h

        y = contact_point[:, 1]
        x = contact_point[:, 0]

        x_cropped = x - crop_x_start
        y_cropped = y - crop_y_start

        x_resized = x_cropped * scale_x
        y_resized = y_cropped * scale_y

        transformed_points = torch.stack([x_resized, y_resized], dim=-1)

        return transformed_points

    def _crop_image(self, org_image, bbox):
        if self.return_cropped_image:
            y1, x1, y2, x2 = [int(x) for x in bbox]
            bbox_offset = 20
            y1, x1, y2, x2 = (
                int(y1) - bbox_offset,
                int(x1) - bbox_offset,
                int(y2) + bbox_offset,
                int(x2) + bbox_offset,
            )
            if y1 < 0:
                y1 = 0
            if x1 < 0:
                x1 = 0
            if y2 > org_image.shape[0]:
                y2 = org_image.shape[0]
            if x2 > org_image.shape[1]:
                x2 = org_image.shape[1]

            width = y2 - y1
            height = x2 - x1

            diff = width - height
            if width > height:
                y1 += int(diff / np.random.uniform(1.5, 2.5))
                y2 -= int((diff / (np.random.uniform(1.5, 2.5) + diff % 2)))
            else:
                diff = height - width
                x1 += int(diff / np.random.uniform(1.5, 2.5))
                x2 -= int((diff / (np.random.uniform(1.5, 2.5) + diff % 2)))

            # Crop the image using the bounding box coordinates
            cropped_image = org_image[x1:x2, y1:y2]
            return cropped_image
        else:
            return org_image

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
        bbox = data["contact_object_det"]["bbox"]
        # print(data["contact_object_det"].keys())
        if "label" in data["contact_object_det"]:
            object_label = data["contact_object_det"]["label"]
        else:
            object_label = data["contact_object_det"]["class"]
        cropped_image = self._crop_image(org_image, bbox)

        # Get contact points
        contact_points = torch.FloatTensor(data["contact"])[:5]
        if len(contact_points) < 5:
            # Append the last element until we have 5 points
            last_point = contact_points[-1].unsqueeze(0)
            while len(contact_points) < 5:
                contact_points = torch.cat([contact_points, last_point], dim=0)

        if self.transform_contact:
            contact_points = self._transform_contact_point(
                contact_points, cropped_image
            )

        # NOTE: Not sure if i want to return the original image as well
        image = self.transform(cropped_image).clamp(
            0, 1
        )  # Should clamp it to input to clip preprocessor
        grasp_rotation = torch.FloatTensor(R.from_matrix(data["H"]).as_rotvec())
        task_description = data["narration"]
        hand_pose = torch.FloatTensor(hand_pose)[0]

        return (
            image,
            contact_points,
            hand_pose[:3],
            hand_pose[3:],
            object_label,
        )

    def __getitem__(self, idx):
        pkl_path = self.pkl_files[idx]
        pkl_data = self.load_pkl(pkl_path)
        return pkl_data

    def __len__(self):
        return len(self.pkl_files)


if __name__ == "__main__":
    dataset = GraspDataset(
        pkl_dir="/data_ssd/irmak/deft-data-all/ego4d-r3m/labels_obj_bbox",
        return_cropped_image=False,
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(len(dataset))
    batch = next(iter(dataloader))

    # Visualize the image
    # image = batch[0][0, :, :]
    # task_description = batch[5][0]
    # obj_description = batch[6][0]
    # print(image.shape)
    # # Convert to numpy and ensure correct shape
    # image_np = image.numpy()
    # # If image is in range [0,1], scale to [0,255]
    # if image_np.max() <= 1.0:
    #     image_np = (image_np * 255).astype(np.uint8)
    # else:
    #     image_np = image_np.astype(np.uint8)
    # # Ensure correct shape for PIL
    # if len(image_np.shape) == 3 and image_np.shape[0] == 3:  # CHW format
    #     image_np = np.transpose(image_np, (1, 2, 0))  # Convert to HWC
    # image_pil = Image.fromarray(image_np)
    # image_pil.save(f"debug_image_{task_description}_{obj_description}.png")
    # print(task_description)
    # import pdb

    # pdb.set_trace()
