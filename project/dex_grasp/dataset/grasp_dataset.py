# A dataset class to return mean contact point, grasp rotation and the handpose
import glob
import os
import pickle

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor


class GraspDataset(Dataset):
    def __init__(self, pkl_dir, transform=None):
        self.transform = transform
        self.pkl_files = glob.glob(os.path.join(pkl_dir, "*.pkl"))

        # Initialize the processor with a specific model
        # NOTE: We will be using Dinov2-base as our image processor / and encoder
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.transform = (
            transforms.Compose(  # NOTE: This is basically used for debugging!
                [
                    transforms.ToTensor(),
                    transforms.Resize((224, 224)),
                ]
            )
        )

    def load_pkl(self, pkl_file_path):
        with open(pkl_file_path, "rb") as f:
            data = pickle.load(f)

        # NOTE: For now we're priotizing the right hand
        if "right_hand" in data["contact_grasp"][0]:
            hand_pose = data["contact_grasp"][0]["right_hand"]["pred_hand_pose"]
        elif "left_hand" in data["contact_grasp"][0]:
            hand_pose = data["contact_grasp"][0]["left_hand"]["pred_hand_pose"]
        else:
            raise ValueError("No hand pose found in the pkl file")

        org_image = data["img"]
        # Get bounding box coordinates
        bbox = data["contact_object_det"]["bbox"]
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

        Image.fromarray(cropped_image).save(
            f"debug_image_{data['contact_object_det']['label']}_{data['narration']}.png"
        )

        # NOTE: Not sure if i want to return the original image as well
        pkl_data = dict(
            processed_image=self.processor(
                torch.tensor(cropped_image), return_tensors="pt"
            ).pixel_values[0],
            image=self.transform(cropped_image),
            # image_pil=Image.fromarray(cropped_image),
            mean_contact_points=torch.mean(torch.FloatTensor(data["contact"]), dim=0),
            grasp_rotation=torch.FloatTensor(R.from_matrix(data["H"]).as_rotvec()),
            task_description=data["narration"],
            obj_description=data["contact_object_det"]["label"],
            hand_pose=torch.FloatTensor(hand_pose)[0],
        )

        return (
            pkl_data["image"],
            pkl_data["processed_image"],
            pkl_data["mean_contact_points"],
            pkl_data["grasp_rotation"],
            pkl_data["hand_pose"],
            pkl_data["task_description"],
            pkl_data["obj_description"],
        )

    def __getitem__(self, idx):
        pkl_path = self.pkl_files[idx]
        pkl_data = self.load_pkl(pkl_path)
        return pkl_data

    def __len__(self):
        return len(self.pkl_files)


if __name__ == "__main__":
    dataset = GraspDataset(
        pkl_dir="/data_ssd/irmak/deft-data-all/ego4d-r3m/labels_obj_bbox"
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(len(dataset))
    batch = next(iter(dataloader))

    # Visualize the image
    image = batch[0][0, :, :]
    task_description = batch[5][0]
    obj_description = batch[6][0]
    print(image.shape)
    # Convert to numpy and ensure correct shape
    image_np = image.numpy()
    # If image is in range [0,1], scale to [0,255]
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)
    # Ensure correct shape for PIL
    if len(image_np.shape) == 3 and image_np.shape[0] == 3:  # CHW format
        image_np = np.transpose(image_np, (1, 2, 0))  # Convert to HWC
    image_pil = Image.fromarray(image_np)
    # image_pil.save(f"debug_image_{task_description}_{obj_description}.png")
    # print(task_description)
    # import pdb

    # pdb.set_trace()
