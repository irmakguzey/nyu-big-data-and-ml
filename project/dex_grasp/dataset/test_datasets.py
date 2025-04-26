# This script will go through all the files in a given directory and check if the file is a valid grasp dataset

import glob
import os
import pickle

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torchvision import transforms

transform = transforms.Compose(  # NOTE: This is basically used for debugging!
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ]
)


def get_pkl_files(pkl_dir):
    return glob.glob(os.path.join(pkl_dir, "*.pkl"))


def load_pkl(pkl_file):
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    if "right_hand" in data["contact_grasp"][0]:
        hand_pose = data["contact_grasp"][0]["right_hand"]["pred_hand_pose"]
    elif "left_hand" in data["contact_grasp"][0]:
        hand_pose = data["contact_grasp"][0]["left_hand"]["pred_hand_pose"]

    if len(hand_pose.shape) == 1:
        hand_pose = np.expand_dims(hand_pose, axis=0)

    contact_points = torch.FloatTensor(data["contact"])[:5]
    if len(contact_points) < 5:
        # Append the last element until we have 5 points
        last_point = contact_points[-1].unsqueeze(0)
        while len(contact_points) < 5:
            contact_points = torch.cat([contact_points, last_point], dim=0)

    org_image = transform(data["img"])
    grasp_rotation = torch.FloatTensor(R.from_matrix(data["H"]).as_rotvec())

    assert org_image.shape == (3, 224, 224), f"IMG: {org_image.shape} | {pkl_file}"
    assert grasp_rotation.shape == (
        3,
    ), f"GRASP_ROT: {grasp_rotation.shape} | {pkl_file}"
    assert contact_points.shape == (
        5,
        3,
    ), f"CONTACT_POINTS: {contact_points.shape} | {pkl_file}"
    assert hand_pose.shape == (1, 48), f"HAND_POSE: {hand_pose.shape} | {pkl_file}"

    # print(
    #     f"IMG: {org_image.shape} | GRASP_ROT: {grasp_rotation.shape} | CONTACT_POINTS: {contact_points.shape} | HAND_POSE: {hand_pose.shape}"
    # )


if __name__ == "__main__":
    # pkl_dir = "/data_ssd/irmak/deft-data-all/ek100/labels_obj_bbox"  # 256, 456 - 1,48
    # pkl_dir = "/data_ssd/irmak/deft-data-all/hoi4d/labels"  # 270,480 - 48
    pkl_dir = (
        "/data_ssd/irmak/deft-data-all/ego4d-r3m/labels_obj_bbox"  # 405,540 - 1,48
    )
    pkl_files = get_pkl_files(pkl_dir)

    for pkl_file in pkl_files:
        try:
            load_pkl(pkl_file)
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
