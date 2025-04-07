import os
import pickle
import random
from pathlib import Path

import cv2
import numpy as np
from dex_grasp.utils.video_recorder import VideoRecorder


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_img_and_traj(file_path):
    data = load_pickle(file_path)
    img = data["img"]
    traj = data["traj"]
    obj = data["contact_object_det"]
    return img, traj, obj


def print_keys(file_path):
    data = load_pickle(file_path)
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(key, value.shape)
        else:
            print(key, type(value))
            if isinstance(value, dict):
                print(f"key {key} holds dict")
                for k, v in value.items():
                    print(k)

            if isinstance(value, list):
                print(key, len(value))

            else:
                print(key, value)

        print("-" * 100)


def visualize_traj_and_bbox(img, traj, obj, save_dir, file_name):
    """
    Visualizes trajectory and bounding box on image frames
    Args:
        img: RGB image of shape (H,W,3)
        traj: Dictionary containing trajectory points of shape (N,2)
        obj: Dictionary with bbox coordinates and label
        save_dir: Directory to save visualization frames
    """

    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    video_recorder = VideoRecorder(save_dir=save_dir, resize_and_transpose=False)

    # Get bbox coordinates
    x1, y1, x2, y2 = [int(x) for x in obj["bbox"]]

    # Create copy of image for drawing
    img_copy = img.copy()

    # Draw bounding box
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Get trajectory points
    points = traj["RIGHT"]["traj"]

    # Draw trajectory points one by one
    for i, (x, y) in enumerate(points):
        # Convert to int coordinates
        x, y = int(x), int(y)

        # Draw point
        img_copy = cv2.circle(
            img_copy, (x, y), radius=3, color=(0, 0, 255), thickness=-1
        )

        # Save frame
        # cv2.imwrite(
        #     os.path.join(save_dir, f"frame_{i:03d}.jpg"),
        #     cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR),
        # )
        video_recorder.record(img_copy)

    video_recorder.save(file_name)


def plot_random_files(pkl_dir, save_dir, num_files=20):
    pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith(".pkl")]

    # Randomly select 20 files
    selected_files = random.sample(pkl_files, num_files)

    # Process each selected file
    for i, pkl_file in enumerate(selected_files):
        file_path = os.path.join(pkl_dir, pkl_file)

        # Load and visualize
        img, traj, obj = load_img_and_traj(file_path)
        print(f"Processing file {i+1}/{num_files}: {pkl_file} | Object: {obj}")
        visualize_traj_and_bbox(
            img,
            traj,
            obj,
            save_dir=save_dir,
            file_name=f"visualization_{i+1}.mp4",
        )


if __name__ == "__main__":
    random.seed(42)
    # file_path = "/data_ssd/irmak/deft-data-all/ego4d-r3m/labels_obj_bbox/37205_f97b9bb6-9afc-4743-899f-bfe483e88d97.pkl"
    # Get list of all pkl files in directory
    pkl_dir = "/data_ssd/irmak/deft-data-all/ego4d-r3m/labels_obj_bbox"
    # print_keys(pkl_dir)
    plot_random_files(pkl_dir, save_dir="ego4d-r3m_viz", num_files=20)
