# Given a directory with images and text prompts, generate a dataset with pkl files

import glob
import hashlib
import os
import pickle

import cv2


def generate_single_pkl(img_path, save_dir):
    # Load the image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get the text prompt from the path
    img_name = img_path.split("/")[-1]
    all_underscores = img_name.split("_")
    text_prompt_el = []
    for el in all_underscores:
        if "IMG" in el or el.isdigit():
            continue
        else:
            if "jpg" in el or "png" in el:
                clean_el = el.split(".")[0]
                text_prompt_el.append(clean_el)
            else:
                text_prompt_el.append(el)
    text_prompt = " ".join(text_prompt_el)
    print(text_prompt)

    # Save the image and text prompt to a pkl file
    os.makedirs(save_dir, exist_ok=True)

    # Create a hash of the image name and text prompt to ensure unique filenames
    hash_input = f"{img_name}_{text_prompt}".encode("utf-8")
    hash_object = hashlib.md5(hash_input)
    hash_filename = hash_object.hexdigest() + ".pkl"

    with open(os.path.join(save_dir, hash_filename), "wb") as f:
        pickle.dump({"img": img, "text": text_prompt}, f)


def generate_dataset(img_dir, save_dir):
    # Get all the images in the directory
    img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
    img_paths.extend(glob.glob(os.path.join(img_dir, "*.png")))

    for img_path in img_paths:
        generate_single_pkl(img_path, save_dir)


if __name__ == "__main__":
    generate_dataset(
        "/home/irmak/Workspace/nyu-big-data-and-ml/project/dex-grasp-ood-dataset-2",
        "/home/irmak/Workspace/nyu-big-data-and-ml/project/dex_grasp_eval_dataset_2",
    )
