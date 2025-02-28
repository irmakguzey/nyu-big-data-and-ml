import glob
import os

import numpy as np
import pdfplumber
import torch.utils.data as data
from datasets import CausalLMDataset

# from PyPDF2 import PdfReader
from tqdm import tqdm
from transformers import AutoTokenizer


def preprocess_single_file(file_path):
    # It'll dump a txt file to the same location with the same name but with txt extension
    file_name = file_path.split("/")[-1].split(".")[0]
    file_root = "/".join(file_path.split("/")[:-1])
    txt_path = f"{file_root}/{file_name}.txt"

    if os.path.exists(txt_path):
        return

    with pdfplumber.open(file_path) as pdf:
        text = "\n".join(
            page.extract_text() for page in pdf.pages if page.extract_text()
        )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)


def get_datasets(root_dir, train_test_split=0.9, model_name="gpt-2"):
    # get the di`rectory of all pdf files
    all_file_paths = glob.glob(f"{root_dir}/*.pdf")

    # If not preprocessed preprocess all the files
    pbar = tqdm(total=len(all_file_paths))
    for file_path in all_file_paths:
        pbar.set_description(f"Preprocessing file {file_path.split('/')[-1]}")
        preprocess_single_file(file_path)
        pbar.update(1)
    pbar.close()

    # Shuffle and split the file paths
    np.random.shuffle(all_file_paths)
    split_idx = int(train_test_split * len(all_file_paths))
    train_files, test_files = all_file_paths[:split_idx], all_file_paths[split_idx:]
    print(len(train_files), len(test_files))

    # Initialize all datasets
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_datasets, test_datasets = [], []
    for file_path in train_files:
        train_datasets.append(
            CausalLMDataset(file_path=file_path, tokenizer=tokenizer, max_length=512)
        )
    train_dset = data.ConcatDataset(train_datasets)

    for file_path in test_files:
        test_datasets.append(
            CausalLMDataset(file_path=file_path, tokenizer=tokenizer, max_length=512)
        )
    test_dset = data.ConcatDataset(test_datasets)

    return train_dset, test_dset


if __name__ == "__main__":

    train_dset, test_dset = get_datasets(
        root_dir="climate_text_dataset", model_name="gpt2"
    )
    print(f"train_dset len: {len(train_dset)}")
    print(f"len(test_dset): {len(test_dset)}")
