import torch
from dex_grasp.dataset.grasp_dataset import GraspDataset
from torch.utils.data import ConcatDataset, DataLoader, random_split


def get_dataloaders(
    batch_size=32, num_workers=32, train_dset_split=0.8, crop_image=False
):
    pkl_dirs = [
        "/data_ssd/irmak/deft-data-all/ego4d-r3m/labels_obj_bbox",
        "/data_ssd/irmak/deft-data-all/ego4d-sta/labels_obj_bbox",
        # "/data_ssd/irmak/deft-data-all/ek100/labels_obj_bbox",
        # "/data_ssd/irmak/deft-data-all/hoi4d/labels",
    ]  # TODO: Fix this!!

    dsets = []
    for pkl_dir in pkl_dirs:
        dsets.append(
            GraspDataset(
                pkl_dir=pkl_dir,
                return_cropped_image=crop_image,
            )
        )

    dataset = ConcatDataset(dsets)

    train_dset_size = int(len(dataset) * train_dset_split)
    test_dset_size = len(dataset) - train_dset_size

    # Random split the train and validation datasets
    train_dset, test_dset = random_split(
        dataset,
        [train_dset_size, test_dset_size],
        generator=torch.Generator().manual_seed(45),
    )

    # Initialize the dataloader
    train_dataloader = DataLoader(
        train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_dataloader, test_dataloader


if __name__ == "__main__":
    dataloader = get_dataloader()
    for batch in dataloader:
        print(batch)
