import torch
from dex_grasp.dataset.grasp_dataset import DexGraspEvalDataset, GraspDataset
from torch.utils.data import ConcatDataset, DataLoader, random_split


def get_dataloaders(
    batch_size=32, num_workers=32, train_dset_split=0.8, crop_image=False
):
    pkl_dirs = [
        "/data_ssd/irmak/deft-data-all/ego4d-r3m/labels_obj_bbox",
        "/data_ssd/irmak/deft-data-all/ego4d-sta/labels_obj_bbox",
        "/data_ssd/irmak/deft-data-all/ek100/labels_obj_bbox",
        "/data_ssd/irmak/deft-data-all/hoi4d/labels",
    ]

    dsets = []
    for pkl_dir in pkl_dirs:
        dsets.append(
            GraspDataset(
                pkl_dir=pkl_dir,
                return_cropped_image=crop_image,
                transform_contact=True,
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


def get_eval_dataloader(batch_size=32, num_workers=32):
    pkl_dirs = [
        "/home/irmak/Workspace/nyu-big-data-and-ml/project/dex_grasp_eval_dataset"
    ]

    dsets = []
    for pkl_dir in pkl_dirs:
        dsets.append(
            DexGraspEvalDataset(
                pkl_dir=pkl_dir,
            )
        )

    dataset = ConcatDataset(dsets)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return dataloader


if __name__ == "__main__":
    import cv2
    from dex_grasp.utils.visualization import plot_bbox, plot_contact

    _, test_loader = get_dataloaders(
        batch_size=32, num_workers=32, train_dset_split=0.8, crop_image=True
    )

    for batch in test_loader:
        for i in range(len(batch[0])):
            img = batch[0][i]
            contact_points = batch[1][i]
            object_label = batch[4][i]
            bbox = batch[5][i]

            # print(img.shape, contact_points.shape, bbox.shape, object_label)

            img = plot_contact(
                img=img, gt_contact=contact_points, task_description=object_label
            )
            img = plot_bbox(img=img, bbox=bbox)

            cv2.imwrite(f"debug_image_{i}_crop_True.png", img)
            i += 1

        break
