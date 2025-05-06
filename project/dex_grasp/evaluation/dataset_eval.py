import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from dex_grasp.dataset.get_dataloaders import get_dataloaders
from dex_grasp.utils.model import load_model
from dex_grasp.utils.visualization import (
    draw_text,
    fig_to_img,
    plot_contact,
    plot_hand_pose_3d,
    plot_rotation,
)
from manotorch.manolayer import ManoLayer


class Evaluator:
    def __init__(self, checkpoint_path, dump_dir, device, use_clip=True):
        self.device = device
        self.dump_dir = dump_dir
        self.use_clip = use_clip
        self.checkpoint_path = checkpoint_path
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

        if os.path.exists(f"{checkpoint_dir}/config.txt"):
            # Copy config file to dump directory
            print(f"Copying config file to {dump_dir}")
            shutil.copy2(f"{checkpoint_dir}/config.txt", f"{dump_dir}/config.txt")

        self._load_data()
        self._load_model()

    def _load_data(self):
        _, self.test_loader = get_dataloaders(
            batch_size=32, num_workers=32, train_dset_split=0.8, crop_image=False
        )

    def _load_model(self):
        self.affordance_model, self.grasp_transformer = load_model(
            device=self.device,
            checkpoint_path=self.checkpoint_path,
            use_clip=self.use_clip,
            freeze_rep=True,
        )
        self.affordance_model.to(self.device)
        self.grasp_transformer.to(self.device)

        self.mano_layer = ManoLayer(
            use_pca=True,
            flat_hand_mean=False,
            ncomps=45,
            mano_assets_root="/home/irmak/Workspace/nyu-big-data-and-ml/project/submodules/manotorch/assets/mano",
        ).to(self.device)

    def forward(self, img, task_description):
        with torch.no_grad():
            img_feat, text_feat = self.affordance_model.get_clip_features(
                img, task_description
            )
            if self.use_clip:
                mu, cvar = self.affordance_model.get_mu_cvar(img_feat=img_feat)
            else:
                img_feat = self.affordance_model.get_resnet_features(img)
                mu, cvar = self.affordance_model.get_mu_cvar(img=img)
            grasp_rotation, grasp_pose = self.grasp_transformer(text_feat, img_feat)

        return mu.cpu(), cvar, grasp_rotation, grasp_pose

    def plot_contact(
        self, img, pred_contact=None, gt_contact=None, task_description=None
    ):

        return plot_contact(img, pred_contact, gt_contact, task_description)

    def plot_grasp(self, pred_grasp=None, gt_grasp=None):
        fig, ax = None, None
        if pred_grasp is not None:
            fig, ax = plot_hand_pose_3d(pred_grasp, linestyle="dashed", fig=fig, ax=ax)

        if gt_grasp is not None:
            fig, ax = plot_hand_pose_3d(gt_grasp, fig=fig, ax=ax)

        return fig, ax

    def plot_rot(
        self,
        fig,
        ax,
        pred_rot=None,
        gt_rot=None,
        origin=np.zeros(3),
        task_description=None,
    ):
        if pred_rot is not None:
            fig, ax = plot_rotation(
                fig=fig, ax=ax, rotation=pred_rot, origin=origin, linestyle="dashed"
            )

        if gt_rot is not None:
            fig, ax = plot_rotation(fig=fig, ax=ax, rotation=gt_rot, origin=origin)

        return fig, ax

    def get_gt_mano_output(self, gt_grasp_rotation, gt_grasp_pose):
        random_shape = torch.rand(gt_grasp_pose.shape[0], 10).to(self.device)
        mano_pose = torch.cat([gt_grasp_rotation, gt_grasp_pose], dim=-1)
        mano_output = self.mano_layer(mano_pose, random_shape)
        joints = mano_output.joints  # (B, 21, 3), root relative

        return joints.cpu()

    def evaluate(self):
        for batch in self.test_loader:

            img = batch[0].to(self.device)
            task_description = batch[4]
            gt_mu = batch[1].to(self.device)[:, :, :2]
            gt_grasp_rotation = batch[2].to(self.device)
            gt_grasp_pose = batch[3].to(self.device)

            pred_mu, _, pred_grasp_rotation, pred_grasp_pose = self.forward(
                img, task_description
            )
            joint_pose = self.get_gt_mano_output(gt_grasp_rotation, gt_grasp_pose).cpu()
            pred_joint_pose = self.get_gt_mano_output(
                gt_grasp_rotation, pred_grasp_pose
            ).cpu()

            for test_id in range(32):

                contact_img = self.plot_contact(
                    img=img[test_id],
                    pred_contact=pred_mu[test_id],
                    gt_contact=gt_mu[test_id],
                    task_description=task_description[test_id],
                )
                fig, ax = self.plot_grasp(
                    pred_grasp=pred_joint_pose[test_id], gt_grasp=joint_pose[test_id]
                )
                fig, ax = self.plot_rot(
                    fig,
                    ax,
                    pred_rot=pred_grasp_rotation[test_id].cpu(),
                    gt_rot=gt_grasp_rotation[test_id].cpu(),
                    origin=joint_pose[test_id][0],
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
            break


if __name__ == "__main__":
    import time

    timestamp = time.time()
    time_local = time.localtime(timestamp)
    string_local = time.strftime("%m-%d-%H-%M-%S", time_local)
    evaluator = Evaluator(
        checkpoint_path="/home/irmak/Workspace/nyu-big-data-and-ml/project/checkpoints/grasp_dex_05-01_17:59:11/model_best.pth",
        dump_dir=f"eval_results/dataset_evals/{string_local}",
        device="cuda",
        use_clip=False,
    )
    evaluator.evaluate()
