import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from dex_grasp.dataset.get_dataloaders import get_dataloaders
from dex_grasp.utils.model import load_model
from dex_grasp.utils.visualization import draw_text, plot_hand_pose_3d, plot_rotation
from manotorch.manolayer import ManoLayer, MANOOutput


class Evaluator:
    def __init__(self, dump_dir, device, use_clip=True):
        self.device = device
        self.dump_dir = dump_dir
        self.use_clip = use_clip

        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

        self._load_data()
        self._load_model()

    def _load_data(self):
        _, self.test_loader = get_dataloaders(
            batch_size=32, num_workers=32, train_dset_split=0.8, crop_image=False
        )

    def _load_model(self):
        self.affordance_model, self.grasp_transformer = load_model(
            device=self.device,
            checkpoint_path="/home/irmak/Workspace/nyu-big-data-and-ml/project/checkpoints/grasp_dex_04-29_20:37:00/model_best.pth",
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
        # Convert image to numpy array if it's a tensor
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy().transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)

        # Create a copy of the image to draw on
        img_with_points = img.copy()

        # Plot ground truth contact points
        if gt_contact is not None:
            gt_contact = gt_contact.cpu().numpy()
            for point in gt_contact:
                x, y = int(point[0]), int(point[1])
                cv2.circle(
                    img_with_points, (x, y), radius=5, color=(0, 255, 0), thickness=-1
                )  # Green circles for GT

        # Plot predicted contact points
        if pred_contact is not None:
            pred_contact = pred_contact.cpu().numpy()
            for point in pred_contact:
                x, y = int(point[0]), int(point[1])
                cv2.circle(
                    img_with_points, (x, y), radius=5, color=(255, 0, 0), thickness=-1
                )  # Red circles for predictions

        # Save the image with contact points
        if task_description is not None:
            img_with_points = draw_text(
                img_with_points, task_description, position=(10, 30)
            )

        cv2.imwrite(
            f"{self.dump_dir}/contact_points_{'_'.join(task_description.split(' '))}.png",
            cv2.cvtColor(img_with_points, cv2.COLOR_RGB2BGR),
        )

        return img_with_points

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

        fig.savefig(
            os.path.join(
                self.dump_dir, f"gt_rot_{'_'.join(task_description.split(' '))}.png"
            ),
            bbox_inches="tight",
        )
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

                self.plot_contact(
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
            break


if __name__ == "__main__":
    import time

    timestamp = time.time()
    time_local = time.localtime(timestamp)
    string_local = time.strftime("%m-%d", time_local)
    evaluator = Evaluator(
        dump_dir=f"eval_results/dataset_evals/{string_local}",
        device="cuda",
        use_clip=False,
    )
    evaluator.evaluate()
