import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from dex_grasp.dataset.get_dataloaders import get_dataloaders
from dex_grasp.utils.visualization import draw_axis, plot_hand_pose_3d
from manotorch.manolayer import ManoLayer, MANOOutput
from scipy.spatial.transform import Rotation as R


class Evaluator:
    def __init__(self, dump_dir, device):
        self.device = device
        self.dump_dir = dump_dir

        _, self.test_loader = get_dataloaders(
            batch_size=32, num_workers=32, train_dset_split=0.8, crop_image=False
        )

        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

        self.skeleton_parents = [
            -1,  # 0: wrist
            0,  # 1: thumb base
            1,  # 2: thumb mid
            2,  # 3: thumb tip
            0,  # 4: index base
            4,  # 5: index mid
            5,  # 6: index tip
            0,  # 7: middle base
            7,  # 8: middle mid
            8,  # 9: middle tip
            0,  # 10: ring base
            10,  # 11: ring mid
            11,  # 12: ring tip
            0,  # 13: pinky base
            13,  # 14: pinky mid
            14,  # 15: pinky tip
        ]

        # Rest pose bone directions (example values)
        self.rest_bone_dirs = np.array(
            [
                [0.0, 0.0, 0.0],  # wrist
                [-0.1, 0.2, 0.0],  # thumb base
                [-0.1, 0.1, 0.0],  # thumb mid
                [0.1, 0.1, 0.0],  # thumb tip
                [-0.05, 0.2, 0.0],  # index base
                [-0.05, 0.1, 0.0],  # index mid
                [-0.05, 0.1, 0.0],  # index tip
                [0.0, 0.1, 0.0],  # middle base
                [0.0, 0.2, 0.0],  # middle mid
                [0.0, 0.3, 0.0],  # middle tip
                [-0.05, 0.2, 0.0],  # ring base
                [-0.05, 0.1, 0.0],  # ring mid
                [-0.05, 0.1, 0.0],  # ring tip
                [-0.1, 0.2, 0.0],  # pinky base
                [-0.05, 0.1, 0.0],  # pinky mid
                [-0.05, 0.1, 0.0],  # pinky tip
            ],
            dtype=np.float32,
        )

    def _load_model(self):
        pass

    # def plot_contact(self, img, pred_contact=None, gt_contact=None):

    def plot_contact(self, img, pred_contact=None, gt_contact=None):
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
        cv2.imwrite(
            f"{self.dump_dir}/contact_points.png",
            cv2.cvtColor(img_with_points, cv2.COLOR_RGB2BGR),
        )

        return img_with_points

    def plot_grasp(self, img, pred_grasp=None, gt_grasp=None):

        if pred_grasp is not None:
            pred_grasp = pred_grasp.cpu().numpy()
            for grasp in pred_grasp:
                x, y = int(grasp[0]), int(grasp[1])
                cv2.circle(img, (x, y), radius=5, color=(255, 0, 0), thickness=-1)

        if gt_grasp is not None:
            # print(gt_grasp)

            fig, ax = plot_hand_pose_3d(gt_grasp)
            fig.savefig(os.path.join(self.dump_dir, f"gt_grasp.png"))

            # Convert the figure to a numpy array
            fig.canvas.draw()
            plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # Resize plot_img to match img height
            height = img.shape[0]
            plot_img = cv2.resize(
                plot_img, (int(plot_img.shape[1] * height / plot_img.shape[0]), height)
            )

            # Stack images horizontally
            img = np.hstack((img, plot_img))

            # Save the combined image
            cv2.imwrite(os.path.join(self.dump_dir, f"combined_img.png"), img)

        return img

    def plot_rot(self, img, pred_pose=None, gt_pose=None):
        # I think the rotation is the rotvec of the gt_pose (first 3 elements)
        gt_rotvec = gt_pose[:3]
        gt_rotmat = R.from_rotvec(gt_rotvec).as_matrix()

        # from dataclasses import dataclass
        # @dataclass
        # class Intrinsics:
        #     F_X: float
        #     F_Y: float
        #     C_X: float
        #     C_Y: float
        #     W: int
        #     H: int
        #     D: np.ndarray

        #     @property
        #     def K(self) -> np.ndarray:
        #         return np.array(
        #             [[self.F_X, 0, self.C_X], [0, self.F_Y, self.C_Y], [0, 0, 1]],
        #             dtype=np.float64,
        #         )

        # intrinsics = Intrinsics(
        #     F_X=706.0837807109395,
        #     F_Y=705.9212493046017,
        #     C_X=491.093931534625,
        #     C_Y=372.3078286475583,
        #     W=960,
        #     H=720,
        #     D=np.array(
        #         [0.19482301, -0.86972093, 0.00588824, 0.0052822, 1.25992676],
        #         dtype=np.float32,
        #     ),
        # )

        # img = draw_axis(img, gt_rotmat, intrinsics, length=0.1)

    def evaluate(self):
        for batch in self.test_loader:

            img = batch[0].to(self.device)
            task_description = batch[4]
            gt_mu = batch[1].to(self.device)[:, :, :2]
            gt_grasp_rotation = batch[2].to(self.device)
            gt_grasp_pose = batch[3]

            # initialize layers
            ncomps = 45
            mano_layer = ManoLayer(
                use_pca=True,
                flat_hand_mean=False,
                ncomps=ncomps,
                mano_assets_root="/home/irmak/Workspace/nyu-big-data-and-ml/project/submodules/manotorch/assets/mano",
            )

            # batch_size = 2
            # Generate random shape parameters
            random_shape = torch.rand(gt_grasp_pose.shape[0], 10)
            # Generate random pose parameters, including 3 values for global axis-angle rotation
            # random_pose = torch.rand(gt_grasp_pose.shape[0], 3 + ncomps)

            mano_output: MANOOutput = mano_layer(gt_grasp_pose, random_shape)
            joints = mano_output.joints  # (B, 21, 3), root relative
            test_id = 0

            img = self.plot_contact(img[test_id], None, gt_mu[test_id])
            img = self.plot_grasp(img, None, joints[test_id])

            break


# Example usage:
# pose_axis_angle = np.random.randn(16, 3)  # Random pose for test
# joint_positions = forward_kinematics(pose_axis_angle, skeleton_parents, rest_bone_dirs)


if __name__ == "__main__":
    evaluator = Evaluator(dump_dir="eval_results", device="cuda")
    evaluator.evaluate()
