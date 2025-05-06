# TODO: Implement the OOD evaluation - this will take some example images and check if the model is able to detect poses

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from dex_grasp.dataset.get_dataloaders import get_eval_dataloader
from dex_grasp.evaluation.dataset_eval import Evaluator
from dex_grasp.utils.visualization import fig_to_img


class OOD_Evaluator(Evaluator):

    def __init__(self, ood_dset_dir, **kwargs):
        self.ood_dset_dir = ood_dset_dir
        super().__init__(**kwargs)

    def _load_data(self):
        self.test_loader = get_eval_dataloader(batch_size=32, num_workers=4)

    def evaluate(self):
        for batch in self.test_loader:

            img = batch[0].to(self.device)
            task_description = batch[1]

            pred_mu, _, pred_grasp_rotation, pred_grasp_pose = self.forward(
                img, task_description
            )
            pred_joint_pose = self.get_gt_mano_output(
                pred_grasp_rotation, pred_grasp_pose
            ).cpu()

            # for test_id in range(len(img)):

            #     self.plot_contact(
            #         img=img[test_id],
            #         pred_contact=pred_mu[test_id],
            #         task_description=task_description[test_id],
            #     )
            #     fig, ax = self.plot_grasp(pred_grasp=pred_joint_pose[test_id])
            #     fig, ax = self.plot_rot(
            #         fig,
            #         ax,
            #         pred_rot=pred_grasp_rotation[test_id].cpu(),
            #         origin=pred_joint_pose[test_id][0],
            #         task_description=task_description[test_id],
            #     )

            # # break

            for test_id in range(len(img)):

                contact_img = self.plot_contact(
                    img=img[test_id],
                    pred_contact=pred_mu[test_id],
                    task_description=task_description[test_id],
                )
                fig, ax = self.plot_grasp(pred_grasp=pred_joint_pose[test_id])
                fig, ax = self.plot_rot(
                    fig,
                    ax,
                    pred_rot=pred_grasp_rotation[test_id].cpu(),
                    origin=pred_joint_pose[test_id][0],
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
    evaluator = OOD_Evaluator(
        ood_dset_dir="/home/irmak/Workspace/nyu-big-data-and-ml/project/dex_grasp_eval_dataset",
        checkpoint_path="/home/irmak/Workspace/nyu-big-data-and-ml/project/checkpoints/grasp_dex_05-01_18:02:04/model_best.pth",
        dump_dir=f"eval_results/ood_evals/{string_local}",
        device="cuda",
        use_clip=False,
    )
    evaluator.evaluate()
