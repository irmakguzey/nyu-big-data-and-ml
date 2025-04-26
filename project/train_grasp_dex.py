import os
import time
from dataclasses import dataclass

import torch
from dex_grasp.dataset.get_dataloaders import get_dataloaders
from dex_grasp.models.affordance_model import AffordanceModel
from dex_grasp.models.grasp_transformer import GraspTransformer
from dex_grasp.utils.logger import Logger
from torch import nn
from tqdm import tqdm


@dataclass
class TrainingConfig:
    num_epochs: int = 500
    batch_size: int = 512
    num_workers: int = 32
    train_dset_split: float = 0.8
    lambda_m: float = 1e-5
    lambda_g: float = 5
    lambda_r: float = 0.1
    hidden_dim: int = 512
    crop_image: bool = False
    test_every_n_epochs: int = 10
    device: int = 1
    log: bool = True
    save_model: bool = True


class Trainer:
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.device = torch.device(
            f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu"
        )
        timestamp = time.time()
        time_local = time.localtime(timestamp)
        string_local = time.strftime("%m-%d_%H:%M:%S", time_local)
        self.wandb_exp_name = f"grasp_dex_{string_local}"
        if self.cfg.log:
            self.logger = Logger(self.wandb_exp_name, out_dir=".")

    def _init_models(self):
        self.affordance_model = AffordanceModel(
            src_in_features=self.cfg.hidden_dim, freeze_rep=True, device=self.device
        ).to(self.device)
        self.grasp_transformer = GraspTransformer(text_dim=512, image_dim=512).to(
            self.device
        )
        self.optimizer = torch.optim.AdamW(
            list(self.affordance_model.parameters())
            + list(self.grasp_transformer.parameters()),
            lr=1e-4,
        )

    def _init_dataloaders(self):
        self.train_dataloader, self.test_dataloader = get_dataloaders(
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            train_dset_split=self.cfg.train_dset_split,
            crop_image=self.cfg.crop_image,
        )

    def train_one_epoch(self, epoch):
        self.affordance_model.train()
        self.grasp_transformer.train()

        pbar = tqdm(total=len(self.train_dataloader))

        train_loss = 0
        for batch in self.train_dataloader:
            img = batch[0].to(self.device)
            task_description = batch[4]
            gt_mu = batch[1].to(self.device)[:, :, :2]
            gt_grasp_rotation = batch[2].to(self.device)
            gt_grasp_pose = batch[3].to(self.device)

            self.optimizer.zero_grad()

            img_feat, text_feat = self.affordance_model.get_clip_features(
                img, task_description
            )
            mu, cvar = self.affordance_model.get_mu_cvar(img_feat)
            grasp_rotation, grasp_pose = self.grasp_transformer(text_feat, img_feat)
            # print(f"img feat: {img_feat.shape}, text feat: {text_feat.shape}")
            # print(f"b grasp: {gt_grasp_pose.shape}, grasp: {grasp_pose.shape}")

            contact_loss = nn.functional.mse_loss(mu, gt_mu)
            grasp_rotation_loss = nn.functional.mse_loss(
                grasp_rotation, gt_grasp_rotation
            )
            grasp_pose_loss = nn.functional.mse_loss(grasp_pose, gt_grasp_pose)

            loss = (
                self.cfg.lambda_m * contact_loss
                + self.cfg.lambda_r * grasp_rotation_loss
                + self.cfg.lambda_g * grasp_pose_loss
            )
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            pbar.update(1)
            pbar.set_description(
                f"Epoch: {epoch} | Loss: {loss.item():.2f} | Contact: {self.cfg.lambda_m * contact_loss:.2f} | Rotation: {self.cfg.lambda_r * grasp_rotation_loss:.2f} | Pose: {self.cfg.lambda_g * grasp_pose_loss:.2f}"
            )

            if self.cfg.log:
                self.logger.log(
                    {
                        "train/loss/total": loss.item(),
                        "train/loss/contact": self.cfg.lambda_m * contact_loss.item(),
                        "train/loss/rotation": self.cfg.lambda_r
                        * grasp_rotation_loss.item(),
                        "train/loss/pose": self.cfg.lambda_g * grasp_pose_loss.item(),
                    }
                )

        pbar.close()
        train_loss /= len(self.train_dataloader)
        if self.cfg.log:
            self.logger.log({"train/loss/avg": train_loss})

        return train_loss

    def train(self):

        self._init_models()
        self._init_dataloaders()

        best_loss = float("inf")

        if not os.path.exists(f"checkpoints/{self.wandb_exp_name}"):
            os.makedirs(f"checkpoints/{self.wandb_exp_name}")

        self.save_model(f"checkpoints/{self.wandb_exp_name}", 0)
        for epoch in range(self.cfg.num_epochs):
            self.train_one_epoch(epoch)
            if epoch % self.cfg.test_every_n_epochs == 0:
                test_loss = self.test_one_epoch(epoch)
                if test_loss < best_loss:
                    best_loss = test_loss
                    self.save_model(f"checkpoints/{self.wandb_exp_name}", "best")
                self.save_model(f"checkpoints/{self.wandb_exp_name}", epoch)

    def test_one_epoch(self, epoch):
        self.affordance_model.eval()
        self.grasp_transformer.eval()

        test_loss = 0
        pbar = tqdm(total=len(self.test_dataloader))
        for batch in self.test_dataloader:
            img = batch[0].to(self.device)
            task_description = batch[4]
            gt_mu = batch[1].to(self.device)[:, :, :2]
            gt_grasp_rotation = batch[2].to(self.device)
            gt_grasp_pose = batch[3].to(self.device)

            with torch.no_grad():
                img_feat, text_feat = self.affordance_model.get_clip_features(
                    img, task_description
                )
                mu, cvar = self.affordance_model.get_mu_cvar(img_feat)
                grasp_rotation, grasp_pose = self.grasp_transformer(text_feat, img_feat)

            contact_loss = nn.functional.mse_loss(mu, gt_mu)
            grasp_rotation_loss = nn.functional.mse_loss(
                grasp_rotation, gt_grasp_rotation
            )
            grasp_pose_loss = nn.functional.mse_loss(grasp_pose, gt_grasp_pose)

            loss = (
                self.cfg.lambda_m * contact_loss
                + self.cfg.lambda_r * grasp_rotation_loss
                + self.cfg.lambda_g * grasp_pose_loss
            )
            test_loss += loss.item()
            pbar.update(1)
            pbar.set_description(
                f"Epoch: {epoch} | Loss: {loss.item():.2f} | Contact: {self.cfg.lambda_m * contact_loss:.2f} | Rotation: {self.cfg.lambda_r * grasp_rotation_loss:.2f} | Pose: {self.cfg.lambda_g * grasp_pose_loss:.2f}"
            )

            if self.cfg.log:
                self.logger.log(
                    {
                        "eval/loss/total": loss.item(),
                        "eval/loss/contact": self.cfg.lambda_m * contact_loss.item(),
                        "eval/loss/rotation": self.cfg.lambda_r
                        * grasp_rotation_loss.item(),
                        "eval/loss/pose": self.cfg.lambda_g * grasp_pose_loss.item(),
                    }
                )

        pbar.close()
        test_loss /= len(self.test_dataloader)
        if self.cfg.log:
            self.logger.log({"eval/loss/avg": test_loss})

        return test_loss

    def save_model(self, checkpoint_dir, epoch):
        if self.cfg.save_model:
            torch.save(
                {
                    "epoch": epoch,
                    "affordance_model": self.affordance_model.state_dict(),
                    "grasp_transformer": self.grasp_transformer.state_dict(),
                },
                f"{checkpoint_dir}/model_{epoch}.pth",
            )


if __name__ == "__main__":
    cfg = TrainingConfig(device=2, log=True, save_model=True)
    trainer = Trainer(cfg)
    trainer.train()
