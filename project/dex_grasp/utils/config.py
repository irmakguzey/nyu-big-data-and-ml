import os
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    num_epochs: int = 601
    batch_size: int = 128
    num_workers: int = 32
    train_dset_split: float = 0.8
    lambda_m: float = 5e-4
    lambda_g: float = 15
    lambda_r: float = 5
    hidden_dim: int = 512
    crop_image: bool = False
    test_every_n_epochs: int = 50
    device: int = 1
    log: bool = True
    save_model: bool = True
    use_clip: bool = False
    freeze_rep: bool = False
    use_quat_loss: bool = True

    def save_config(self, save_dir: str):
        """Save config to a file"""
        os.makedirs(save_dir, exist_ok=True)
        config_path = os.path.join(save_dir, "config.txt")
        with open(config_path, "w") as f:
            for key, value in self.__dict__.items():
                f.write(f"{key}: {value}\n")

    def load_config(self, save_dir: str):
        """Load config from a file"""
        config_path = os.path.join(save_dir, "config.txt")
        with open(config_path, "r") as f:
            for line in f:
                key, value = line.split(": ")
                setattr(self, key, value)


def generate_fake_config(save_dir):
    config = TrainingConfig(
        use_clip=True,
        freeze_rep=True,
        use_quat_loss=True,
    )
    config.save_config(save_dir)


if __name__ == "__main__":
    generate_fake_config(
        "/home/irmak/Workspace/nyu-big-data-and-ml/project/checkpoints/grasp_dex_05-01_18:02:04"
    )
