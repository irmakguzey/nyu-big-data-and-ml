import torch
from dex_grasp.models.affordance_model import AffordanceModel
from dex_grasp.models.grasp_transformer import GraspTransformer


def load_model(device, checkpoint_path, use_clip=True, freeze_rep=True):
    affordance_model = AffordanceModel(
        src_in_features=512, freeze_rep=freeze_rep, device=device, use_clip=use_clip
    )
    grasp_transformer = GraspTransformer(text_dim=512, image_dim=512)

    checkpoint = torch.load(checkpoint_path)
    print(checkpoint["affordance_model"].keys())
    affordance_model.load_state_dict(checkpoint["affordance_model"])
    grasp_transformer.load_state_dict(checkpoint["grasp_transformer"])

    return affordance_model, grasp_transformer


if __name__ == "__main__":
    affordance_model, grasp_transformer = load_model(
        device="cuda",
        checkpoint_path="/home/irmak/Workspace/nyu-big-data-and-ml/project/checkpoints/grasp_dex_04-28_23:46:51/model_best.pth",
    )

    print(affordance_model.state_dict().keys())
