import torch


def rotvec_to_quaternion(rotvec: torch.Tensor) -> torch.Tensor:
    """
    Convert a rotation vector (axis-angle) to quaternion.

    Args:
        rotvec: (B, 3) torch tensor, rotation vectors.

    Returns:
        quaternions: (B, 4) torch tensor, unit quaternions (x, y, z, w).
    """
    theta = torch.norm(rotvec, dim=-1, keepdim=True)  # (B, 1)
    half_theta = 0.5 * theta

    small_angle = theta < 1e-6

    # Normalize axis
    axis = torch.where(small_angle, torch.zeros_like(rotvec), rotvec / theta)

    sin_half_theta = torch.where(
        small_angle,
        0.5 - theta**2 / 48,  # sin(x/2) ~ x/2 - x^3/48
        torch.sin(half_theta) / theta,
    )

    quat_xyz = axis * sin_half_theta  # (B, 3)
    quat_w = torch.cos(half_theta)  # (B, 1)

    quat = torch.cat([quat_xyz, quat_w], dim=-1)  # (B, 4)
    return quat
