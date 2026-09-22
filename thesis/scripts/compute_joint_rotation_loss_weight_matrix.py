import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from thesis.src.utils.pipeline_utils import load_config
from thesis.src.dataloader import get_dataloader
from thesis.src.model_backbones import generate_x0


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Converts 3D axis-angle vectors (norm is angle in radians) to 3x3 rotation matrices.
    
    Uses Rodrigues' rotation formula.
    """
    shape = axis_angle.shape
    r = axis_angle.view(-1, 3)
    theta = torch.norm(r, dim=-1, keepdim=True)
    theta_safe = torch.clamp(theta, min=1e-8)
    k = r / theta_safe

    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    zero = torch.zeros_like(kx)

    # Skew-symmetric cross product matrix K
    K = torch.stack([
        zero, -kz, ky,
        kz, zero, -kx,
        -ky, kx, zero
    ], dim=-1).view(-1, 3, 3)

    I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).unsqueeze(0)

    # Rodrigues formula: R = I + sin(theta)*K + (1 - cos(theta))*K^2
    sin_t = torch.sin(theta).unsqueeze(-1)
    cos_t = torch.cos(theta).unsqueeze(-1)
    K2 = torch.bmm(K, K)

    R = I + sin_t * K + (1.0 - cos_t) * K2

    # Where angle is essentially zero, return Identity
    small_angle_mask = (theta < 1e-6).view(-1, 1, 1)
    R = torch.where(small_angle_mask, I, R)

    return R.view(*shape[:-1], 3, 3)


def so3_geodesic_distance_deg(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    """Computes the exact geodesic angular distance on SO(3) in degrees between two rotation matrices."""
    m = torch.matmul(r1, r2.transpose(-1, -2))
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    return torch.acos(cos_angle) * (180.0 / np.pi)


def run_diagnostics():
    cfg = load_config("thesis/configs/overfit_baseline_3d.yaml")
    loader = get_dataloader(cfg, mode="train")
    batch = next(iter(loader))

    pose = batch["pose"]  # [B, T, J, 3] in radians
    trans = batch["trans"]  # [B, T, 3] in meters
    prefix_len = cfg["windowing"]["prefix_length"]
    prior_scale = cfg["training"].get("prior_noise_scale", 0.2)

    x_1 = {"pose": pose, "trans": trans}
    x_0 = generate_x0(x_1, prefix_len, prior_scale)

    # Target continuous vector field: u_true = x_1 - x_0
    u_true_pose = x_1["pose"] - x_0["pose"]
    u_true_trans = x_1["trans"] - x_0["trans"]

    print("\n" + "=" * 70)
    print("      FLOW MATCHING 3D TARGET FIELD (u_true) STATISTICS")
    print("=" * 70)

    # 1. Target field distribution over the active target frames
    u_pose_target = u_true_pose[:, prefix_len:].cpu().numpy()
    u_trans_target = u_true_trans[:, prefix_len:].cpu().numpy()

    pose_dim = u_pose_target.shape[3]
    assert pose_dim == 3, f"Expected 3D axis-angle, got pose_dim={pose_dim}"

    # Joint-wise standard deviation (radians/step)
    std_per_joint = np.std(u_pose_target, axis=(0, 1, 3))  # [J]
    var_per_joint = std_per_joint ** 2

    print(f"Target Pose Shape:              {u_pose_target.shape} (B, T_targ, Joints, 3)")
    print(f"Pose Velocity Mean Absolute:    {np.mean(np.abs(u_pose_target)):.6f} rad")
    print(f"Pose Velocity Max Absolute:     {np.max(np.abs(u_pose_target)):.6f} rad")
    print(f"Average Joint Std:              {np.mean(std_per_joint):.6f} rad")
    print(f"Min Joint Std (Quiescent):      {np.min(std_per_joint):.6f} rad (Joint {np.argmin(std_per_joint)})")
    print(f"Max Joint Std (Active):         {np.max(std_per_joint):.6f} rad (Joint {np.argmax(std_per_joint)})")
    print(f"Variance Disparity (Max / Min): {np.max(var_per_joint) / (np.min(var_per_joint) + 1e-8):.1f}x")

    print("\n" + "-" * 70)
    print("Translation Velocity (m):")
    print(f"Trans Velocity Mean Absolute:   {np.mean(np.abs(u_trans_target)):.6f} m")
    print(f"Trans Velocity Max Absolute:    {np.max(np.abs(u_trans_target)):.6f} m")

    # 2. Mathematical Mapping: 3D Axis-Angle Coordinate Error -> Angular Error (Degrees)
    print("\n" + "=" * 70)
    print("   MAPPING: 3D COORDINATE ERROR -> GEODESIC ROTATION ERROR (DEGREES)")
    print("=" * 70)

    # Base rotation representative of an active limb joint (~0.3 rad angle)
    base_aa = torch.tensor([0.2, -0.2, 0.1]).repeat(5000, 1)
    base_rot = axis_angle_to_matrix(base_aa)

    perturbation_stds = [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15]
    print(f"{'Perturb Std (rad)':<18} | {'Implied MSE':<14} | {'Mean Geodesic Error':<20}")
    print("-" * 70)

    for sigma in perturbation_stds:
        jitter = torch.randn_like(base_aa) * sigma
        perturbed_aa = base_aa + jitter
        perturbed_rot = axis_angle_to_matrix(perturbed_aa)

        deg_error = so3_geodesic_distance_deg(base_rot, perturbed_rot).mean().item()
        implied_mse = sigma ** 2  # Mean squared error per coordinate
        print(f"{sigma:<18.4f} | {implied_mse:<14.6f} | {deg_error:<20.2f}°")

    # TODO: fix that this weighting actually increases with higher variance (more important) walking joints
    # 3. Precision Matrix Weights Calculation (Bregman Quadratic Form)
    # W_j = 1 / Var(u_j)
    var_per_coord = np.var(u_pose_target, axis=(0, 1), keepdims=True)  # [1, 1, J, 3]
    precision_weights = 1.0 / (var_per_coord + 1e-5)
    normalized_weights = precision_weights / np.mean(precision_weights)

    out_dir = Path("thesis/data/metadata/loss_weighting")
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_file = out_dir / "precision_weights_3d.npy"
    np.save(weights_file, normalized_weights)

    print("\n" + "=" * 70)
    print(f"Precision weights successfully saved to: {weights_file}")
    print(f"Weights Shape: {normalized_weights.shape} | Mean: {np.mean(normalized_weights):.2f} | Min: {np.min(normalized_weights):.2f} | Max: {np.max(normalized_weights):.2f}")
    print("=" * 70)


if __name__ == "__main__":
    run_diagnostics()