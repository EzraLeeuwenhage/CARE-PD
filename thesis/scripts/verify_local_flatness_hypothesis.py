import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from thesis.src.utils.pipeline_utils import load_config
from thesis.src.dataloader import get_dataloader
from thesis.src.model_backbones import generate_x0

# TODO: work this out, important proof for (rotation data + prior generation) linear interpolation on 3D rotations is fine
# because it is very very close to the rotation velocities on the actual SO(3) manifold

# ==========================================
# Fast Vectorized SO(3) Math for GPU
# ==========================================
def axis_angle_to_quaternion(axis_angle):
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def compute_batch_geodesic_distance(pose1, pose2):
    """Computes exact Riemannian MPJAE across entire batch (radians)."""
    R1 = quaternion_to_matrix(axis_angle_to_quaternion(pose1))
    R2 = quaternion_to_matrix(axis_angle_to_quaternion(pose2))
    
    R_rel = torch.matmul(R2, R1.transpose(-1, -2))
    trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    d_geo = torch.acos(cos_theta)
    
    # Return mean across sequence length (dim=1) and joints (dim=2)
    return d_geo.mean(dim=(1, 2))


# ==========================================
# Main Verification Routine
# ==========================================
def verify_dataset_local_flatness():
    print("\n" + "="*70)
    print("   DATASET-WIDE VERIFICATION: EUCLIDEAN VS. RIEMANNIAN GEOMETRY")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] Running vectorized computation on: {device}")

    # 1. Load Config and Override to Full Dataset
    cfg = load_config("thesis/configs/overfit_baseline_3d.yaml")
    
    # Disable overfitting to load all 3,306 chunks
    cfg['training']['overfit_severity_class'] = -1
    cfg['training']['prior_noise_scale'] = 1.0  # Deterministic prior
    prefix_len = cfg['windowing']['prefix_length']
    
    loader = get_dataloader(cfg, mode='train')
    
    # 2. Tracking Variables
    taus = np.linspace(0.0, 1.0, 11)
    
    # We will accumulate the mean distances over the entire dataset
    dataset_expected_means = {t: [] for t in taus}
    dataset_actual_means = {t: [] for t in taus}
    
    # We will track the absolute worst-case distortion at tau=0.5 (maximum wobble point)
    all_midpoint_distortions = []
    all_total_distances = []

    print(f"\n[Processing] Sweeping over {len(loader.dataset)} sequence chunks...")
    
    for batch in tqdm(loader, desc="Calculating Geodesic Geometries"):
        x_1 = {
            'pose': batch['pose'].to(device), 
            'trans': batch['trans'].to(device)
        }
        
        # Generate the deterministic stationary prior (x_0)
        x_0 = generate_x0(x_1, prefix_len, prior_noise_scale=1.0)

        # Isolate target frames only (frames 15 to 60)
        targ_x1 = x_1['pose'][:, prefix_len:]
        targ_x0 = x_0['pose'][:, prefix_len:]

        # Compute Total Distance on the Manifold (x_0 to x_1) for each sample in batch
        total_dists_rad = compute_batch_geodesic_distance(targ_x0, targ_x1)
        all_total_distances.extend(total_dists_rad.cpu().numpy() * (180.0 / np.pi))

        # Sweep across tau
        for tau in taus:
            # Linear Euclidean interpolation in 3D axis-angle space
            x_tau_euclidean = (1 - tau) * targ_x0 + tau * targ_x1
            
            expected_dist_rad = tau * total_dists_rad
            actual_dist_rad = compute_batch_geodesic_distance(targ_x0, x_tau_euclidean)
            
            dataset_expected_means[tau].append(expected_dist_rad.mean().item() * (180.0 / np.pi))
            dataset_actual_means[tau].append(actual_dist_rad.mean().item() * (180.0 / np.pi))
            
            # Record per-sample distortion specifically at the tau=0.5 midpoint
            if np.isclose(tau, 0.5):
                errors_rad = torch.abs(actual_dist_rad - expected_dist_rad)
                all_midpoint_distortions.extend(errors_rad.cpu().numpy() * (180.0 / np.pi))

    # 3. Aggregate Results
    final_expected = [np.mean(dataset_expected_means[t]) for t in taus]
    final_actual = [np.mean(dataset_actual_means[t]) for t in taus]
    
    mean_dist = np.mean(all_total_distances)
    max_dist = np.max(all_total_distances)
    
    p50_err = np.percentile(all_midpoint_distortions, 50)
    p95_err = np.percentile(all_midpoint_distortions, 95)
    p99_err = np.percentile(all_midpoint_distortions, 99)
    max_err = np.max(all_midpoint_distortions)

    print("\n" + "="*70)
    print("   DATASET AGGREGATE RESULTS")
    print("="*70)
    print(f"Total distance to target (Mean):      {mean_dist:.2f}°")
    print(f"Total distance to target (Max):       {max_dist:.2f}°")
    print("-" * 70)
    print("Midpoint Distortion Error (Wobble at tau=0.5):")
    print(f"  * Median (50th %):                  {p50_err:.4f}°")
    print(f"  * 95th Percentile:                  {p95_err:.4f}°")
    print(f"  * 99th Percentile:                  {p99_err:.4f}°")
    print(f"  * Absolute Worst Case (Max):        {max_err:.4f}°")
    print("="*70)
    
    if p99_err < 0.5:
        print("\n>> VERDICT: The Local Flatness hypothesis is strongly confirmed.")
        print(">> In 99% of the full training dataset, Euclidean interpolation")
        print(">> deviates from the Riemannian geodesic by less than 0.5 degrees.")
    else:
        print("\n>> VERDICT: Significant curvature detected in edge cases.")
        
    # 4. Plot 1: Average Trajectory Curve
    out_dir = Path("thesis/visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(taus, final_expected, 'k--', label="Expected (True Riemannian Geodesic)", linewidth=2)
    plt.plot(taus, final_actual, 'b-', label="Actual (Euclidean Interpolation)", linewidth=2)
    plt.fill_between(taus, final_expected, final_actual, color='red', alpha=0.2, label="Mean Geodesic Distortion")
    plt.title("Dataset Mean: Euclidean vs. Riemannian Integration Path")
    plt.xlabel("Flow Matching Time (\u03C4)")
    plt.ylabel("Distance from Prior (Degrees)")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / "dataset_flatness_curve.png")
    plt.close()
    
    # 5. Plot 2: Histogram of Midpoint Distortions
    plt.figure(figsize=(10, 6))
    plt.hist(all_midpoint_distortions, bins=50, color='red', alpha=0.7, edgecolor='black')
    plt.axvline(x=p99_err, color='k', linestyle='dashed', linewidth=2, label=f'99th % ({p99_err:.3f}°)')
    plt.title("Distribution of Maximum Distortion Error (Wobble) Across Dataset")
    plt.xlabel("Maximum Distortion Error at Midpoint (Degrees)")
    plt.ylabel("Number of Sequences")
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(out_dir / "dataset_distortion_histogram.png")
    plt.close()

    print(f"\nSaved summary plots to: {out_dir.resolve()}")

if __name__ == "__main__":
    verify_dataset_local_flatness()