import argparse
import json

import torch
import numpy as np
from pathlib import Path
from smplx.body_models import SMPL

from thesis.src.dataloader import get_dataloader
from thesis.src.care_pd.visualize_skel_walk_func import visualize_sequence, SMPL_joint_paths

STATIONARY_KEYS = {
    2: ['026__026-14-001917_wid00_3', '026__026-14-001917_wid00_3', '038__038-15-003620_wid00_0', '038__038-15-003620_wid00_0', 
        '038__038-15-003620_wid00_0', '038__038-15-003620_wid00_0', '038__038-15-003620_wid00_0', '038__038-15-003620_wid00_0', 
        '038__038-15-003620_wid00_0', '038__038-15-003620_wid00_0', '042__042-14-006028_wid01_0', '042__042-14-006028_wid01_0', 
        '042__042-14-006028_wid01_0', '042__042-14-006028_wid01_0', '042__042-14-006028_wid01_1', '042__042-14-006028_wid01_1', 
        '042__042-14-006028_wid01_1', '057__057-15-000940_wid00_1', '057__057-15-000940_wid00_3', '057__057-15-000940_wid00_3', 
        '057__057-15-003234_wid00_0', '057__057-15-003234_wid00_0', '057__057-15-003234_wid00_0', '057__057-15-003234_wid00_0', 
        '057__057-15-003234_wid00_0', '057__057-15-003234_wid00_1', '057__057-15-003234_wid00_1', '057__057-15-003234_wid00_1', 
        '057__057-15-003234_wid00_2', '057__057-15-003234_wid00_2', '057__057-15-008093_wid00_0', '057__057-15-008093_wid00_0', 
        '057__057-15-008093_wid00_0', '057__057-15-008093_wid00_0'],
    3: ['004__004-12-105182_wid00_1', '004__004-12-105182_wid00_2', '004__004-12-105182_wid00_3', '004__004-12-105182_wid00_4', 
        '004__004-12-105182_wid00_7', '004__004-12-105182_wid00_8', '004__004-12-105182_wid00_9', '004__004-13-007586_wid00_0', 
        '004__004-13-007586_wid00_1', '004__004-13-007586_wid00_2', '004__004-13-007586_wid00_3', '004__004-13-007586_wid00_4', 
        '004__004-13-007586_wid00_5', '019__019-14-005778_wid00_0', '019__019-14-005778_wid00_1', '019__019-14-005778_wid00_2', 
        '019__019-14-005778_wid00_3', '019__019-14-005778_wid00_4', '019__019-14-005795_wid01_0', '019__019-14-005795_wid01_1', 
        '019__019-14-005795_wid01_2', '019__019-14-005795_wid01_3', '019__019-14-005795_wid01_4', '019__019-14-005795_wid01_5'],
}

def visualize_single_smpl_sequence():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Reusing existing project paths
    smpl_model_path = Path('thesis/data/care_pd_preprocessing/SMPL_NEUTRAL.pkl')
    raw_smpl_npz = Path("thesis/data/raw/PD-GaM/3D_SMPL/PD-GaM_3D_SMPL_rot_trans_canonical.npz")
    labels_path = Path("thesis/data/metadata/pd_gam_labels.json")

    cfg = {
        'data': {
            'smpl_path': raw_smpl_npz,
            'severity_labels_path': labels_path,
            'patient_prefix': 'all',
        },
        'windowing': {
            'total_window_size': 60,
            'prefix_length': 15,
            'step_size': 45,
            'max_sequence_len': 200,
            'full_seq_step_size': 200,
            'min_z_travel': 0.5,
            'filter_z_travel': True,
        },
        'training': {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 0,
            'eval_split': 0.0,
            'test_split': 0.0,
            'overfit_severity_class': 1,
        },
        'model': {
            'generation_mode': 'one_shot',
        }
    }

    # 1. Fetch a single sequence
    loader = get_dataloader(cfg, mode='train')
    batch = next(iter(loader))

    key = batch['key'][0]
    actual_len = batch['seq_len'][0].item()
    sev = batch['severity'][0].item()

    # Pose shape: (T, 24, 3) or (T, 72)
    pose_raw = batch['pose'][0, :actual_len].to(dtype=torch.float32, device=device)
    
    # 2. Extract translation from batch or canonical NPZ fallback
    if 'trans' in batch and batch['trans'] is not None:
        trans = batch['trans'][0, :actual_len].to(dtype=torch.float32, device=device)
    else:
        # Load translation saved alongside canonical pose
        npz_data = np.load(raw_smpl_npz)
        trans_key = f"{key}_trans"
        if trans_key in npz_data:
            trans_np = npz_data[trans_key][:actual_len]
            trans = torch.tensor(trans_np, dtype=torch.float32, device=device)
        else:
            trans = torch.zeros((actual_len, 3), dtype=torch.float32, device=device)

    # 3. Format inputs for standard SMPL forward pass
    if pose_raw.ndim == 3 and pose_raw.shape[1] == 24:
        global_orient = pose_raw[:, 0:1, :].reshape(actual_len, 3)
        body_pose = pose_raw[:, 1:24, :].reshape(actual_len, 69)
    else:
        global_orient = pose_raw[:, :3].reshape(actual_len, 3)
        body_pose = pose_raw[:, 3:72].reshape(actual_len, 69)

    betas = torch.zeros((actual_len, 10), dtype=torch.float32, device=device)

    # 4. Forward Kinematics via smplx.body_models.SMPL
    smpl_model = SMPL(model_path=str(smpl_model_path), num_betas=10).to(device)

    with torch.no_grad():
        out = smpl_model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient
        )
        
        # Native 24 SMPL joints in world space: (T, 24, 3)
        joints_3d = (out.joints[:, :24, :] + trans[:, None, :]).cpu().numpy()

    print(f"\nSequence: {key} | Class: {sev}")
    print(f"3D Joints Array Shape: {joints_3d.shape}")
    print("Launching Matplotlib visualizer...")

    # 5. Render with index annotations and SMPL bone paths
    visualize_sequence(
        joints_3d,
        f"{key} (SMPL 24 Joints | Severity {sev})",
        show_joint_indexes=True,
        joint_paths=SMPL_joint_paths,
        projection='3d',
        fps=30,
        save_gif=False,
        severity=sev
    )

def _visualize_sequence_by_key(key: str, sev: int, smpl_model, raw_npz_data, device):
    """Loads a specific sequence key from the NPZ, computes SMPL forward pass, and plots."""
    if key not in raw_npz_data:
        print(f"Warning: Key '{key}' not found in raw NPZ data. Skipping.")
        return

    pose_raw = raw_npz_data[key]
    trans_raw = raw_npz_data.get(f"{key}_trans", None)
    actual_len = pose_raw.shape[0]

    # Convert to torch tensors
    pose_tensor = torch.tensor(pose_raw, dtype=torch.float32, device=device)
    if trans_raw is not None:
        trans_tensor = torch.tensor(trans_raw, dtype=torch.float32, device=device)
    else:
        trans_tensor = torch.zeros((actual_len, 3), dtype=torch.float32, device=device)

    # Format inputs for standard SMPL forward pass
    if pose_tensor.ndim == 3 and pose_tensor.shape[1] == 24:
        global_orient = pose_tensor[:, 0:1, :].reshape(actual_len, 3)
        body_pose = pose_tensor[:, 1:24, :].reshape(actual_len, 69)
    else:
        global_orient = pose_tensor[:, :3].reshape(actual_len, 3)
        body_pose = pose_tensor[:, 3:72].reshape(actual_len, 69)

    betas = torch.zeros((actual_len, 10), dtype=torch.float32, device=device)

    with torch.no_grad():
        out = smpl_model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient
        )
        # Native 24 SMPL joints in world space: (T, 24, 3)
        joints_3d = (out.joints[:, :24, :] + trans_tensor[:, None, :]).cpu().numpy()

    # Calculate exact anterior-posterior Z-travel
    z_travel = abs(trans_tensor[-1, 2] - trans_tensor[0, 2]).item()

    print(f"\nRendering: {key} | Class: {sev} | Frames: {actual_len} | Z-Travel: {z_travel:.3f} m")
    print("-> Close the plot window to proceed to the next sequence...")

    visualize_sequence(
        joints_3d,
        f"{key} (Class {sev} | Z-Travel: {z_travel:.2f}m)",
        show_joint_indexes=True,
        joint_paths=SMPL_joint_paths,
        projection='3d',
        fps=30,
        save_gif=False,
        severity=sev
    )


def visualize_stationary_sequences():
    parser = argparse.ArgumentParser(description="Visualize SMPL 3D Skeleton Sequences.")
    parser.add_argument("--key", type=str, default=None, help="Visualize a single specific sequence key.")
    parser.add_argument("--stationary", action="store_true", help="Step through stationary sequences sequentially.")
    parser.add_argument("--severity", type=int, default=None, choices=[0, 1, 2, 3], 
                        help="Filter stationary sequences to a specific severity class (e.g. 2).")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    smpl_model_path = Path('thesis/data/care_pd_preprocessing/SMPL_NEUTRAL.pkl')
    raw_smpl_npz = Path("thesis/data/raw/PD-GaM/3D_SMPL/PD-GaM_3D_SMPL_rot_trans_canonical.npz")
    labels_path = Path("thesis/data/metadata/pd_gam_labels.json")

    print("Loading SMPL model and canonical NPZ dataset...")
    smpl_model = SMPL(model_path=str(smpl_model_path), num_betas=10).to(device)
    raw_npz_data = np.load(raw_smpl_npz)

    with open(labels_path, 'r') as f:
        labels = json.load(f)["key_to_severity"]

    if args.key:
        sev = labels.get(args.key.split('_down')[0], -1)
        _visualize_sequence_by_key(args.key, sev, smpl_model, raw_npz_data, device)
        return

    if args.stationary:
        if not STATIONARY_KEYS:
            print("Error: STATIONARY_KEYS dictionary is empty. Paste the printed dictionary at the top of the file.")
            return

        target_classes = [args.severity] if args.severity is not None else sorted(STATIONARY_KEYS.keys())

        for sev_c in target_classes:
            keys = STATIONARY_KEYS.get(sev_c, [])
            print(f"\n{'='*70}\nStepping through {len(keys)} Stationary Sequences for Severity Class {sev_c}\n{'='*70}")
            for idx, k in enumerate(keys, 1):
                print(f"[{idx}/{len(keys)} of Class {sev_c}]")
                _visualize_sequence_by_key(k, sev_c, smpl_model, raw_npz_data, device)
        return

    print("No mode selected. Run with `--stationary` or `--stationary --severity 2`, or specify `--key <seq_key>`.")


if __name__ == "__main__":
    visualize_stationary_sequences()