import torch
import numpy as np
from pathlib import Path
from smplx.body_models import SMPL

from thesis.src.dataloader import get_dataloader
from thesis.src.care_pd.visualize_skel_walk_func import visualize_sequence, SMPL_joint_paths

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

if __name__ == "__main__":
    visualize_single_smpl_sequence()