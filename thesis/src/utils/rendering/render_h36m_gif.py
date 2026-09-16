import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch

from thesis.src.care_pd.smpl2h36m import convert_smpl_to_h36m
from thesis.src.utils.smpl_io import save_smpl_pkl
from thesis.src.generate_prior import generate_motion_prior_from_prefix

h36m_joint_paths = [
    [10, 9, 8, 7, 0, 1, 2, 3],
    [0, 4, 5, 6],
    [8, 11, 12, 13],
    [8, 14, 15, 16]
]

def render_three_way_gif(gt_seq, prior_seq, gen_seq, severity, output_path, fps=15, elev=35, azim=110, roll=0, gen_severity=None):
    num_frames = min(gt_seq.shape[0], prior_seq.shape[0], gen_seq.shape[0])
    
    # Pool all 3 sequences across all frames and joints
    all_x = np.concatenate([gt_seq[:, :, 0], prior_seq[:, :, 0], gen_seq[:, :, 0]])  # Lateral
    all_y = np.concatenate([gt_seq[:, :, 2], prior_seq[:, :, 2], gen_seq[:, :, 2]])  # Forward Travel
    all_z = np.concatenate([gt_seq[:, :, 1], prior_seq[:, :, 1], gen_seq[:, :, 1]])  # Vertical Height
    
    # Dynamic limits with safety margins
    x_pad, y_pad, z_pad = 0.4, 0.5, 0.2
    x_min, x_max = float(np.min(all_x) - x_pad), float(np.max(all_x) + x_pad)
    y_min, y_max = float(np.min(all_y) - y_pad), float(np.max(all_y) + y_pad)
    z_min = float(min(0.0, np.min(all_z) - z_pad))
    z_max = float(max(2.0, np.max(all_z) + z_pad))
    
    # Ensure minimum bounding volume so the box does not collapse if stationary
    x_range = max(x_max - x_min, 1.5)
    y_range = max(y_max - y_min, 2.0)
    z_range = max(z_max - z_min, 2.0)
    aspect_ratio = [x_range, y_range, z_range]

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f"Ground-truth and Synthetic Motion | Severity Class: {severity}", fontsize=15, fontweight='bold')
    
    ax_gt = fig.add_subplot(131, projection='3d')
    ax_prior = fig.add_subplot(132, projection='3d')
    ax_gen = fig.add_subplot(133, projection='3d')

    def setup_axis(ax, title):
        ax.view_init(elev=elev, azim=azim, roll=roll)
        ax.set_xlim3d([x_min, x_max])
        ax.set_ylim3d([y_min, y_max])
        ax.set_zlim3d([z_min, z_max])
        ax.set_box_aspect(aspect_ratio)
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    def update(frame):
        ax_gt.clear()
        ax_prior.clear()
        ax_gen.clear()
        
        if gen_severity is not None:
            gen_title = f"3. Synthetic Model Output\nGen Class: {gen_severity} | Frame: {frame}/{num_frames}"
        else:
            gen_title = f"3. Synthetic Model Output\nFrame: {frame}/{num_frames}"
            
        setup_axis(ax_gt, f"1. Original Ground Truth\nFrame: {frame}/{num_frames}")
        setup_axis(ax_prior, f"2. Generated FM Prior (x_0)\nFrame: {frame}/{num_frames}")
        setup_axis(ax_gen, gen_title)

        axes_and_seqs = [
            (ax_gt, gt_seq, 'cornflowerblue'),
            (ax_prior, prior_seq, 'grey'),
            (ax_gen, gen_seq, 'salmon')
        ]

        # Draw sequences with correctly mapped spatial axes
        for ax, seq, color in axes_and_seqs:
            for joint_path in h36m_joint_paths:
                xs = [seq[frame, j, 0] for j in joint_path]  # Lateral
                ys = [seq[frame, j, 2] for j in joint_path]  # Forward Travel
                zs = [seq[frame, j, 1] for j in joint_path]  # Height
                ax.plot(xs, ys, zs, color=color, linewidth=2, marker='o', markersize=3)

    interval = int((1 / fps) * 1000)
    ani = FuncAnimation(fig, update, frames=num_frames, interval=interval)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ani.save(output_path, writer='pillow', fps=fps)
    plt.close(fig)
    
    return output_path


if __name__ == "__main__":
    model_folder = "JointModel-MLP-Baseline"
    base_dir = Path(f"thesis/data/processed/{model_folder}")
    out_dir = Path("thesis/visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    gt_6d_path = base_dir / "6D_SMPL" / "ground_truth_6d.npz"
    gt_h36m_path = base_dir / "h36m" / "ground_truth_3d_world.npz"
    gen_h36m_path = base_dir / "h36m" / "generated_3d_world.npz"
    labels_path = base_dir / "h36m" / "gen_labels.json"

    temp_prior_pkl = base_dir / "SMPL" / "temp_extracted_prior.pkl"
    temp_prior_npz = base_dir / "h36m" / "temp_extracted_prior_3d_world.npz"
    
    print(f"Loading data from {base_dir}...")
    gt_6d_data = np.load(gt_6d_path, allow_pickle=True)
    gt_h36m_data = np.load(gt_h36m_path, allow_pickle=True)
    gen_h36m_data = np.load(gen_h36m_path, allow_pickle=True)
    
    gt_6d_data = gt_6d_data['arr_0'].item() if 'arr_0' in gt_6d_data.files else {k: gt_6d_data[k] for k in gt_6d_data.files}
    gt_h36m_data = gt_h36m_data['arr_0'].item() if 'arr_0' in gt_h36m_data.files else {k: gt_h36m_data[k] for k in gt_h36m_data.files}
    gen_h36m_data = gen_h36m_data['arr_0'].item() if 'arr_0' in gen_h36m_data.files else {k: gen_h36m_data[k] for k in gen_h36m_data.files}
    
    with open(labels_path, 'r') as f:
        labels = json.load(f)["key_to_severity"]
        
    # Use the first sequence pair
    gen_key = list(gen_h36m_data.keys())[0]
    idx_str = gen_key.split('_')[-1] 
    gt_h36m_key = f"GT__gt_{idx_str}"
    gt_6d_key = f"seq_{idx_str}"
    
    print(f"\nExtracting Prior for sequence: {gt_6d_key}")
    
    # Extract 6D Prefix and Target from the Ground Truth
    prefix_length = 15 
    gt_pose = torch.tensor(gt_6d_data[gt_6d_key]).unsqueeze(0)  # (1, T, 24, 6)
    gt_trans = torch.tensor(gt_6d_data[f"{gt_6d_key}_trans"]).unsqueeze(0)  # (1, T, 3)
    
    prefix_dict = {'pose': gt_pose[:, :prefix_length], 'trans': gt_trans[:, :prefix_length]}
    target_dict = {'pose': gt_pose[:, prefix_length:], 'trans': gt_trans[:, prefix_length:]}
    
    # Generate the 6D FM Prior (x_0)
    x_0_dict = generate_motion_prior_from_prefix(prefix_dict['pose'], prefix_dict['trans'], num_frames=target_dict['pose'].shape[1])
    
    # 3. Concatenate Prefix + Prior to get the full timeline
    prior_full_pose = torch.cat([prefix_dict['pose'], x_0_dict['pose']], dim=1)
    prior_full_trans = torch.cat([prefix_dict['trans'], x_0_dict['trans']], dim=1)
    
    # Push through the SMPL -> H36M conversion pipeline
    print("Converting 6D Prior -> SMPL -> H36M...")
    save_smpl_pkl(prior_full_pose, prior_full_trans, str(temp_prior_pkl), "PRIOR", "prior")
    convert_smpl_to_h36m(str(temp_prior_pkl), str(temp_prior_npz.parent), temp_prior_npz.name)
    
    # Load the newly created 3D H36M Prior
    prior_h36m_data = np.load(temp_prior_npz, allow_pickle=True)
    prior_h36m_data = prior_h36m_data['arr_0'].item() if 'arr_0' in prior_h36m_data.files else {k: prior_h36m_data[k] for k in prior_h36m_data.files}
    prior_h36m_key = list(prior_h36m_data.keys())[0]
    
    seq_gt = gt_h36m_data[gt_h36m_key]
    seq_prior = prior_h36m_data[prior_h36m_key]
    seq_gen = gen_h36m_data[gen_key]
    
    # Determine severity
    base_label_key = gen_key.replace('generated_walk_', '').split('_down')[0]
    severity = labels.get(base_label_key, labels.get(gen_key, "Unknown"))
    
    out_gif = out_dir / "test_three_way_render.gif"
    print(f"\nRendering 3-Way test GIF...")
    render_three_way_gif(seq_gt, seq_prior, seq_gen, severity, out_gif, fps=15, elev=35, azim=110, roll=0)
    print(f"Successfully saved 3-Way test GIF to: {out_gif}")
    
    # Cleanup
    temp_prior_pkl.unlink(missing_ok=True)
    temp_prior_npz.unlink(missing_ok=True)