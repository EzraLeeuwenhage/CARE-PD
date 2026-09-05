import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch

from thesis.utils.pipeline_utils import build_smpl_pkl_from_6d_smpl, convert_smpl_to_h36m
from thesis.src.generate_prior import generate_prior_from_prefix

h36m_joint_paths = [
    [10, 9, 8, 7, 0, 1, 2, 3],
    [0, 4, 5, 6],
    [8, 11, 12, 13],
    [8, 14, 15, 16]
]

def render_side_by_side_gif(prior_seq, gen_seq, severity, output_path, fps=15, elev=55, azim=55, roll=135):
    """
    Headlessly renders a side-by-side 3D comparison of Prior vs Generated motion.
    
    Args:
        prior_seq (np.ndarray): Shape (T, 17, 3) - The prefix + generated prior.
        gen_seq (np.ndarray): Shape (T, 17, 3) - The prefix + generated suffix.
        severity (int): The clinical severity class.
        output_path (str): Where to save the resulting .gif.
    """
    num_frames = min(prior_seq.shape[0], gen_seq.shape[0])
    
    # Calculate global boundaries so the camera doesn't jump around
    all_data = np.concatenate([prior_seq, gen_seq], axis=0)
    min_x, min_y, min_z = np.min(all_data, axis=(0, 1))
    max_x, max_y, max_z = np.max(all_data, axis=(0, 1))
    
    x_range, y_range, z_range = max_x - min_x, max_y - min_y, max_z - min_z
    aspect_ratio = [x_range, y_range, z_range]

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle(f"Epoch Evolution | Severity Class: {severity}", fontsize=14, fontweight='bold')
    
    ax_prior = fig.add_subplot(121, projection='3d')
    ax_gen = fig.add_subplot(122, projection='3d')

    def setup_axis(ax, title):
        ax.view_init(elev=elev, azim=azim, roll=roll)
        ax.set_xlim3d([min_x, max_x])
        ax.set_ylim3d([min_y, max_y])
        ax.set_zlim3d([min_z, max_z])
        ax.set_box_aspect(aspect_ratio)
        ax.set_title(title, fontsize=12)
        # Remove tick labels for a cleaner look
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    def update(frame):
        ax_prior.clear()
        ax_gen.clear()
        
        # Change the titles depending on what you pass in
        setup_axis(ax_prior, f"Reference Sequence\nFrame: {frame}/{num_frames}")
        setup_axis(ax_gen, f"Generated Suffix (Model Output)\nFrame: {frame}/{num_frames}")

        # Draw Reference (Prior or GT)
        for joint_path in h36m_joint_paths:
            x = [prior_seq[frame, j, 0] for j in joint_path]
            y = [prior_seq[frame, j, 1] for j in joint_path]
            z = [prior_seq[frame, j, 2] for j in joint_path]
            ax_prior.plot(x, y, z, color='grey', linewidth=2, marker='o', markersize=3)
            
        # Draw Generated Output
        for joint_path in h36m_joint_paths:
            x = [gen_seq[frame, j, 0] for j in joint_path]
            y = [gen_seq[frame, j, 1] for j in joint_path]
            z = [gen_seq[frame, j, 2] for j in joint_path]
            ax_gen.plot(x, y, z, color='salmon', linewidth=2, marker='o', markersize=3)

    interval = int((1 / fps) * 1000)
    
    ani = FuncAnimation(fig, update, frames=num_frames, interval=interval)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ani.save(output_path, writer='pillow', fps=fps)
    plt.close(fig)
    
    return output_path

def render_three_way_gif(gt_seq, prior_seq, gen_seq, severity, output_path, fps=15, elev=55, azim=55, roll=135, gen_severity=None):
    """
    Headlessly renders a 3-panel 3D comparison of GT vs Prior vs Generated motion.
    """
    num_frames = min(gt_seq.shape[0], prior_seq.shape[0], gen_seq.shape[0])
    
    # Calculate global boundaries across all 3 sequences so the camera doesn't jump
    all_data = np.concatenate([gt_seq, prior_seq, gen_seq], axis=0)
    min_x, min_y, min_z = np.min(all_data, axis=(0, 1))
    max_x, max_y, max_z = np.max(all_data, axis=(0, 1))
    
    x_range, y_range, z_range = max_x - min_x, max_y - min_y, max_z - min_z
    aspect_ratio = [x_range, y_range, z_range]

    # Create a 1x3 grid for the plots
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f"Ground-truth and Synthetic Motion | Severity Class: {severity}", fontsize=16, fontweight='bold')
    
    ax_gt = fig.add_subplot(131, projection='3d')
    ax_prior = fig.add_subplot(132, projection='3d')
    ax_gen = fig.add_subplot(133, projection='3d')

    def setup_axis(ax, title):
        ax.view_init(elev=elev, azim=azim, roll=roll)
        ax.set_xlim3d([min_x, max_x])
        ax.set_ylim3d([min_y, max_y])
        ax.set_zlim3d([min_z, max_z])
        ax.set_box_aspect(aspect_ratio)
        ax.set_title(title, fontsize=12)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    def update(frame):
        ax_gt.clear()
        ax_prior.clear()
        ax_gen.clear()
        
        # Build the dynamic title for the synthetic output
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

        # Draw all three sequences
        for ax, seq, color in axes_and_seqs:
            for joint_path in h36m_joint_paths:
                x = [seq[frame, j, 0] for j in joint_path]
                y = [seq[frame, j, 1] for j in joint_path]
                z = [seq[frame, j, 2] for j in joint_path]
                ax.plot(x, y, z, color=color, linewidth=2, marker='o', markersize=3)

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
    
    # Handle dict unwrapping
    gt_6d_data = gt_6d_data['arr_0'].item() if 'arr_0' in gt_6d_data.files else {k: gt_6d_data[k] for k in gt_6d_data.files}
    gt_h36m_data = gt_h36m_data['arr_0'].item() if 'arr_0' in gt_h36m_data.files else {k: gt_h36m_data[k] for k in gt_h36m_data.files}
    gen_h36m_data = gen_h36m_data['arr_0'].item() if 'arr_0' in gen_h36m_data.files else {k: gen_h36m_data[k] for k in gen_h36m_data.files}
    
    with open(labels_path, 'r') as f:
        labels = json.load(f)["key_to_severity"]
        
    # Grab the first available sequence pair
    gen_key = list(gen_h36m_data.keys())[0]
    idx_str = gen_key.split('_')[-1] 
    gt_h36m_key = f"GT__gt_{idx_str}"
    gt_6d_key = f"seq_{idx_str}"
    
    print(f"\nExtracting Prior for sequence: {gt_6d_key}")
    
    # 1. Extract 6D Prefix and Target from the Ground Truth
    prefix_length = 15 
    gt_pose = torch.tensor(gt_6d_data[gt_6d_key]).unsqueeze(0)        # (1, T, 24, 6)
    gt_trans = torch.tensor(gt_6d_data[f"{gt_6d_key}_trans"]).unsqueeze(0)  # (1, T, 3)
    
    prefix_dict = {'pose': gt_pose[:, :prefix_length], 'trans': gt_trans[:, :prefix_length]}
    target_dict = {'pose': gt_pose[:, prefix_length:], 'trans': gt_trans[:, prefix_length:]}
    
    # 2. Generate the 6D FM Prior (x_0)
    x_0_dict = generate_prior_from_prefix(prefix_dict, target_dict)
    
    # 3. Concatenate Prefix + Prior to get the full timeline
    prior_full_pose = torch.cat([prefix_dict['pose'], x_0_dict['pose']], dim=1)
    prior_full_trans = torch.cat([prefix_dict['trans'], x_0_dict['trans']], dim=1)
    
    # 4. Push through the SMPL -> H36M conversion pipeline
    print("Converting 6D Prior -> SMPL -> H36M...")
    build_smpl_pkl_from_6d_smpl(prior_full_pose, prior_full_trans, str(temp_prior_pkl), "PRIOR", "prior")
    convert_smpl_to_h36m(str(temp_prior_pkl), str(temp_prior_npz.parent), temp_prior_npz.name)
    
    # 5. Load the newly created 3D H36M Prior
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
    render_three_way_gif(seq_gt, seq_prior, seq_gen, severity, out_gif, fps=15, elev=55, azim=55, roll=135)
    print(f"Successfully saved 3-Way test GIF to: {out_gif}")
    
    # Cleanup temporary files
    temp_prior_pkl.unlink(missing_ok=True)
    temp_prior_npz.unlink(missing_ok=True)