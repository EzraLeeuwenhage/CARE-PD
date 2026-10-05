import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

h36m_joint_paths = [
    [10, 9, 8, 7, 0, 1, 2, 3],
    [0, 4, 5, 6],
    [8, 11, 12, 13],
    [8, 14, 15, 16]
]


def render_three_way_gif(gt_seq, prior_seq, gen_seq, severity, output_path, fps=15, elev=35, azim=110, roll=0, gen_severity=None):
    num_frames = min(gt_seq.shape[0], prior_seq.shape[0], gen_seq.shape[0])
    
    # Pool spatial bounds once across all frames and joints
    all_x = np.concatenate([gt_seq[:, :, 0], prior_seq[:, :, 0], gen_seq[:, :, 0]])  # Lateral
    all_y = np.concatenate([gt_seq[:, :, 2], prior_seq[:, :, 2], gen_seq[:, :, 2]])  # Forward Travel
    all_z = np.concatenate([gt_seq[:, :, 1], prior_seq[:, :, 1], gen_seq[:, :, 1]])  # Vertical Height
    
    x_pad, y_pad, z_pad = 0.4, 0.5, 0.2
    x_min, x_max = float(np.min(all_x) - x_pad), float(np.max(all_x) + x_pad)
    y_min, y_max = float(np.min(all_y) - y_pad), float(np.max(all_y) + y_pad)
    z_min = float(min(0.0, np.min(all_z) - z_pad))
    z_max = float(max(2.0, np.max(all_z) + z_pad))
    
    aspect_ratio = [max(x_max - x_min, 1.5), max(y_max - y_min, 2.0), max(z_max - z_min, 2.0)]

    fig = plt.figure(figsize=(12, 4), dpi=100) # reduce dpi further if necessary for speed
    fig.suptitle(f"Ground-truth and Synthetic Motion | Severity Class: {severity}", fontsize=13, fontweight='bold')
    
    axes = [
        fig.add_subplot(131, projection='3d'),
        fig.add_subplot(132, projection='3d'),
        fig.add_subplot(133, projection='3d')
    ]

    titles = [
        "1. Original Ground Truth",
        "2. Generated FM Prior (x_0)",
        "3. Synthetic Model Output" if gen_severity is None else f"3. Synthetic Model Output (Gen Class: {gen_severity})"
    ]

    # Initialize 3D axis projections and title text objects only once
    title_text_objs = []
    for ax, title in zip(axes, titles):
        ax.view_init(elev=elev, azim=azim, roll=roll)
        ax.set_xlim3d([x_min, x_max])
        ax.set_ylim3d([y_min, y_max])
        ax.set_zlim3d([z_min, z_max])
        ax.set_box_aspect(aspect_ratio)
        t_obj = ax.set_title(f"{title}\nFrame: 0/{num_frames}", fontsize=10, pad=6)
        title_text_objs.append((t_obj, title))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    # Instantiate persistent Line3D artists once
    colors = ['cornflowerblue', 'grey', 'salmon']
    lines = []
    for ax, col in zip(axes, colors):
        sub_lines = [
            ax.plot([], [], [], color=col, linewidth=2, marker='o', markersize=2.5)[0]
            for _ in h36m_joint_paths
        ]
        lines.append(sub_lines)

    seqs = [gt_seq, prior_seq, gen_seq]

    # Update callback
    def update(frame):
        for sub_lines, seq in zip(lines, seqs):
            for line, path in zip(sub_lines, h36m_joint_paths):
                xs = seq[frame, path, 0]
                ys = seq[frame, path, 2]
                zs = seq[frame, path, 1]
                line.set_data(xs, ys)
                line.set_3d_properties(zs)

        for t_obj, base_title in title_text_objs:
            t_obj.set_text(f"{base_title}\nFrame: {frame + 1}/{num_frames}")

    interval = int((1 / fps) * 1000)
    ani = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ani.save(output_path, writer='pillow', fps=fps)
    plt.close(fig)
    
    return output_path