import numpy as np
import matplotlib.pyplot as plt

def switch_axes(v):
    """Swaps Y and Z axes for Matplotlib visualization."""
    return np.array([v[0], v[2], v[1]])


def plot_sphere(ax, center, radius, color='#b0c4de'):
    """Draws 3D sphere with shading to represent a joint."""
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 40)

    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    ax.plot_surface(x, z, y, color=color, alpha=0.8, 
                    edgecolors='white', linewidth=0.2, shade=True, zorder=1)

def plot_local_frame(ax, origin, R_matrix, label_suffix, scale=0.8, custom_offsets=None):
    """Draws the local X(Red), Y(Green), and Z(Blue) axes."""
    if custom_offsets is None:
        custom_offsets = {'x': np.zeros(3), 'y': np.zeros(3), 'z': np.zeros(3)}

    x_vec = R_matrix @ np.array([1, 0, 0]) * scale
    y_vec = R_matrix @ np.array([0, 1, 0]) * scale
    z_vec = R_matrix @ np.array([0, 0, 1]) * scale

    orig_p = switch_axes(origin)
    ax.quiver(*orig_p, *switch_axes(x_vec), color='red', linewidth=2, arrow_length_ratio=0.15, zorder=5)
    ax.quiver(*orig_p, *switch_axes(y_vec), color='green', linewidth=2, arrow_length_ratio=0.15, zorder=5)
    ax.quiver(*orig_p, *switch_axes(z_vec), color='blue', linewidth=2, arrow_length_ratio=0.15, zorder=5)

    # text positions with optional offsets
    pos_x = origin + (x_vec * 1.2) + custom_offsets.get('x', np.zeros(3))
    pos_y = origin + (y_vec * 1.2) + custom_offsets.get('y', np.zeros(3))
    pos_z = origin + (z_vec * 1.2) + custom_offsets.get('z', np.zeros(3))

    ax.text(*switch_axes(pos_x), f'$x_{{{label_suffix}}}$', color='red', fontsize=14, fontweight='bold')
    ax.text(*switch_axes(pos_y), f'$y_{{{label_suffix}}}$', color='green', fontsize=14, fontweight='bold')
    ax.text(*switch_axes(pos_z), f'$z_{{{label_suffix}}}$', color='blue', fontsize=14, fontweight='bold')

def plot_rotation_arc(ax, center, rotation_axis, start_vec, angle_rad, color='goldenrod', radius=0.35, label=None):
    """Draws a curved arrow tightly around a local axis to indicate rotation."""
    rot_axis = rotation_axis / np.linalg.norm(rotation_axis)
    s_vec = start_vec / np.linalg.norm(start_vec)

    # orthoginal start vector to rotation axis
    s_vec = s_vec - np.dot(s_vec, rot_axis) * rot_axis
    s_vec = s_vec / np.linalg.norm(s_vec)
    v_vec = np.cross(rot_axis, s_vec)
    
    # arc points
    draw_angle = angle_rad * 0.85
    t = np.linspace(0, draw_angle, 30)
    arc_pts = np.array([center + radius * (np.cos(theta) * s_vec + np.sin(theta) * v_vec) for theta in t])
    
    # plotting arc
    arc_pts_p = np.array([switch_axes(pt) for pt in arc_pts])
    ax.plot(arc_pts_p[:, 0], arc_pts_p[:, 1], arc_pts_p[:, 2], color=color, linewidth=2.5, zorder=6)
    
    # draw arrowhead
    tip = arc_pts[-1]
    tangent = -np.sin(draw_angle) * s_vec + np.cos(draw_angle) * v_vec
    tangent = (tangent / np.linalg.norm(tangent)) * 0.1 
    ax.quiver(*switch_axes(tip), *switch_axes(tangent), color=color, linewidth=2, arrow_length_ratio=1.0, zorder=6)

    if label:
        mid_theta = draw_angle / 2
        label_offset_vec = (np.cos(mid_theta) * s_vec + np.sin(mid_theta) * v_vec)
        label_pos = center + (radius * 1.4) * label_offset_vec
        ax.text(*switch_axes(label_pos), label, color=color, fontsize=14, fontweight='bold')

# Defines matrices from the example in the paper
R_A = np.array([
    [1,  0,  0],
    [0,  0, -1],
    [0,  1,  0]
])

R_B = np.array([
    [ 0,  0,  1],
    [ 0,  1,  0],
    [-1,  0,  0]
])

# Compute global orientation and joint positions
R_global = R_A @ R_B
P_A = np.array([0, 0, 0])
rod_length = 2.0
P_B = P_A + R_A @ np.array([0, rod_length, 0]) 

# Plotting
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Draw global coordinate system
g_scale = 2.5
ax.quiver(*switch_axes([0,0,0]), *switch_axes([g_scale, 0, 0]), color='gray', linestyle='dashed', arrow_length_ratio=0.05)
ax.quiver(*switch_axes([0,0,0]), *switch_axes([0, g_scale, 0]), color='gray', linestyle='dashed', arrow_length_ratio=0.05)
ax.quiver(*switch_axes([0,0,0]), *switch_axes([0, 0, g_scale]), color='gray', linestyle='dashed', arrow_length_ratio=0.05)

ax.text(*switch_axes([g_scale + 0.1, 0, 0]), 'Global X', color='black', fontsize=8)
ax.text(*switch_axes([0, g_scale + 0.1, 0]), 'Global Y', color='black', fontsize=8)
ax.text(*switch_axes([0, -0.2, g_scale + 0.1]), 'Global Z', color='black', fontsize=8)

# Connect joints with rod
ax.plot([switch_axes(P_A)[0], switch_axes(P_B)[0]], [switch_axes(P_A)[1], switch_axes(P_B)[1]], [switch_axes(P_A)[2], switch_axes(P_B)[2]], 
        color='black', linewidth=3, zorder=2)

# Draw joints as spheres and local frames
joint_radius = 0.12
plot_sphere(ax, P_A, joint_radius)
plot_sphere(ax, P_B, joint_radius)

offsets_A = {
    'x': np.array([-0.3, 0, -0.4]), 
    'y': np.array([0.45, 0, -0.55])
}
plot_local_frame(ax, P_A, R_A, 'A', scale=1.0, custom_offsets=offsets_A)
plot_local_frame(ax, P_B, R_global, 'B', scale=1.0)

# Draw rotation arc for joint rotations
plot_rotation_arc(ax, center=P_A, rotation_axis=np.array([1, 0, 0]), 
                  start_vec=np.array([0, 1, 0]), angle_rad=np.pi/2, 
                  color='goldenrod', label=r'$90^\circ$')

axis_B = R_A @ np.array([0, 1, 0])
start_vec_B = R_A @ np.array([1, 0, 0])
plot_rotation_arc(ax, center=P_B, rotation_axis=axis_B, 
                  start_vec=start_vec_B, angle_rad=np.pi/2, 
                  color='goldenrod', label=r'$90^\circ$')

# formatting (note that Y and Z are swapped for visualization)
ax.set_xlim([-1.5, 2.5])
ax.set_ylim([-1.5, 2.5])
ax.set_zlim([-1.5, 2.5])

ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_zlabel('Y')
ax.set_axis_off()
ax.set_title('Chained 9D Rotations', fontsize=15, fontweight='bold')

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.view_init(elev=20, azim=-60)
plt.tight_layout()
plt.show()