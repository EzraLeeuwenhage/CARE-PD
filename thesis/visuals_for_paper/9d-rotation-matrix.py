import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_sphere(ax, center, radius, color='silver', alpha=0.9):
    """Draws a 3D sphere to represent a joint."""
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor='none', zorder=1)

def plot_local_frame(ax, origin, R_matrix, label_suffix, scale=0.8):
    """Draws the local X(Red), Y(Green), and Z(Blue) axes emerging from a joint."""
    x_vec = R_matrix @ np.array([1, 0, 0]) * scale
    y_vec = R_matrix @ np.array([0, 1, 0]) * scale
    z_vec = R_matrix @ np.array([0, 0, 1]) * scale

    # Plot arrows
    ax.quiver(*origin, *x_vec, color='red', linewidth=2, arrow_length_ratio=0.15, zorder=5)
    ax.quiver(*origin, *y_vec, color='green', linewidth=2, arrow_length_ratio=0.15, zorder=5)
    ax.quiver(*origin, *z_vec, color='blue', linewidth=2, arrow_length_ratio=0.15, zorder=5)

    # Add labels to the axes
    ax.text(*(origin + x_vec * 1.2), f'$x_{{{label_suffix}}}$', color='red', fontsize=14, fontweight='bold')
    ax.text(*(origin + y_vec * 1.2), f'$y_{{{label_suffix}}}$', color='green', fontsize=14, fontweight='bold')
    ax.text(*(origin + z_vec * 1.2), f'$z_{{{label_suffix}}}$', color='blue', fontsize=14, fontweight='bold')

# Defines matrices from example given in paper
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

# compute global orientation
R_global = R_A @ R_B
P_A = np.array([0, 0, 0])
rod_length = 2.0
P_B = P_A + R_A @ np.array([0, rod_length, 0])

# Plotting the visual
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# global coordinate system
global_scale = 2.5
ax.quiver(0, 0, 0, global_scale, 0, 0, color='gray', linewidth=1.5, linestyle='dashed', arrow_length_ratio=0.05)
ax.quiver(0, 0, 0, 0, global_scale, 0, color='gray', linewidth=1.5, linestyle='dashed', arrow_length_ratio=0.05)
ax.quiver(0, 0, 0, 0, 0, global_scale, color='gray', linewidth=1.5, linestyle='dashed', arrow_length_ratio=0.05)

ax.text(global_scale + 0.1, 0, 0, 'Global X', color='black', fontsize=12)
ax.text(0, global_scale + 0.1, 0, 'Global Y', color='black', fontsize=12)
ax.text(0, 0, global_scale + 0.1, 'Global Z', color='black', fontsize=12)

# Draw the bone (green rod) connecting the two joints
ax.plot([P_A[0], P_B[0]], [P_A[1], P_B[1]], [P_A[2], P_B[2]], color='forestgreen', linewidth=6, zorder=2)

# Draw joint spheres and local coordinate frames
joint_radius = 0.2
plot_sphere(ax, P_A, joint_radius)
plot_sphere(ax, P_B, joint_radius)
plot_local_frame(ax, P_A, R_A, 'A')
plot_local_frame(ax, P_B, R_global, 'B')

# some formatting
ax.set_xlim([-1.5, 2.5])
ax.set_ylim([-1.5, 2.5])
ax.set_zlim([-1.5, 2.5])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Kinematic Chain: Global Orientation of Joint B', fontsize=14, fontweight='bold')
ax.view_init(elev=25, azim=45)

plt.tight_layout()
plt.show()