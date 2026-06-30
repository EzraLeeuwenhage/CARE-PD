import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R

def get_cube_vertices():
    v = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]
    ])
    return v * 0.5 

def get_cube_faces(vertices):
    return [
        [vertices[0], vertices[1], vertices[2], vertices[3]], 
        [vertices[4], vertices[5], vertices[6], vertices[7]], 
        [vertices[0], vertices[1], vertices[5], vertices[4]], 
        [vertices[2], vertices[3], vertices[7], vertices[6]], 
        [vertices[0], vertices[3], vertices[7], vertices[4]], 
        [vertices[1], vertices[2], vertices[6], vertices[5]]  
    ]

axis = np.array([1.0, 1.2, 2.0])
axis_hat = axis / np.linalg.norm(axis)
angle_rad = np.pi / 2

rot_vec = axis_hat * angle_rad
rotation = R.from_rotvec(rot_vec)

vertices = get_cube_vertices()
rotated_vertices = rotation.apply(vertices)

# plotting the figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw the Rotated Cube
faces = get_cube_faces(rotated_vertices)
cube_collection = Poly3DCollection(faces, facecolors='#a2b9bc', linewidths=1.5, edgecolors='black', alpha=0.6)
ax.add_collection3d(cube_collection)

# Draw the Rotation Axis
axis_length = 2
ax.plot([-axis_hat[0]*axis_length, axis_hat[0]*axis_length],
        [-axis_hat[1]*axis_length, axis_hat[1]*axis_length],
        [-axis_hat[2]*axis_length, axis_hat[2]*axis_length],
        color='forestgreen', linewidth=4)

# Draw the Coordinate System
ax.quiver(0, 0, 0, 1.5, 0, 0, color='r', arrow_length_ratio=0.1, alpha=0.5)
ax.quiver(0, 0, 0, 0, 1.5, 0, color='g', arrow_length_ratio=0.1, alpha=0.5)
ax.quiver(0, 0, 0, 0, 0, 1.5, color='b', arrow_length_ratio=0.1, alpha=0.5)

# Draw the Rotation Indicators
# Create vectors u and v orthogonal to the axis to define the circular plane
random_vec = np.array([1, 0, 0]) if abs(axis_hat[0]) < 0.9 else np.array([0, 1, 0])
u = np.cross(axis_hat, random_vec)
u = u / np.linalg.norm(u)
v = np.cross(axis_hat, u)

radius = 0.15
shift = axis_hat * 1.2  # Shift up the axis so it sits above the cube

# Draw the 360-degree dashed circle
t_full = np.linspace(0, 2 * np.pi, 100)
circle_pts = np.array([radius * (np.cos(theta) * u + np.sin(theta) * v) + shift for theta in t_full])
ax.plot(circle_pts[:, 0], circle_pts[:, 1], circle_pts[:, 2], color='goldenrod', linewidth=1.5, linestyle='--', alpha=0.6)

# Draw the rotation arc
t_arc = np.linspace(0, angle_rad, 50)
arc_pts = np.array([radius * (np.cos(theta) * u + np.sin(theta) * v) + shift for theta in t_arc])
ax.plot(arc_pts[:, 0], arc_pts[:, 1], arc_pts[:, 2], color='goldenrod', linewidth=3)
ax.scatter(*arc_pts[-1], color='goldenrod', s=60, marker='>') 

tip_pos = axis_hat * (axis_length + 0.1)
ax.text(tip_pos[0], tip_pos[1], tip_pos[2], r'$\hat{\omega}$', color='forestgreen', fontsize=20, fontweight='bold')

mid_angle = angle_rad / 2
label_offset_vec = (np.cos(mid_angle) * u + np.sin(mid_angle) * v)
label_pos = (radius * 1.3) * label_offset_vec + shift
ax.text(label_pos[0], label_pos[1], label_pos[2], r'$||\vec{\omega}||$', color='goldenrod', fontsize=18, fontweight='bold')

# Formatting
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])
# ax.set_axis_off()
ax.set_title('Axis-Angle Representation', fontsize=16, fontweight='bold')

ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.show()