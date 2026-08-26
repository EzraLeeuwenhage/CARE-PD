"""
You can execute this script directly from your normal Python 3.9 environment:
    python -m thesis.utils.render_smpl_with_blender \
        --pkl_path "thesis/data/processed/JointModel-MLP-Baseline/SMPL/ground_truth.pkl" \
        --sequence_key "gt_000" \
        --frame_idx 15 \
        --output_path "thesis/visualizations/gt_000_frame15.png"

    # Render a full sequence:
    python thesis/utils/render_smpl_with_blender.py --sequence_key "gt_000"
    
    # Render a single frame:
    python thesis/utils/render_smpl_with_blender.py --sequence_key "gt_000" --frame_idx 15

It automatically detects if it is running in standard Python or Blender Python 
and routes the execution accordingly.
"""

import sys
import os
import argparse
import subprocess
import numpy as np
import math
from pathlib import Path

# If bpy imports successfully, we are running inside Blender's embedded Python.
# If it fails, we are in the normal Python 3.9 environment.
try:
    import bpy
    INSIDE_BLENDER = True
except ImportError:
    INSIDE_BLENDER = False


def parse_arguments(inside_blender=False):
    """Parses arguments depending on execution context."""
    if inside_blender:
        # Blender passes its own arguments; slice only the ones after '--'
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    else:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Headless Blender SMPL Renderer")
    
    # Dataset and Sequence Selectors (Python phase)
    parser.add_argument("--pkl_path", type=str, 
                        default="thesis/data/processed/JointModel-MLP-Baseline/SMPL/ground_truth.pkl", 
                        help="Path to the SMPL dataset .pkl file")
    parser.add_argument("--sequence_key", type=str, default="gt_000", 
                        help="Name or index of sequence to render (e.g., 'gt_000', 'gen_001', 'seq_000')")
    parser.add_argument("--frame_idx", type=int, default=None, 
                        help="Specific frame to render. If None, renders the entire sequence.")
    
    # Model and Mesh Assets
    parser.add_argument("--smpl_model_path", type=str, 
                        default="thesis/data/care_pd_preprocessing/SMPL_NEUTRAL.pkl", 
                        help="Path to neutral SMPL body model")
    parser.add_argument("--faces_path", type=str, 
                        default="thesis/data/care_pd_preprocessing/smpl_faces.npy", 
                        help="Path to SMPL topological faces .npy file")
    
    # Render Settings
    parser.add_argument("--model_name", type=str, default="JointModel-MLP-Baseline",
                        help="Model directory name under thesis/visualizations")
    parser.add_argument("--output_path", type=str, default=None, 
                        help="Direct file output path (overrides model_name / rendered_smpl folder if set)")
    parser.add_argument("--resolution", type=int, default=500, 
                        help="Square resolution of the output image")
    parser.add_argument("--render_engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"], 
                        help="Blender render engine to use")
    
    # Internal Bridge (Blender phase)
    parser.add_argument("--temp_vertices_path", type=str, default="temp_blender_vertices.npy", 
                        help="Internal bridge path for computed 3D vertices")

    return parser.parse_args(argv)

def find_sequence_in_pkl(pkl_data, sequence_key):
    if "__" in sequence_key:
        subj, walk = sequence_key.split("__", 1)
        if subj in pkl_data and walk in pkl_data[subj]:
            return pkl_data[subj][walk], sequence_key

    for subj, walks in pkl_data.items():
        if isinstance(walks, dict):
            if sequence_key in walks:
                return walks[sequence_key], f"{subj}__{sequence_key}"
            
            idx_str = sequence_key.split("_")[-1]
            for walk_k in walks.keys():
                if walk_k.endswith(f"_{idx_str}") or walk_k == sequence_key:
                    return walks[walk_k], f"{subj}__{walk_k}"

    raise KeyError(f"No valid sequences found in PKL data for sequence_key: '{sequence_key}'")


def compute_mesh_vertices(seq_data, smpl_model_path, faces_path, frame_idx=None):
    import torch
    from smplx.body_models import SMPL

    pose = seq_data["pose"]
    trans = seq_data["trans"]
    betas = seq_data.get("beta", np.zeros((1, 10), dtype=np.float32))
    T = pose.shape[0]

    pose_t = torch.tensor(pose, dtype=torch.float32)
    trans_t = torch.tensor(trans, dtype=torch.float32)
    betas_t = torch.tensor(np.tile(betas, (T, 1)) if betas.shape[0] != T else betas, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smpl = SMPL(model_path=smpl_model_path, num_betas=10).to(device)

    faces_file = Path(faces_path)
    if not faces_file.exists():
        faces_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(faces_file, smpl.faces)

    with torch.no_grad():
        out = smpl(
            betas=betas_t.to(device),
            body_pose=pose_t[:, 3:72].to(device),
            global_orient=pose_t[:, :3].to(device)
        )
        vertices = (out.vertices + trans_t.to(device)[:, None, :]).cpu().numpy()

    # If sequence requested, return all frames. If single frame, slice it but keep the (1, V, 3) dimension
    if frame_idx is not None:
        idx = max(0, min(frame_idx, T - 1))
        return vertices[[idx]], T, "Single"
    else:
        return vertices, T, "Sequence"


def trigger_blender_subprocess(args):
    import joblib

    print(f"\n--- [Pipeline] Loading SMPL dataset: {args.pkl_path} ---")
    pkl_file = Path(args.pkl_path).resolve()
    pkl_data = joblib.load(pkl_file)
    seq_data, resolved_key = find_sequence_in_pkl(pkl_data, args.sequence_key)

    # Compute 3D mesh vertices
    vertices_array, total_frames, mode = compute_mesh_vertices(
        seq_data, args.smpl_model_path, args.faces_path, args.frame_idx
    )
    print(f"  -> Extracted {mode}: '{resolved_key}' (Vertices Shape: {vertices_array.shape})")

    temp_vert_path = Path("temp_blender_vertices.npy").resolve()
    np.save(temp_vert_path, vertices_array)

    # Resolve and ensure output folder exists
    if args.output_path is not None:
        out_file = Path(args.output_path).resolve()
    else:
        out_dir = Path("thesis/visualizations") / args.model_name / "rendered_smpl"
        out_file = (out_dir / f"{resolved_key}.png").resolve()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "blender", "--background", "--python", Path(__file__).resolve().as_posix(), "--",
        "--temp_vertices_path", temp_vert_path.as_posix(),
        "--faces_path", Path(args.faces_path).resolve().as_posix(),
        "--output_path", out_file.as_posix(),
        "--sequence_key", resolved_key,
        "--render_engine", args.render_engine,
        "--resolution", str(args.resolution)
    ]
    if args.frame_idx is not None:
        cmd.extend(["--frame_idx", str(args.frame_idx)])

    print(f"\n[Python 3.9] Triggering Blender Subprocess...")
    try:
        subprocess.run(cmd, check=True)
        print(f"\n[Python 3.9] Render Process Complete!")
    except subprocess.CalledProcessError as e:
        print(f"\n[Python 3.9] ERROR: Blender process failed with exit code {e.returncode}")
    finally:
        if temp_vert_path.exists():
            temp_vert_path.unlink()

# blender python logic 
def setup_blender_scene(resolution, engine):
    bpy.ops.wm.read_factory_settings(use_empty=True)

    bpy.context.scene.render.engine = engine
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.film_transparent = False
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'

    # FIXED CAMERA: Pulled back and up to capture the whole head
    bpy.ops.object.camera_add(location=(3.5, -3.5, 3.0))
    camera = bpy.context.object
    camera.rotation_euler = (math.radians(65), 0.0, math.radians(45))
    bpy.context.scene.camera = camera

    # Solid Light-Grey Studio Floor
    bpy.ops.mesh.primitive_plane_add(size=50, location=(0, 0, 0))
    floor = bpy.context.object
    floor_mat = bpy.data.materials.new(name="StudioFloor")
    floor_mat.use_nodes = True
    floor_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
    floor_mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.9
    floor.data.materials.append(floor_mat)

    # Lighting
    bpy.ops.object.light_add(type="SUN", location=(5, 5, 5))
    sun = bpy.context.object
    sun.data.energy = 4.0
    sun.rotation_euler = (math.radians(45), math.radians(10), math.radians(45))

    bpy.ops.object.light_add(type="AREA", location=(-5, -5, 3))
    fill = bpy.context.object
    fill.data.energy = 100.0
    fill.data.size = 5.0

    return camera

def create_mld_golden_material():
    mat = bpy.data.materials.new(name="GoldenSMPL")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.85, 0.60, 0.15, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.35
    bsdf.inputs["Metallic"].default_value = 0.15
    return mat

def build_smpl_mesh(vertices, faces, name="SMPL_Character"):
    mesh = bpy.data.meshes.new(name + "_Mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    for poly in mesh.polygons:
        poly.use_smooth = True
    return obj

def main_blender(args):
    print(f"\n[Blender] Processing scene for: {args.sequence_key}")

    vertices_array = np.load(args.temp_vertices_path) # Shape: (T, 6893, 3)
    faces = np.load(args.faces_path)
    num_frames = vertices_array.shape[0]

    setup_blender_scene(args.resolution, args.render_engine)
    camera = bpy.context.scene.camera

    # Calculate the center of the entire walking trajectory across all T frames
    center_x = np.mean(vertices_array[:, :, 0])
    center_y = np.mean(-vertices_array[:, :, 2])

    # Position the camera relative to this global center point
    camera.location.x = center_x + 4.5
    camera.location.y = center_y - 4.5
    camera.location.z = 3.5
    
    # Initialize the mesh using the first frame
    character_obj = build_smpl_mesh(vertices_array[0], faces)
    character_obj.data.materials.append(create_mld_golden_material())
    out_path = Path(args.output_path)

    # Loop over the T dimension and render each frame
    for t in range(num_frames):
        character_obj.data.vertices.foreach_set("co", vertices_array[t].flatten())
        character_obj.data.update()
        
        # Reset transforms, orient Z-up, update View Layer
        character_obj.rotation_euler = (math.radians(90), 0, 0)
        character_obj.location.z = -np.min(vertices_array[t, :, 1])

        if num_frames == 1:
            bpy.context.scene.render.filepath = out_path.as_posix()
        else:
            frame_path = out_path.with_name(f"{out_path.stem}_f{t:03d}{out_path.suffix}")
            bpy.context.scene.render.filepath = frame_path.as_posix()

        bpy.ops.render.render(write_still=True)
        print(f"[Blender] Rendered frame {t+1}/{num_frames} -> {bpy.context.scene.render.filepath}")


if __name__ == "__main__":
    if not INSIDE_BLENDER:
        trigger_blender_subprocess(parse_arguments(inside_blender=False))
    else:
        main_blender(parse_arguments(inside_blender=True))
