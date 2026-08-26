"""
Usage:
    python -m thesis.utils.render_smpl_gif \
        --sequence_key "GT__gt_000" \
        --labels_path "thesis/data/processed/JointModel-MLP-Baseline/h36m/gt_labels.json"
"""

import argparse
import json
import numpy as np
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

def parse_args():
    parser = argparse.ArgumentParser(description="Create an annotated GIF from rendered SMPL frames.")
    parser.add_argument("--model_name", type=str, default="JointModel-MLP-Baseline",
                        help="Model directory name under thesis/visualizations")
    parser.add_argument("--sequence_key", type=str, required=True, 
                        help="Sequence key to process (e.g., GT__gt_000 or GEN__gen_001)")
    parser.add_argument("--labels_path", type=str, default=None, 
                        help="Path to gt_labels.json or gen_labels.json to extract the severity score")
    parser.add_argument("--fps", type=int, default=15, 
                        help="Frames per second for the output GIF")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Define directories
    vis_dir = Path("thesis/visualizations") / args.model_name
    frames_dir = vis_dir / "rendered_smpl"
    frame_files = sorted(frames_dir.glob(f"{args.sequence_key}_f*.png"))
    
    if not frame_files:
        print(f"ERROR: No frame images found for '{args.sequence_key}' in '{frames_dir}'.")
        print("Did you render the full sequence in Blender first?")
        return
        
    print(f"Found {len(frame_files)} frames for '{args.sequence_key}'. Compiling...")

    severity = "Unknown"
    if args.labels_path and Path(args.labels_path).exists():
        try:
            with open(args.labels_path, 'r') as f:
                labels_dict = json.load(f).get("key_to_severity", {})
                severity = labels_dict.get(args.sequence_key, "Unknown")
        except Exception as e:
            print(f"Warning: Could not parse labels file. {e}")
    else:
        print("Note: No valid --labels_path provided. Severity score will show as 'Unknown'.")

    # Try to load font
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    frames = []
    for t, frame_path in enumerate(frame_files):
        img = Image.open(frame_path).convert("RGBA")
        txt_overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt_overlay)

        text = f"Sequence: {args.sequence_key}\nSeverity Class: {severity}\nFrame: {t:03d}"
        
        # Draw text outline for readability against grey background
        x, y = 20, 20
        shadow_color = (0, 0, 0, 255)
        text_color = (255, 255, 255, 255)
        draw.text((x-1, y-1), text, font=font, fill=shadow_color)
        draw.text((x+1, y-1), text, font=font, fill=shadow_color)
        draw.text((x-1, y+1), text, font=font, fill=shadow_color)
        draw.text((x+1, y+1), text, font=font, fill=shadow_color)
        
        # White main text
        draw.text((x, y), text, font=font, fill=text_color)
        img = Image.alpha_composite(img, txt_overlay)
        frames.append(np.array(img.convert("RGB")))

    out_path = frames_dir / f"{args.sequence_key}_smpl_render.gif"
    imageio.mimsave(out_path, frames, fps=args.fps, loop=0)
    print(f"\nSUCCESS! Animated GIF saved to: {out_path}")


if __name__ == "__main__":
    main()