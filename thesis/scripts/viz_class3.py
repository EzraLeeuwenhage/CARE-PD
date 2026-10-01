from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from thesis.src.care_pd.visualize_skel_walk_func import (
    visualize_sequence,
    h36m_joint_paths,
)


def resolve_default_paths() -> tuple[str, str]:
    """Resolves default file paths for raw H36M coordinates and metadata."""
    default_npz = Path("thesis/data/raw/PD-GaM/h36m/PD-GaM_h36m_rot_trans_canonical.npz")
    labels = Path("thesis/data/metadata/pd_gam_labels.json")
    return str(default_npz), str(labels)


def main():
    default_npz, default_labels = resolve_default_paths()

    parser = argparse.ArgumentParser(description="Visualize Class 3 walking sequences from 3D H36M coordinates.")
    parser.add_argument("-n", "--npzf", type=str, default=default_npz, help="Path to ground truth H36M NPZ file.")
    parser.add_argument("-l", "--labels", type=str, default=default_labels, help="Path to severity labels JSON file.")
    parser.add_argument("-fps", "--fps", type=int, default=30, help="Frames per second for playback.")
    parser.add_argument("-p", "--projection", type=str, default="3d", choices=["2d", "3d"], help="Projection type.")
    parser.add_argument("--save-gif", action="store_true", help="Save visualization to GIF instead of interactive display.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of sequences to display.")
    args = parser.parse_args()

    npz_path = Path(args.npzf)
    labels_path = Path(args.labels)

    if not npz_path.exists():
        raise FileNotFoundError(
            f"H36M file not found at '{npz_path}'. "
            "Please run 'python -m thesis.scripts.convert_raw_smpl_to_h36m' first."
        )
    if not labels_path.exists():
        raise FileNotFoundError(f"Severity labels metadata file not found at '{labels_path}'.")

    print(f"Loading H36M data from:   {npz_path}")
    print(f"Loading metadata from:    {labels_path}")

    # 1. Load H36M sequences
    raw_data = np.load(npz_path, allow_pickle=True)
    fname = npz_path.name

    # 2. Load severity labels
    with open(labels_path, "r") as f:
        meta = json.load(f)
        key_to_severity = meta.get("key_to_severity", meta)

    # 3. Filter strictly for Class 3 walking sequences
    class3_sequences = {}
    for key in raw_data.files:
        if key.endswith("_trans") or key.endswith("_frame_ids"):
            continue

        base_key = key.split("_down")[0] if "_down" in key else key
        base_key = base_key.replace("generated_walk_", "")

        sev = key_to_severity.get(base_key, key_to_severity.get(key, None))
        if sev == 3:
            class3_sequences[key] = raw_data[key]

    total_c3 = len(class3_sequences)
    print(f"\nFound {total_c3} Class 3 sequences in dataset.")
    if total_c3 == 0:
        print("No Class 3 sequences found. Verify key matching with the labels JSON.")
        return

    # 4. Sequential visualization loop using H36M joint paths
    keys_to_play = list(class3_sequences.keys())
    if args.limit:
        keys_to_play = keys_to_play[:args.limit]
        print(f"Limiting visualization to first {args.limit} sequences.")

    for idx, name in enumerate(keys_to_play, 1):
        seq = class3_sequences[name]  # Expected shape: (T, 17, 3)

        num_frames = seq.shape[0]
        duration_sec = num_frames / args.fps
        print(f"\n[{idx}/{len(keys_to_play)}] Visualizing: {name}")
        print(f"    * Frames: {num_frames} ({duration_sec:.2f}s) | Skeleton: H36M (17 joints) | Severity: Class 3")

        visualize_sequence(
            seq,
            f"{name} (Class 3)\nfrom {fname}",
            show_joint_indexes=True,
            joint_paths=h36m_joint_paths,
            projection=args.projection,
            fps=args.fps,
            invert=None,
            minmax=None,
            save_gif=args.save_gif,
            heel_strikes=None,
            severity=3,
        )


if __name__ == "__main__":
    main()