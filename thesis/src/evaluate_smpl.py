import json
import torch
import numpy as np
from collections import defaultdict
from pathlib import Path

class SMPLEvaluator:
    def __init__(self):
        """Evaluator for 6D SMPL pose sequences using Geodesic Distance on SO(3)."""
        pass

    def _convert_6d_to_rmat(self, pose_6d_tensor):
        """Gram-Schmidt to convert 6d pose tensors to (T, 3, 3) rotation matrices.
        
        Supports batched training tensors (B, T, J, 6) or single sequences (T, J, 6).
        Constructs the rotation matrix by stacking rows due to original CARE-PD formatting.
        """
        if isinstance(pose_6d_tensor, np.ndarray):
            pose_6d_tensor = torch.tensor(pose_6d_tensor, dtype=torch.float32)

        # 25th joint shouldn't exist during evaluation
        if pose_6d_tensor.shape[-2] == 25:
            raise ValueError("Data to evaluate contains 25 joints. Should be 24.")
            
        v1 = pose_6d_tensor[..., :3]
        v2 = pose_6d_tensor[..., 3:]
        
        x = torch.nn.functional.normalize(v1, dim=-1)
        y_raw = v2 - (torch.sum(x * v2, dim=-1, keepdim=True) * x)
        y = torch.nn.functional.normalize(y_raw, dim=-1)
        z = torch.cross(x, y, dim=-1)
        
        # Stack into (T, 3, 3) rotation matrices
        # and handle x,y,z as rows because CARE-PD for some reason 
        # decided to format 6D rotations as first 2 rows instead of first 2 columns like normal humans
        rot_mats = torch.stack([x, y, z], dim=-2)
        return rot_mats

    @torch.no_grad()
    def compute_mpjae(self, gt_6d, gen_6d):
        """Computes the Mean Per Joint Angular Error (MPJAE) using Geodesic Distance.

        Safely handles arbitrary leading dimensions (e.g., T, J, 6 or B, T, J, 6).
        Returns error in radians.
        """
        R_gt = self._convert_6d_to_rmat(gt_6d)   
        R_gen = self._convert_6d_to_rmat(gen_6d) 

        # Truncate to the length of the shortest sequence along the Temporal (T) dimension
        # In a (B, T, J, 3, 3) tensor, T is at index -4. In a (T, J, 3, 3) tensor, T is at -3.
        t_dim = -4 if R_gt.dim() == 5 else -3
        min_frames = min(R_gt.shape[t_dim], R_gen.shape[t_dim])
        
        if R_gt.dim() == 4:
            R_gt = R_gt[:min_frames]
            R_gen = R_gen[:min_frames]
        else:
            R_gt = R_gt[:, :min_frames]
            R_gen = R_gen[:, :min_frames]

        # Compute relative rotation matrix: R_rel = R_gen * R_gt^T
        R_rel = torch.matmul(R_gen, R_gt.transpose(-1, -2))

        # Get trace (sum of diagonal elements) for each 3x3 matrix
        trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

        # Compute cosine of the angle and clamp value for numerical stability
        cos_theta = (trace - 1.0) / 2.0
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

        # Compute geodesic distance and mean over all frames/joints/batches
        d_geo = torch.acos(cos_theta)
        mpjae = torch.mean(d_geo).item()
        
        return mpjae

    def evaluate_and_cache(self, gt_npz_path, gen_npz_path, labels_path, cache_output_path, verbose=True):
        """Loads unified GT/Gen 6D datasets, computes MPJAE per severity class, caches result."""
        with open(labels_path, 'r') as f:
            labels = json.load(f)["key_to_severity"]

        gt_data = np.load(gt_npz_path, allow_pickle=True)
        gen_data = np.load(gen_npz_path, allow_pickle=True)
        
        gt_data = gt_data['arr_0'].item() if 'arr_0' in gt_data.files else {k: gt_data[k] for k in gt_data.files}
        gen_data = gen_data['arr_0'].item() if 'arr_0' in gen_data.files else {k: gen_data[k] for k in gen_data.files}

        common_keys = [k for k in gt_data.keys() if k in gen_data.keys() and not k.endswith('_trans')]
        
        if not common_keys:
            if verbose:
                print("Error: No matching pose sequences found between GT and Gen datasets.")
            return None

        results = defaultdict(list)
        
        for k in common_keys:
            mpjae = self.compute_mpjae(gt_data[k], gen_data[k])
            results["Overall"].append(mpjae)
            
            # Map sequence key to clinical severity
            if k in labels:
                sev = labels[k]
                results[f"Class {sev}"].append(mpjae)

        if verbose:
            print(f"\n--- SMPL SO(3) Evaluation ---")
            print(f"Evaluated {len(common_keys)} matching sequences.")
        
        summary_results = {}
        
        # Display sorted results and build summary dictionary
        for cls in ["Overall"] + sorted([c for c in results.keys() if c != "Overall"]):
            mean_mpjae = float(np.mean(results[cls]))
            summary_results[cls] = mean_mpjae
            
            if verbose:
                count = len(results[cls])
                print(f"  -> {cls} (N={count}): {mean_mpjae:.6f} rad")
                
        # Cache results to disk
        out_path = Path(cache_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save both the final means and the raw distributions (for future plotting)
        cache_data = {
            "summary_means": summary_results,
            "raw_distributions": {k: [float(x) for x in v] for k, v in results.items()}
        }
        
        with open(out_path, 'w') as f:
            json.dump(cache_data, f, indent=4)
            
        if verbose:
            print(f"Saved MPJAE evaluation results to: {out_path}")
            
        return summary_results