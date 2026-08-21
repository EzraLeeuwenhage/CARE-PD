import json
import torch
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
from sklearn.decomposition import PCA

class SMPLEvaluator:
    def __init__(self, fps=30):
        """Evaluator for 6D SMPL pose sequences using Geodesic Distance on SO(3)."""
        self.fps = fps
        # Standard 24 SMPL model joint names ordered by index
        self.JOINT_NAMES = [
            'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee',
            'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot',
            'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
            'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
        ]
        
        # Self-defined categories for analysis
        self.JOINT_GROUPS = {
            'Overall': list(range(24)),
            'Lower Body': [0, 1, 2, 4, 5, 7, 8, 10, 11],
            'Upper Body': [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            'Hips': [1, 2],
            'Knees': [4, 5],
            'Ankles': [7, 8],
            'Shoulders': [16, 17],
            'Left Body': [1, 4, 7, 10, 13, 16, 18, 20, 22],
            'Right Body': [2, 5, 8, 11, 14, 17, 19, 21, 23]
        }

        self.HARD_JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 16, 17, 18, 19]

    # ---------
    # MPJAE
    # ---------
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
    def compute_mpjae(self, gt_6d, gen_6d, return_per_joint=False):
        """Computes the Mean Per Joint Angular Error (MPJAE) using Geodesic Distance.

        Args:
            gt_6d: Ground truth tensor (..., J, 6)
            gen_6d: Generated tensor (..., J, 6)
            return_per_joint: If True, returns array of shape (24,) with error per joint.
                              If False, returns overall scalar float (radians).
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
        
        if not return_per_joint:
            # Compute mean error over 12 hardest joints to learn
            return torch.mean(d_geo[..., self.HARD_JOINTS]).item()
            
        # Collapse all leading dimensions EXCEPT the last joint dimension (dim=-1)
        dims_to_collapse = tuple(range(d_geo.dim() - 1))
        per_joint_mpjae = torch.mean(d_geo, dim=dims_to_collapse) # Shape: (24,)
        
        return per_joint_mpjae.cpu().numpy()

    # -------------------
    # Arm Swing Asymmetry
    # -------------------
    def _compute_pairwise_geodesic(self, R1, R2):
        """Pairwise geodesic distance between two (3, 3) numpy rotation matrices."""
        R_rel = np.dot(R1, R2.T)
        trace = np.trace(R_rel)
        cos_theta = np.clip((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
        return np.arccos(cos_theta)

    def extract_and_validate_arm_swing(self, rot_matrices_3x3, prominence=0.05):
        """
        Extracts 1D pendular swing via PCA and cross-validates against SO(3) Geodesic Distance.
        """
        rotvecs = Rotation.from_matrix(rot_matrices_3x3).as_rotvec()
        
        # 1D PCA Pendulum Projection
        pca = PCA(n_components=1)
        swing_1d = pca.fit_transform(rotvecs).flatten()
        
        peaks, _ = find_peaks(swing_1d, prominence=prominence)
        valleys, _ = find_peaks(-swing_1d, prominence=prominence)
        
        cycle_validations = []
        amplitudes_pca = []
        
        # Cross-validate each peak-valley pair with max geodesic distance
        for p in peaks:
            prior_valleys = valleys[valleys < p]
            if len(prior_valleys) == 0:
                continue
            v = prior_valleys[-1]
            
            # PCA amplitude (scalar projection)
            pca_amp = abs(swing_1d[p] - swing_1d[v])
            amplitudes_pca.append(pca_amp)
            
            # Ground-truth SO(3) Geodesic ROM
            geo_rom = self._compute_pairwise_geodesic(rot_matrices_3x3[p], rot_matrices_3x3[v])
            
            # verify that peak and valley are aligned with max geodesic distance frames
            window_matrices = rot_matrices_3x3[v:p+1]
            geo_distances_from_v = [self._compute_pairwise_geodesic(rot_matrices_3x3[v], R_t) for R_t in window_matrices]
            geo_argmax_frame = v + np.argmax(geo_distances_from_v)
            
            frame_aligned = (geo_argmax_frame == p)
            amplitude_error = abs(pca_amp - geo_rom) / geo_rom if geo_rom > 0 else 0.0
            
            cycle_validations.append({
                "valley_frame": int(v),
                "peak_frame": int(p),
                "geo_argmax_frame": int(geo_argmax_frame),
                "frame_aligned": bool(frame_aligned),
                "pca_amplitude_rad": float(pca_amp),
                "geo_rom_rad": float(geo_rom),
                "rel_error": float(amplitude_error)
            })
            
        mean_rom = np.mean(amplitudes_pca) if amplitudes_pca else 0.0
        return mean_rom, cycle_validations

    def compute_arm_swing_asymmetry(self, seq_6d, prominence=0.05):
        """
        Wrapper to compute L/R swing asymmetry directly from a 6D tensor sequence.
        ROM_L/R are the means of the ROM for each sequence.
        Asymmetry is the absolute difference between L and R mean ROM.

        Args:
            seq_6d: numpy array or tensor of shape (T, 24, 6)
        """
        R_seq = self._convert_6d_to_rmat(seq_6d).numpy()
        
        L_shoulder_idx = self.JOINT_NAMES.index('L_Shoulder')
        R_shoulder_idx = self.JOINT_NAMES.index('R_Shoulder')
        
        rom_L, val_L = self.extract_and_validate_arm_swing(R_seq[:, L_shoulder_idx, :, :], prominence)
        rom_R, val_R = self.extract_and_validate_arm_swing(R_seq[:, R_shoulder_idx, :, :], prominence)
        
        # Calculate Robinson Symmetry Index
        denominator = rom_L + rom_R + 1e-7
        si_asymmetry = (2.0 * abs(rom_L - rom_R) / denominator) * 100.0

        return rom_L, rom_R, si_asymmetry, val_L, val_R

    # ---------
    # SPARC
    # ---------
    def _compute_single_sparc(self, a, padlevel=4, fc_max=5.0, amp_th=0.01):
        """Computes Spectral Arc Length for a single 1D signal. (TODO: cite SPARC paper)"""
        if len(a) < 2 or np.all(a == 0):
            return np.nan, None, None, None
            
        nfft = int(pow(2, np.ceil(np.log2(len(a))) + padlevel))
        
        A = np.abs(np.fft.rfft(a, n=nfft))
        A_0 = A[0]
        if A_0 == 0:
            return 0.0, None, None, None
            
        A_norm = A / A_0
        f = np.fft.rfftfreq(nfft, d=1.0/self.fps)
        
        # Adaptive cutoff bounded by fc_max (typically 5Hz for human gait)
        valid_f_mask = f <= fc_max
        f_search = f[valid_f_mask]
        A_search = A_norm[valid_f_mask]
        
        # Find minimum frequency omega where A_norm(r) < amp_th for all r > omega
        above_th_idxs = np.where(A_search >= amp_th)[0]
        if len(above_th_idxs) > 0:
            idx_c = above_th_idxs[-1]
            fc_adj = f_search[idx_c]
        else:
            idx_c = 0
            fc_adj = f_search[0]
            
        if fc_adj == 0:
            return 0.0, f, A_norm, fc_adj
            
        f_int = f[:idx_c + 1]
        A_int = A_norm[:idx_c + 1]
        
        # Discrete integral
        dx = np.diff(f_int) / fc_adj
        dy = np.diff(A_int)
        arc_length = -np.sum(np.sqrt(dx**2 + dy**2))
        
        return arc_length, f, A_norm, fc_adj

    def compute_sparc_for_sequence(self, seq_6d, plot_joint=None, plot_prefix=""):
        """Extracts angular velocity magnitude and computes SPARC for all 24 joints."""
        rot_mats = self._convert_6d_to_rmat(seq_6d).numpy()
        T, J, _, _ = rot_mats.shape
        if T < 2:
            return np.full(J, np.nan)
            
        R_t = rot_mats[:-1]
        R_next = rot_mats[1:]
        
        # Transpose each 3x3 matrix in R_t: (T-1, J, 3, 3)
        R_t_T = np.swapaxes(R_t, -1, -2)
        R_rel = np.matmul(R_t_T, R_next)
        
        R_rel_flat = R_rel.reshape(-1, 3, 3)
        rotvecs = Rotation.from_matrix(R_rel_flat).as_rotvec()
        rotvecs = rotvecs.reshape(T-1, J, 3)
        
        omega_t = rotvecs * self.fps
        a_t = np.linalg.norm(omega_t, axis=-1)
        
        sparc_vals = np.zeros(J)
        for j in range(J):
            arc_len, f, A_norm, fc_adj = self._compute_single_sparc(a_t[:, j])
            sparc_vals[j] = arc_len
            
            if plot_joint and self.JOINT_NAMES[j] == plot_joint and f is not None:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8, 4))
                plt.plot(f, A_norm, label=f'Normalized Spectrum ({plot_prefix})')
                plt.axvline(fc_adj, color='r', linestyle='--', label=f'Adaptive Cutoff = {fc_adj:.2f} Hz')
                plt.title(f'SPARC Frequency Spectrum - {plot_joint} ({plot_prefix})\nSPARC: {arc_len:.4f}')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Normalized Magnitude')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'sparc_spectrum_{plot_prefix}_{plot_joint}.png')
                plt.close()
                print(f"Saved SPARC spectrum plot for {plot_joint} ({plot_prefix})")
                
        return sparc_vals

    def evaluate_and_cache(self, gt_npz_path, gen_npz_path, labels_path, cache_output_path, verbose=True, plot_sparc_joint=None):
        """Loads unified GT/Gen 6D datasets, computes MPJAE for all categories/joints, caches result."""
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

        # structure: results[severity_class][metric_name] = [list of sequence errors]
        results = defaultdict(lambda: defaultdict(list))
        per_sequence_results = {}
        misaligned_records = []
        total_cycles_count = 0
        misaligned_count = 0
        has_plotted_sparc = False
        
        for k in common_keys:
            per_joint_err = self.compute_mpjae(gt_data[k], gen_data[k], return_per_joint=True) # (24,)
            sev = labels.get(k, "Unknown")
            
            plot_this_iter = plot_sparc_joint if not has_plotted_sparc else None
            
            # SPARC evaluation
            sparc_gt = self.compute_sparc_for_sequence(gt_data[k], plot_joint=plot_this_iter, plot_prefix="GT")
            sparc_gen = self.compute_sparc_for_sequence(gen_data[k], plot_joint=plot_this_iter, plot_prefix="Gen")
            if plot_this_iter:
                has_plotted_sparc = True

            # broad category errors & SPARC
            for group_name, joint_indices in self.JOINT_GROUPS.items():
                group_val = float(np.mean(per_joint_err[joint_indices]))
                sparc_gt_val = float(np.mean(sparc_gt[joint_indices]))
                sparc_gen_val = float(np.mean(sparc_gen[joint_indices]))
                
                results["Overall"][group_name].append(group_val)
                results["Overall"][f"GT_SPARC_{group_name}"].append(sparc_gt_val)
                results["Overall"][f"Gen_SPARC_{group_name}"].append(sparc_gen_val)
                
                if sev != "Unknown":
                    results[f"Class {sev}"][group_name].append(group_val)
                    results[f"Class {sev}"][f"GT_SPARC_{group_name}"].append(sparc_gt_val)
                    results[f"Class {sev}"][f"Gen_SPARC_{group_name}"].append(sparc_gen_val)

            # individual joint errors & SPARC
            for idx, joint_name in enumerate(self.JOINT_NAMES):
                joint_val = float(per_joint_err[idx])
                s_gt_val = float(sparc_gt[idx])
                s_gen_val = float(sparc_gen[idx])
                
                results["Overall"][joint_name].append(joint_val)
                results["Overall"][f"GT_SPARC_{joint_name}"].append(s_gt_val)
                results["Overall"][f"Gen_SPARC_{joint_name}"].append(s_gen_val)
                
                if sev != "Unknown":
                    results[f"Class {sev}"][joint_name].append(joint_val)
                    results[f"Class {sev}"][f"GT_SPARC_{joint_name}"].append(s_gt_val)
                    results[f"Class {sev}"][f"Gen_SPARC_{joint_name}"].append(s_gen_val)

            rom_L_gt, rom_R_gt, si_asym_gt, val_L_gt, val_R_gt = self.compute_arm_swing_asymmetry(gt_data[k], prominence=0.05)
            rom_L_gen, rom_R_gen, si_asym_gen, val_L_gen, val_R_gen = self.compute_arm_swing_asymmetry(gen_data[k], prominence=0.05)

            arm_metrics = {
                "GT_ROM_L": float(rom_L_gt),
                "GT_ROM_R": float(rom_R_gt),
                "GT_Symmetry_Index": float(si_asym_gt),
                "Gen_ROM_L": float(rom_L_gen),
                "Gen_ROM_R": float(rom_R_gen),
                "Gen_Symmetry_Index": float(si_asym_gen),
                "Symmetry_Index_Error": float(abs(si_asym_gt - si_asym_gen))
            }

            for metric_name, val in arm_metrics.items():
                results["Overall"][metric_name].append(val)
                if sev != "Unknown":
                    results[f"Class {sev}"][metric_name].append(val)

            for split_name, side_name, val_list in [
                ("GT", "L", val_L_gt),
                ("GT", "R", val_R_gt),
                ("Gen", "L", val_L_gen),
                ("Gen", "R", val_R_gen)
            ]:
                for cycle in val_list:
                    total_cycles_count += 1
                    if not cycle["frame_aligned"]:
                        misaligned_count += 1
                        misaligned_records.append({
                            "sequence": k,
                            "split": split_name,
                            "side": side_name,
                            "valley_frame": cycle["valley_frame"],
                            "peak_frame": cycle["peak_frame"],
                            "geo_argmax_frame": cycle["geo_argmax_frame"],
                            "rel_error": cycle["rel_error"]
                        })

            per_sequence_results[k] = {
                "severity": sev,
                "overall_mpjae": float(np.mean(per_joint_err[self.HARD_JOINTS])),
                "per_joint_mpjae": {j_name: float(per_joint_err[i]) for i, j_name in enumerate(self.JOINT_NAMES)},
                "sparc": {
                    "gt": {
                        "overall": float(np.mean(sparc_gt)),
                        "per_joint": {j_name: float(sparc_gt[i]) for i, j_name in enumerate(self.JOINT_NAMES)}
                    },
                    "gen": {
                        "overall": float(np.mean(sparc_gen)),
                        "per_joint": {j_name: float(sparc_gen[i]) for i, j_name in enumerate(self.JOINT_NAMES)}
                    }
                },
                "arm_swing": {
                    "gt": {
                        "rom_L": float(rom_L_gt),
                        "rom_R": float(rom_R_gt),
                        "symmetry_index": float(si_asym_gt),
                        "cycles_L": val_L_gt,
                        "cycles_R": val_R_gt
                    },
                    "gen": {
                        "rom_L": float(rom_L_gen),
                        "rom_R": float(rom_R_gen),
                        "symmetry_index": float(si_asym_gen),
                        "cycles_L": val_L_gen,
                        "cycles_R": val_R_gen
                    }
                }
            }

        if verbose:
            print(f"\n--- SMPL SO(3) Evaluation ---")
            print(f"Evaluated {len(common_keys)} matching sequences.")
            print("\nOverall Category Means (radians):")
            for group_name in self.JOINT_GROUPS.keys():
                mean_val = np.mean(results["Overall"][group_name])
                print(f"  -> {group_name:<20}: {mean_val:.6f} rad")

            print("\nSPARC Smoothness Summary:")
            print(f"  -> {'GT SPARC Overall':<20}: {np.mean(results['Overall']['GT_SPARC_Overall']):.4f} | {'Gen SPARC Overall':<20}: {np.mean(results['Overall']['Gen_SPARC_Overall']):.4f}")

            print("\nArm Swing & Asymmetry Summary (radians):")
            print(f"  -> {'GT L_ROM':<20}: {np.mean(results['Overall']['GT_ROM_L']):.4f} rad | {'Gen L_ROM':<20}: {np.mean(results['Overall']['Gen_ROM_L']):.4f} rad")
            print(f"  -> {'GT R_ROM':<20}: {np.mean(results['Overall']['GT_ROM_R']):.4f} rad | {'Gen R_ROM':<20}: {np.mean(results['Overall']['Gen_ROM_R']):.4f} rad")
            print(f"  -> {'GT Symmetry Index':<20}: {np.mean(results['Overall']['GT_Symmetry_Index']):.4f} % | {'Gen Symmetry Index':<20}: {np.mean(results['Overall']['Gen_Symmetry_Index']):.4f} %")
            print(f"  -> {'Symmetry Index Error':<20}: {np.mean(results['Overall']['Symmetry_Index_Error']):.4f} %")

            print(f"\nArm Swing SO(3) Validation Alignment Report:")
            print(f"  -> Total swing cycles evaluated: {total_cycles_count}")
            print(f"  -> Misaligned cycles (frame_aligned == False): {misaligned_count} ({(misaligned_count / total_cycles_count * 100) if total_cycles_count > 0 else 0.0:.2f}%)")
            if misaligned_count > 0:
                print(f"  -> Logged {len(misaligned_records)} misaligned cycle entries.")
        
        # Build summary means dictionary
        summary_results = {}
        for cls_key, metrics_dict in results.items():
            summary_results[cls_key] = {
                metric_name: float(np.mean(vals))
                for metric_name, vals in metrics_dict.items()
            }
                
        # Store results
        out_path = Path(cache_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        cache_data = {
            "summary_means": summary_results,
            "raw_distributions": {
                cls_key: {m: [float(x) for x in vals] for m, vals in metrics_dict.items()}
                for cls_key, metrics_dict in results.items()
            },
            "per_sequence_results": per_sequence_results,
            "arm_swing_validation": {
                "total_cycles_evaluated": total_cycles_count,
                "misaligned_count": misaligned_count,
                "misaligned_records": misaligned_records
            }
        }
        
        with open(out_path, 'w') as f:
            json.dump(cache_data, f, indent=4)
            
        if verbose:
            print(f"\nSaved detailed MPJAE evaluation results to: {out_path}")
            
        return summary_results


if __name__ == "__main__":
    evaluator = SMPLEvaluator(fps=30)
    base_dir = Path("thesis/data/processed/JointModel-MLP-Baseline")
    
    gt_path = base_dir / "6D_SMPL" / "ground_truth_6d.npz"
    gen_path = base_dir / "6D_SMPL" / "generated_6d.npz"
    labels_path = base_dir / "h36m" / "gen_labels.json"
    output_path = base_dir / "evaluation" / "smpl_evaluation.json"

    print(f"--- Running SMPL SO(3) Evaluation ---")
    print(f"GT Data path:  {gt_path}")
    print(f"Gen Data path: {gen_path}")
    print(f"Labels path:   {labels_path}")
    print(f"Output path:   {output_path}")

    smpl_summary = evaluator.evaluate_and_cache(
        gt_npz_path=str(gt_path),
        gen_npz_path=str(gen_path),
        labels_path=str(labels_path),
        cache_output_path=str(output_path),
        verbose=True,
        plot_sparc_joint='R_Shoulder'
    )

    print("\nSMPL SO(3) Evaluation Complete!")
    for severity, metrics in smpl_summary.items():
        print(f"  -> {severity:<10}: Overall MPJAE = {metrics['Overall']:.6f} rad | \
              GT SPARC = {metrics['GT_SPARC_Overall']:.4f} | Gen SPARC = {metrics['Gen_SPARC_Overall']:.4f}")