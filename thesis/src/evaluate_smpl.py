import json
import torch
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy.spatial.transform import Rotation
from thesis.src.utils.geometry_utils import pose_to_rmat


class SMPLEvaluator:
    def __init__(self, fps: int = 30, nfft: int = 2048):
        """Evaluator for SMPL pose sequences using SO(3) Geodesic Distance and Clinical Gait Metrics."""
        self.fps = fps
        self.nfft = nfft
        self.trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

        # Standard 24 SMPL model joint names ordered by index
        self.JOINT_NAMES = [
            'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee',
            'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot',
            'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
            'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
        ]
        
        # Self-defined categories for MPJAE analysis
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

        self.HARD_MPJAE_JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 16, 17, 18, 19]

    # ---------
    # MPJAE
    # ---------
    @torch.no_grad()
    def compute_mpjae(self, gt_pose, gen_pose, return_per_joint=False):
        """Computes the Mean Per Joint Angular Error (MPJAE) using Geodesic Distance.

        Args:
            gt_pose: Ground truth tensor (..., J, D)
            gen_pose: Generated tensor (..., J, D)
            return_per_joint: If True, returns array of shape (24,) with error per joint.
                              If False, returns overall scalar float (radians).
        """
        assert gt_pose.shape == gen_pose.shape, (
            f"[MPJAE Error] Shape mismatch between Ground Truth {tuple(gt_pose.shape)} "
            f"and Generated {tuple(gen_pose.shape)}. Sequences must have identical lengths."
        )

        R_gt = pose_to_rmat(gt_pose)   
        R_gen = pose_to_rmat(gen_pose) 

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
            return torch.mean(d_geo[..., self.HARD_MPJAE_JOINTS]).item()
            
        # Collapse all leading dimensions EXCEPT the last joint dimension (dim=-1)
        dims_to_collapse = tuple(range(d_geo.dim() - 1))
        per_joint_mpjae = torch.mean(d_geo, dim=dims_to_collapse)
        return per_joint_mpjae.cpu().numpy()

    # -------------------------------------------------------------
    # CLINICAL GAIT METRICS (BRADYKINESIA, RIGIDITY, SMOOTHNESS)
    # -------------------------------------------------------------
    def compute_clinical_metrics(self, seq_pose):
        """Extracts the 5 core clinical metrics from sequence poses:
        1. Ankle Bradykinesia: min(L, R) physical amplitude AUC (0.5–8.0 Hz, deg/s·Hz)
        2. Spine Rigidity: mean(Spine2, Spine3) physical amplitude AUC (0.5–8.0 Hz, deg/s·Hz)
        3. Ankle SI: Robinson Symmetry Index on bilateral ankles (%)
        4. Wrist Smoothness AUC: max(L, R) relative PSD energy in jitter band (3.0–8.0 Hz, %)
        5. Hand Smoothness AUC: max(L, R) relative PSD energy in jitter band (3.0–8.0 Hz, %)
        """
        rot_mats = pose_to_rmat(seq_pose)
        if isinstance(rot_mats, torch.Tensor):
            rot_mats = rot_mats.detach().cpu().numpy()

        T, J, _, _ = rot_mats.shape
        if T < 2:
            return {
                "ankle_bradykinesia": np.nan,
                "spine_rigidity": np.nan,
                "ankle_si": np.nan,
                "wrist_smoothness_auc": np.nan,
                "hand_smoothness_auc": np.nan
            }

        # Calculate relative angular velocities on SO(3)
        R_t = rot_mats[:-1]
        R_next = rot_mats[1:]
        R_t_T = np.swapaxes(R_t, -1, -2)
        R_rel = np.matmul(R_t_T, R_next)

        R_rel_flat = R_rel.reshape(-1, 3, 3)
        rotvecs = Rotation.from_matrix(R_rel_flat).as_rotvec()
        rotvecs = rotvecs.reshape(T - 1, J, 3)

        omega_t = rotvecs * self.fps  # rad/s
        a_t = np.linalg.norm(omega_t, axis=-1)  # (T-1, J) in rad/s

        N = a_t.shape[0]
        RAD2DEG = 180.0 / np.pi
        f = np.fft.rfftfreq(self.nfft, d=1.0 / self.fps)

        mask_brady = (f >= 0.5) & (f <= 8.0)
        mask_jitter = (f > 3.0) & (f <= 8.0)

        def _compute_phys_auc(joint_idx: int) -> float:
            """Computes Cumulative Harmonic Amplitude (0.5-8.0 Hz) in deg/s·Hz to quantify bradykinesia."""
            velocity_rad_per_s = a_t[:, joint_idx]
            
            # Remove DC component (0 Hz) to normalize for walking speed
            velocity_centered = velocity_rad_per_s - np.mean(velocity_rad_per_s)
            
            # DFT using FFT algorithm magnitudes
            fft_magnitudes = np.abs(np.fft.rfft(velocity_centered, n=self.nfft))
            
            # Normalize by sequence length (2/N) for duration invariance, convert rad/s -> deg/s
            num_frames = len(velocity_centered)
            harmonic_amplitudes_deg = (2.0 / num_frames) * fft_magnitudes * RAD2DEG
            
            # Integrate physical amplitude across the functional gait band (0.5 - 8.0 Hz)
            band_amplitudes = harmonic_amplitudes_deg[mask_brady]
            band_frequencies = f[mask_brady]
            auc_deg_per_s_hz = float(self.trapz_fn(band_amplitudes, band_frequencies))
            
            return auc_deg_per_s_hz

        def _compute_smoothness_auc(joint_idx: int) -> float:
            """Computes Relative Kinetic Energy (%) consumed by 3.0-8.0 Hz tremor/jitter."""
            velocity_rad_per_s = a_t[:, joint_idx]
            
            # Remove DC component
            velocity_centered = velocity_rad_per_s - np.mean(velocity_rad_per_s)
            
            # FFT magnitudes
            fft_magnitudes = np.abs(np.fft.rfft(velocity_centered, n=self.nfft))
            
            # Square amplitudes to get Power Spectral Density (proportional to rotational kinetic energy)
            raw_power_spectrum = (fft_magnitudes ** 2) / self.nfft
            
            # Normalize by total power to isolate spectral shape independent of overall walking speed
            total_energy = np.sum(raw_power_spectrum) + 1e-8
            relative_energy_distribution = raw_power_spectrum / total_energy
            
            # Sum relative energy in the pathological tremor/jitter band (3.0 - 8.0 Hz) as a percentage
            jitter_energy_fraction = np.sum(relative_energy_distribution[mask_jitter])
            jitter_energy_pct = float(jitter_energy_fraction) * 100.0
            
            return jitter_energy_pct

        # Ankle Bradykinesia & Symmetry Index (L_Ankle=7, R_Ankle=8)
        auc_l_ank = _compute_phys_auc(7)
        auc_r_ank = _compute_phys_auc(8)
        ankle_bradykinesia = min(auc_l_ank, auc_r_ank)
        ankle_si = (2.0 * abs(auc_l_ank - auc_r_ank) / (auc_l_ank + auc_r_ank + 1e-8)) * 100.0

        # Spine Rigidity (Spine2=6, Spine3=9)
        auc_s2 = _compute_phys_auc(6)
        auc_s3 = _compute_phys_auc(9)
        spine_rigidity = 0.5 * (auc_s2 + auc_s3)

        # Wrist Smoothness AUC (L_Wrist=20, R_Wrist=21)
        jit_l_wri = _compute_smoothness_auc(20)
        jit_r_wri = _compute_smoothness_auc(21)
        wrist_smoothness_auc = max(jit_l_wri, jit_r_wri)

        # Hand Smoothness AUC (L_Hand=22, R_Hand=23)
        jit_l_hnd = _compute_smoothness_auc(22)
        jit_r_hnd = _compute_smoothness_auc(23)
        hand_smoothness_auc = max(jit_l_hnd, jit_r_hnd)

        return {
            "ankle_bradykinesia": ankle_bradykinesia,
            "spine_rigidity": spine_rigidity,
            "ankle_si": ankle_si,
            "wrist_smoothness_auc": wrist_smoothness_auc,
            "hand_smoothness_auc": hand_smoothness_auc
        }

    def _process_single_sequence(self, k, gt_seq, gen_seq, sev):
        """Helper method for parallelized metric computations."""
        per_joint_err = self.compute_mpjae(gt_seq, gen_seq, return_per_joint=True)
        gt_clinical = self.compute_clinical_metrics(gt_seq)
        gen_clinical = self.compute_clinical_metrics(gen_seq)

        return {
            "key": k,
            "sev": sev,
            "per_joint_err": per_joint_err,
            "gt_clinical": gt_clinical,
            "gen_clinical": gen_clinical,
        }

    def evaluate_from_memory(self, gt_data, gen_data, labels):
        """Computes metrics from pose dictionaries in memory."""
        from joblib import Parallel, delayed
        
        common_keys = [k for k in gt_data.keys() if k in gen_data.keys() and not k.endswith('_trans')]

        results = defaultdict(lambda: defaultdict(list))
        per_sequence_results = {}
        
        tasks = []
        for k in common_keys:
            sev = labels.get(k, "Unknown")
            tasks.append((k, gt_data[k], gen_data[k], sev))
            
        print(f"  [SMPLEvaluator] Processing {len(tasks)} sequences in parallel...")
        extracted_data = Parallel(n_jobs=-1)(
            delayed(self._process_single_sequence)(*t) for t in tasks
        )
        
        for res in extracted_data:
            k = res["key"]
            sev = res["sev"]
            per_joint_err = res["per_joint_err"]
            gt_c = res["gt_clinical"]
            gen_c = res["gen_clinical"]
            
            # Broad category MPJAE
            for group_name, joint_indices in self.JOINT_GROUPS.items():
                group_val = float(np.mean(per_joint_err[joint_indices]))
                results["Overall"][group_name].append(group_val)
                if sev != "Unknown":
                    results[f"Class {sev}"][group_name].append(group_val)

            # Individual joint MPJAE
            for idx, joint_name in enumerate(self.JOINT_NAMES):
                joint_val = float(per_joint_err[idx])
                results["Overall"][joint_name].append(joint_val)
                if sev != "Unknown":
                    results[f"Class {sev}"][joint_name].append(joint_val)

            # Clinical Gait Metrics (Distributions & Errors)
            clinical_metrics_map = {
                "GT_Ankle_Bradykinesia": float(gt_c["ankle_bradykinesia"]),
                "Gen_Ankle_Bradykinesia": float(gen_c["ankle_bradykinesia"]),
                "Ankle_Bradykinesia_Error": float(abs(gt_c["ankle_bradykinesia"] - gen_c["ankle_bradykinesia"])),

                "GT_Spine_Rigidity": float(gt_c["spine_rigidity"]),
                "Gen_Spine_Rigidity": float(gen_c["spine_rigidity"]),
                "Spine_Rigidity_Error": float(abs(gt_c["spine_rigidity"] - gen_c["spine_rigidity"])),

                "GT_Ankle_SI": float(gt_c["ankle_si"]),
                "Gen_Ankle_SI": float(gen_c["ankle_si"]),
                "Ankle_SI_Error": float(abs(gt_c["ankle_si"] - gen_c["ankle_si"])),

                "GT_Wrist_Smoothness_AUC": float(gt_c["wrist_smoothness_auc"]),
                "Gen_Wrist_Smoothness_AUC": float(gen_c["wrist_smoothness_auc"]),
                "Wrist_Smoothness_AUC_Error": float(abs(gt_c["wrist_smoothness_auc"] - gen_c["wrist_smoothness_auc"])),

                "GT_Hand_Smoothness_AUC": float(gt_c["hand_smoothness_auc"]),
                "Gen_Hand_Smoothness_AUC": float(gen_c["hand_smoothness_auc"]),
                "Hand_Smoothness_AUC_Error": float(abs(gt_c["hand_smoothness_auc"] - gen_c["hand_smoothness_auc"])),
            }

            for metric_name, val in clinical_metrics_map.items():
                results["Overall"][metric_name].append(val)
                if sev != "Unknown":
                    results[f"Class {sev}"][metric_name].append(val)

            per_sequence_results[k] = {
                "severity": sev,
                "overall_mpjae": float(np.mean(per_joint_err[self.HARD_MPJAE_JOINTS])),
                "per_joint_mpjae": {j_name: float(per_joint_err[i]) for i, j_name in enumerate(self.JOINT_NAMES)},
                "clinical_metrics": {
                    "gt": gt_c,
                    "gen": gen_c,
                    "error": {
                        "ankle_bradykinesia": float(abs(gt_c["ankle_bradykinesia"] - gen_c["ankle_bradykinesia"])),
                        "spine_rigidity": float(abs(gt_c["spine_rigidity"] - gen_c["spine_rigidity"])),
                        "ankle_si": float(abs(gt_c["ankle_si"] - gen_c["ankle_si"])),
                        "wrist_smoothness_auc": float(abs(gt_c["wrist_smoothness_auc"] - gen_c["wrist_smoothness_auc"])),
                        "hand_smoothness_auc": float(abs(gt_c["hand_smoothness_auc"] - gen_c["hand_smoothness_auc"])),
                    }
                }
            }

        # Build summary means dictionary
        summary_results = {}
        for cls_key, metrics_dict in results.items():
            summary_results[cls_key] = {
                metric_name: float(np.nanmean(vals))
                for metric_name, vals in metrics_dict.items()
            }
                
        cache_data = {
            "summary_means": summary_results,
            "raw_distributions": {
                cls_key: {m: [float(x) for x in vals] for m, vals in metrics_dict.items()}
                for cls_key, metrics_dict in results.items()
            },
            "per_sequence_results": per_sequence_results
        }
            
        return summary_results, cache_data

    def evaluate_and_cache(self, gt_npz_path, gen_npz_path, labels_path, cache_output_path):
        """Loads unified GT/Gen datasets from disk, computes metrics, and caches results to JSON."""
        with open(labels_path, 'r') as f:
            labels = json.load(f)["key_to_severity"]

        gt_data = np.load(gt_npz_path, allow_pickle=True)
        gen_data = np.load(gen_npz_path, allow_pickle=True)
        
        gt_data = gt_data['arr_0'].item() if 'arr_0' in gt_data.files else {k: gt_data[k] for k in gt_data.files}
        gen_data = gen_data['arr_0'].item() if 'arr_0' in gen_data.files else {k: gen_data[k] for k in gen_data.files}
        
        summary_results, cache_data = self.evaluate_from_memory(gt_data, gen_data, labels)
                
        out_path = Path(cache_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(cache_data, f, indent=4)
            
        return summary_results


if __name__ == "__main__":
    evaluator = SMPLEvaluator(fps=30)
    base_dir = Path("thesis/data/processed/JointModel-MLP-Baseline")
    
    gt_path = base_dir / "6D_SMPL" / "ground_truth_6d.npz"
    gen_path = base_dir / "6D_SMPL" / "generated_6d.npz"
    labels_path = base_dir / "h36m" / "gen_labels.json"
    output_path = base_dir / "evaluation" / "smpl_evaluation.json"

    print("--- Running SMPL Clinical Evaluation ---")
    print(f"GT Data path:  {gt_path}")
    print(f"Gen Data path: {gen_path}")
    print(f"Labels path:   {labels_path}")
    print(f"Output path:   {output_path}")

    smpl_summary = evaluator.evaluate_and_cache(
        gt_npz_path=str(gt_path),
        gen_npz_path=str(gen_path),
        labels_path=str(labels_path),
        cache_output_path=str(output_path)
    )

    print("\nSMPL Clinical Evaluation Complete!")
    for severity, metrics in smpl_summary.items():
        print(f"  -> {severity:<10}: MPJAE = {np.degrees(metrics['Overall']):.2f}° | "
              f"Ankle Brady (GT/Gen) = {metrics['GT_Ankle_Bradykinesia']:.1f} / {metrics['Gen_Ankle_Bradykinesia']:.1f} deg/s·Hz | "
              f"Spine Rigidity (GT/Gen) = {metrics['GT_Spine_Rigidity']:.1f} / {metrics['Gen_Spine_Rigidity']:.1f} deg/s·Hz | "
              f"Wrist Jitter (GT/Gen) = {metrics['GT_Wrist_Smoothness_AUC']:.1f}% / {metrics['Gen_Wrist_Smoothness_AUC']:.1f}%")