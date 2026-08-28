import torch
import numpy as np
from scipy.signal import find_peaks
import pickle
import json
from pathlib import Path


class H36MEvaluator:
    def __init__(self, fps=30, min_z_travel=0.0):
        self.fps = fps
        self.min_z_travel = min_z_travel
        # Human3.6M standard joint indices
        self.PELVIS = 0
        self.R_ANKLE = 3
        self.L_ANKLE = 6
        self.NECK = 8
        self.L_WRIST = 13
        self.R_WRIST = 16
        self.L_SHOULDER = 11
        self.R_SHOULDER = 14
        
        # Major bones for bone-length checks (pelvis - hips - knees - ankles)
        self.major_bones = [(0,1), (1,2), (2,3), (0,4), (4,5), (5,6)]

        self.metric_keys = [
            "sequence_length", "mean_bone_length_variance", "floating", 
            "mean_stance_displacement", "mean_step_length", "mean_step_asymmetry", 
            "mean_walking_speed", "max_ankle_clearance", 
            "mean_emos", "variance_emos"
        ]
        self.nan_metrics = self.metric_keys.copy()
        self.nan_metrics.remove("mean_bone_length_variance")
        self.nan_metrics.remove("sequence_length")

        # heel strike detection params
        self.MIN_FRAMES_BETWEEN_STEPS = 1

    def _load_dataset(self, filepath):
        """Loads flat sequence dictionary from .npz or .pkl"""        
        data = np.load(filepath, allow_pickle=True)
        if hasattr(data, 'files') and len(data.files) == 1 and data.files[0] == 'arr_0':
            data = data['arr_0'].item()
        else:
            # Reconstruct dictionary from npz keys
            data = {k: np.array(data[k]) for k in data.files}
        return data

    def _get_severity_class_subsets(self, data, key_to_severity):
        """Groups sequences by severity using the labels registry."""
        subsets = {}
        
        for clip_id, tensor in data.items():
            # Handle generated data naming vs true data naming
            base_key = clip_id.split('_down')[0] if '_down' in clip_id else clip_id
            base_key = base_key.replace('generated_walk_', '')
            
            if base_key in key_to_severity:
                score = int(key_to_severity[base_key])
            else:
                print(f"Warning: Could not resolve severity for clip '{clip_id}' (base key: '{base_key}'). Skipping.")
                continue # Skip if label has no score
                
            if score not in subsets:
                subsets[score] = []
                
            # Store both the clip_id and the tensor as a tuple
            subsets[score].append((clip_id, tensor))
            
        return subsets
    
    def detect_heel_stikes(self, seq):
        """Detects heel strikes in motion sequence.
        
        Uses base theory of Zeni et al. 2008 (M1).
        Use velocity thresholding by Bonci et al. 2022 (M7) for frame 0 strike detection.
        Returns: list of tuples (frame_idx, joint_idx) of detected strikes.
        """
        T = seq.shape[0]

        # Calculate overall walking speed to set an adaptive velocity threshold (M7)
        total_pelvis_disp = np.linalg.norm(seq[-1, self.PELVIS, :] - seq[0, self.PELVIS, :])
        avg_walking_speed = total_pelvis_disp / (T / self.fps) if T > 0 else 0
        # Threshold: 0.5 * walking speed, with a hard floor of 0.1 m/s for freezing patients
        v_thresh = max(0.5 * avg_walking_speed, 0.1)

        # Calculate 3D velocities of both ankles
        l_ankle_v = np.linalg.norm(np.diff(seq[:, self.L_ANKLE, :], axis=0), axis=-1) * self.fps
        r_ankle_v = np.linalg.norm(np.diff(seq[:, self.R_ANKLE, :], axis=0), axis=-1) * self.fps
        # Pad by repeating the last value to maintain length T
        l_ankle_v = np.append(l_ankle_v, l_ankle_v[-1])
        r_ankle_v = np.append(r_ankle_v, r_ankle_v[-1])

        # Zeni Method (M1): Maximize Anterior-Posterior (Z-axis) distance from pelvis
        l_ap_dist = seq[:, self.L_ANKLE, 2] - seq[:, self.PELVIS, 2]
        r_ap_dist = seq[:, self.R_ANKLE, 2] - seq[:, self.PELVIS, 2]

        # Use a very low prominence to catch tiny shuffling steps
        l_candidates, _ = find_peaks(l_ap_dist, prominence=0.01)
        r_candidates, _ = find_peaks(r_ap_dist, prominence=0.01)

        l_strikes = list(l_candidates)
        r_strikes = list(r_candidates)

        # Corrected Frame 0 initially planted foot check
        # Only register the most forward foot if both happen to be planted
        l_planted = l_ankle_v[0] < v_thresh and seq[0, self.L_ANKLE, 1] < 0.05
        r_planted = r_ankle_v[0] < v_thresh and seq[0, self.R_ANKLE, 1] < 0.05
        
        if l_planted and r_planted:
            if l_ap_dist[0] > r_ap_dist[0]:
                l_strikes.insert(0, 0)
            else:
                r_strikes.insert(0, 0)
        elif l_planted:
            l_strikes.insert(0, 0)
        elif r_planted:
            r_strikes.insert(0, 0)

        # Combine and sort chronologically, preserving the joint identifier
        raw_strikes = sorted([(f, self.L_ANKLE) for f in l_strikes] + 
                             [(f, self.R_ANKLE) for f in r_strikes], key=lambda x: x[0])

        if not raw_strikes:
            return []

        # Score candidate heel strikes on methods from Zeni et al. 2008 or Bonci et al. 2022
        def score_candidate(frame, leg):
            """
            Scores a candidate heel strike. Higher is better.
            Currently purely spatial (M1): favors the furthest forward foot.
            Future: can subtract velocity or Y-coordinate to penalize mid-air/moving feet.
            """
            ap_dist = l_ap_dist[frame] if leg == self.L_ANKLE else r_ap_dist[frame]
            return ap_dist

        final_strikes = []
        
        for frame, leg in raw_strikes:
            if not final_strikes:
                final_strikes.append((frame, leg))
                continue
                
            prev_frame, prev_leg = final_strikes[-1]
            
            if leg == prev_leg:
                # Still same leg, keep updating strike if new candidate is better
                if score_candidate(frame, leg) > score_candidate(prev_frame, prev_leg):
                    final_strikes[-1] = (frame, leg)
            else:
                # Alternating to other leg, check if enough frames have passed since the last strike
                enough_frames_between_steps = frame - prev_frame >= self.MIN_FRAMES_BETWEEN_STEPS
                if enough_frames_between_steps:
                    final_strikes.append((frame, leg))
                else:
                    # Not enough frames between strikes, assume noise -> evaluate which candidate is better
                    new_strike_is_better = score_candidate(frame, leg) > score_candidate(prev_frame, prev_leg)
                    if new_strike_is_better:
                        # new candidate better, replace the last strike with the new strike
                        final_strikes.pop()
                        
                        if not final_strikes:
                            # just add strike if list is empty
                            final_strikes.append((frame, leg))
                        else:
                            # if not, new strike and second-to-last strike are of same leg
                            # compare these strikes and keep the better one 
                            old_frame, old_leg = final_strikes[-1]
                            if score_candidate(frame, leg) > score_candidate(old_frame, old_leg):
                                final_strikes[-1] = (frame, leg)

        return final_strikes

    def _extract_sequence_metrics(self, seq_tensor, clip_id="Unknown"):
        """
        Extracts metrics + sequence length for a single sequence.
        Assumes seq_tensor shape is (T, 17, 3) and values are in meters.
        """
        if torch.is_tensor(seq_tensor):
            seq = seq_tensor.detach().cpu().numpy()
        else:
            seq = seq_tensor

        T = seq.shape[0]
        metrics = {"sequence_length": T}

        # Return NaNs for sequences with no forward movement (in severity class 3)
        z_travel = abs(seq[-1, self.PELVIS, 2] - seq[0, self.PELVIS, 2])
        if z_travel < self.min_z_travel:
            for m in self.nan_metrics: metrics[m] = np.nan
            metrics["heel_strikes_info"] = []
            return metrics

        # ---------------------------------------------------------
        # PHYSICAL REALISM
        # ---------------------------------------------------------
        bone_variances = []
        for (j1, j2) in self.major_bones:
            bone_lengths = np.linalg.norm(seq[:, j1, :] - seq[:, j2, :], axis=-1)
            bone_variances.append(np.var(bone_lengths))
        metrics["mean_bone_length_variance"] = np.mean(bone_variances)

        # Heel Strike Detection
        peaks_info = self.detect_heel_stikes(seq)
        
        if len(peaks_info) < 2:
            print(f"  Warning: Sequence '{clip_id}' too short or no heel strikes detected (T={T}, peaks={len(peaks_info)}). Returning NaN metrics.")
            for m in self.nan_metrics: metrics[m] = np.nan
            metrics["heel_strikes_info"] = []
            return metrics

        # Floating
        floats = []
        hs_info = []
        
        for p, stance_idx in peaks_info:
            floats.append(seq[p, stance_idx, 1])
            hs_info.append({
                "frame": int(p), 
                "joint_idx": int(stance_idx), 
                "coord": seq[p, stance_idx, :].tolist()
            })
                
        metrics["floating"] = np.mean(floats)
        metrics["heel_strikes_info"] = hs_info

        # ---------------------------------------------------------
        # PD FEATURES
        # ---------------------------------------------------------

        # Step Length & Vertical Foot Lifting
        step_lengths = []
        ankle_clearances = []
        stance_displacements = []
        
        for i in range(1, len(peaks_info)):
            start, prev_stance_idx = peaks_info[i-1]
            end, curr_stance_idx = peaks_info[i]
                
            # Stance foot (/ankle) displacement
            anchor_pos = seq[start, prev_stance_idx, :]
            end_pos = seq[end, prev_stance_idx, :]
            
            # Calculate 3D Euclidean distance between the stance foot's position at the start and end of the step
            displacement = np.linalg.norm(end_pos - anchor_pos)
            stance_displacements.append(displacement)
            
            # Step Length, horizontal distance between joints at moment of strike
            step_length = np.linalg.norm(seq[end, self.L_ANKLE, [0,2]] - seq[end, self.R_ANKLE, [0,2]])
            step_lengths.append(step_length)
            
            # Absolute Ankle Clearance, swinging foot strikes ground at the end of interval
            clearance = np.max(seq[start:end, curr_stance_idx, 1])
            ankle_clearances.append(clearance)

        metrics["mean_stance_displacement"] = np.mean(stance_displacements) if stance_displacements else np.nan
        metrics["mean_step_length"] = np.mean(step_lengths) if step_lengths else np.nan
        
        # Step Asymmetry (using Robinson Symmetry Index on consecutive alternating steps)
        if len(step_lengths) > 1:
            sl_arr = np.array(step_lengths)
            diffs = np.abs(np.diff(sl_arr))
            sums = sl_arr[:-1] + sl_arr[1:] + 1e-7
            asymmetries = (2.0 * diffs / sums) * 100.0
            metrics["mean_step_asymmetry"] = np.mean(asymmetries)
        else:
            metrics["mean_step_asymmetry"] = np.nan
            
        metrics["max_ankle_clearance"] = np.max(ankle_clearances) if ankle_clearances else np.nan

        # Walking Speed (m/s)
        first_strike, _ = peaks_info[0]
        last_strike, _ = peaks_info[-1]
        pelvis_displacement = np.linalg.norm(seq[last_strike, self.PELVIS, :] - seq[first_strike, self.PELVIS, :])
        time_elapsed = (last_strike - first_strike) / self.fps
        metrics["mean_walking_speed"] = pelvis_displacement / time_elapsed if time_elapsed > 0 else 0.0

        # Estimated Margin of Stability (eMoS) -> Medio-Lateral stability
        pelvis_x = seq[:, self.PELVIS, 0]
        pelvis_y = np.mean(seq[:, self.PELVIS, 1]) 
        pelvis_v_x = np.gradient(pelvis_x) * self.fps

        w0 = np.sqrt(9.81 / (pelvis_y + 1e-6))
        xcom = pelvis_x + (pelvis_v_x / w0)

        emos_at_strikes = []

        # Extract eMoS specifically at Heel Strikes relative to the LEADING foot
        if len(peaks_info) > 1:
            for i in range(1, len(peaks_info)):
                start, _ = peaks_info[i-1]
                end, leading_idx = peaks_info[i]
                trailing_idx = self.R_ANKLE if leading_idx == self.L_ANKLE else self.L_ANKLE
                
                stance_x = seq[end, leading_idx, 0]
                swing_x = seq[end, trailing_idx, 0]
                
                # Direction of instability (from trailing to leading foot) ensures lateral outward measurement
                direction = np.sign(stance_x - swing_x)
                
                # Calculate MoS at the exact frame of heel strike
                emos = (stance_x - xcom[end]) * direction
                emos_at_strikes.append(emos)

            metrics["mean_emos"] = np.mean(emos_at_strikes) if emos_at_strikes else np.nan
            metrics["variance_emos"] = np.var(emos_at_strikes) if emos_at_strikes else np.nan

        return metrics

    def evaluate_from_memory(self, data_dict, key_to_severity):
        """Extracts metrics from a dictionary of sequences in memory."""
        from joblib import Parallel, delayed

        grouped_sequences = self._get_severity_class_subsets(data_dict, key_to_severity)

        distributions = {"overall": {k: [] for k in self.metric_keys}}
        for severity in grouped_sequences.keys():
            distributions[severity] = {k: [] for k in self.metric_keys}

        heel_strikes_registry = {}

        tasks = []
        for severity, seq_list in grouped_sequences.items():
            for clip_id, seq in seq_list:
                tasks.append((severity, clip_id, seq))
                
        print(f"  [H36MEvaluator] Processing {len(tasks)} sequences in parallel...")
        results = Parallel(n_jobs=-1)(
            delayed(self._extract_sequence_metrics)(seq, clip_id) for _, clip_id, seq in tasks
        )

        for (severity, clip_id, _), metrics in zip(tasks, results):
            heel_strikes_registry[clip_id] = metrics.pop("heel_strikes_info", [])
            
            if not np.isnan(metrics["mean_walking_speed"]):
                for k in self.metric_keys:
                    distributions[severity][k].append(metrics[k])
                    distributions["overall"][k].append(metrics[k])
                    
        for group in distributions.keys():
            for k in self.metric_keys:
                distributions[group][k] = np.array(distributions[group][k])
                
        return distributions, heel_strikes_registry

    def process_dataset(self, filepath, labels_path):
        """Extracts metrics for all sequences from a disk file."""
        print(f"\nProcessing dataset: {filepath}")
        with open(labels_path, "r") as f:
            key_to_severity = json.load(f)["key_to_severity"]
            
        data = self._load_dataset(filepath)
        return self.evaluate_from_memory(data, key_to_severity)

    def evaluate_and_cache(self, npz_path, labels_path, cache_output_path, synthetic=False):
        """Process and save the raw distributions and heel strike markers to disk."""
        distributions, heel_strikes_registry = self.process_dataset(npz_path, labels_path)
        
        out_path = Path(cache_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, 'wb') as f:
            pickle.dump(distributions, f)

        if synthetic:
            hs_out_path = Path(npz_path).parent / "gen_heel_strikes.json"
        else:
            hs_out_path = Path(npz_path).parent / "gt_heel_strikes.json"

        with open(hs_out_path, 'w') as f:
            json.dump(heel_strikes_registry, f, indent=4)
            
        print(f"[Evaluator] Saved evaluation distributions to: {cache_output_path}")
        print(f"[Evaluator] Saved heel strike detection data to: {hs_out_path}")
        return distributions


if __name__ == "__main__":
    evaluator = H36MEvaluator(fps=30)
    model_folder = "JointModel-MLP-Baseline"
    base_dir = Path(f"thesis/data/processed/{model_folder}")

    data_path = base_dir / "h36m" / "ground_truth_3d_world.npz"
    labels_path = base_dir / "h36m" / "gt_labels.json"
    output_path = base_dir / "evaluation" / "gt_h36m_distributions.pkl"

    print(f"Data path:   {data_path}")
    print(f"Labels path: {labels_path}")
    print(f"Output path: {output_path}")

    ground_truth_distributions = evaluator.evaluate_and_cache(
        npz_path=str(data_path),
        labels_path=str(labels_path),
        cache_output_path=str(output_path),
        synthetic=False
    )

    print("\n Ground Truth Extraction Complete!")
    for severity, metrics in ground_truth_distributions.items():
        count = len(metrics['sequence_length'])
        print(f"  -> Class '{severity}': {count} valid sequences processed.")

    data_path = base_dir / "h36m" / "generated_3d_world.npz"
    labels_path = base_dir / "h36m" / "gen_labels.json"
    output_path = base_dir / "evaluation" / "gen_h36m_distributions.pkl"

    print(f"Data path:   {data_path}")
    print(f"Labels path: {labels_path}")
    print(f"Output path: {output_path}")

    generated_distributions = evaluator.evaluate_and_cache(
        npz_path=str(data_path),
        labels_path=str(labels_path),
        cache_output_path=str(output_path),
        synthetic=True
    )

    print("\n Generated Data Extraction Complete!")
    for severity, metrics in generated_distributions.items():
        count = len(metrics['sequence_length'])
        print(f"  -> Class '{severity}': {count} valid sequences processed.")