import torch
import numpy as np
from scipy.signal import find_peaks
import pickle
import json
from pathlib import Path


class H36MEvaluator:
    def __init__(self, fps=30):
        self.fps = fps
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
            # If it's a generated sequence, try to extract the original patient prefix
            base_key = base_key.replace('generated_walk_', '')
            
            # Find matching severity
            score = None
            for reg_key, reg_score in key_to_severity.items():
                if reg_key in base_key or base_key in reg_key:
                    score = int(reg_score)
                    break
                    
            if score is None:
                continue # Skip if label cannot be resolved
                
            if score not in subsets:
                subsets[score] = []
                
            subsets[score].append(tensor)
            
        return subsets

    def _extract_sequence_metrics(self, seq_tensor):
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

        # ---------------------------------------------------------
        # PHYSICAL REALISM
        # ---------------------------------------------------------
        # Bone Length Variance
        bone_variances = []
        for (j1, j2) in self.major_bones:
            bone_lengths = np.linalg.norm(seq[:, j1, :] - seq[:, j2, :], axis=-1)
            bone_variances.append(np.var(bone_lengths))
        metrics["mean_bone_length_variance"] = np.mean(bone_variances)

        # Mean Jerk (3rd derivative of position with respect to time)
        velocity = np.diff(seq, axis=0) * self.fps
        acceleration = np.diff(velocity, axis=0) * self.fps
        jerk = np.diff(acceleration, axis=0) * self.fps
        metrics["mean_jerk"] = np.mean(np.linalg.norm(jerk, axis=-1))

        # Heel Strike Detection (to calculate other metrics)
        # TODO: validate visually for high severity score sequences
        ankle_dist = np.linalg.norm(seq[:, self.L_ANKLE, :] - seq[:, self.R_ANKLE, :], axis=-1)

        # TODO: nu 24/35 class 3 sequences valid, 11/35 get filtered out due to no heel strikes
        peaks, _ = find_peaks(ankle_dist, distance=8, prominence=0.005)
        
        # Return NaNs if no proper walking pattern in motion sequence
        if len(peaks) < 2:
            print(f"  Warning: Sequence too short or no heel strikes detected (T={T}, peaks={len(peaks)}). Returning NaN metrics.")
            nan_metrics = ["floating", "foot_skating", "mean_step_length", "variance_step_length", 
                           "mean_walking_speed", "mean_vertical_foot_lifting", "mean_emos", "variance_emos"]
            for m in nan_metrics: metrics[m] = np.nan
            return metrics

        # Floating & Foot Skating (Evaluated at the exact moment of heel strikes)
        floats = []
        skates = []
        horiz_vel = np.linalg.norm(np.diff(seq[:, :, [0, 2]], axis=0), axis=-1) * self.fps
        
        for p in peaks:
            # Identify planted foot (lowest Y coordinate)
            stance_idx = self.L_ANKLE if seq[p, self.L_ANKLE, 1] < seq[p, self.R_ANKLE, 1] else self.R_ANKLE
            # Floating: Y position of the planted foot
            floats.append(seq[p, stance_idx, 1])
            # Skating: Horizontal velocity of the planted foot
            if p > 0:
                skates.append(horiz_vel[p-1, stance_idx])
                
        metrics["floating"] = np.mean(floats)
        metrics["foot_skating"] = np.mean(skates) if skates else np.nan

        # ---------------------------------------------------------
        # PD FEATURES
        # ---------------------------------------------------------
        # Step Length & Vertical Foot Lifting
        step_lengths = []
        foot_lifts = []
        
        for i in range(1, len(peaks)):
            start, end = peaks[i-1], peaks[i]
            
            # Step Length: Horizontal distance between ankles at strike
            sl = np.linalg.norm(seq[end, self.L_ANKLE, [0,2]] - seq[end, self.R_ANKLE, [0,2]])
            step_lengths.append(sl)
            
            # Foot lifting: swinging foot max Y minus stance foot mean Y
            # The swinging foot is the one that traveled further in Z during this step
            l_travel = abs(seq[end, self.L_ANKLE, 2] - seq[start, self.L_ANKLE, 2])
            r_travel = abs(seq[end, self.R_ANKLE, 2] - seq[start, self.R_ANKLE, 2])
            
            swing_idx = self.L_ANKLE if l_travel > r_travel else self.R_ANKLE
            stance_idx = self.R_ANKLE if swing_idx == self.L_ANKLE else self.L_ANKLE
            
            max_swing_y = np.max(seq[start:end, swing_idx, 1])
            mean_stance_y = np.mean(seq[start:end, stance_idx, 1])
            foot_lifts.append(max_swing_y - mean_stance_y)

        metrics["mean_step_length"] = np.mean(step_lengths)
        metrics["variance_step_length"] = np.var(step_lengths)
        metrics["mean_vertical_foot_lifting"] = np.mean(foot_lifts)

        # Walking Speed (m/s)
        first_strike, last_strike = peaks[0], peaks[-1]
        pelvis_displacement = np.linalg.norm(seq[last_strike, self.PELVIS, :] - seq[first_strike, self.PELVIS, :])
        time_elapsed = (last_strike - first_strike) / self.fps
        metrics["mean_walking_speed"] = pelvis_displacement / time_elapsed if time_elapsed > 0 else 0.0

        # Estimated Margin of Stability (eMoS)
        pelvis_x = seq[:, self.PELVIS, 0]
        pelvis_y = np.mean(seq[:, self.PELVIS, 1]) # approximate leg length
        pelvis_v_x = np.gradient(pelvis_x) * self.fps

        w0 = np.sqrt(9.81 / (pelvis_y + 1e-6))
        xcom = pelvis_x + (pelvis_v_x / w0)

        emos_at_strikes = []

        # Extract eMoS specifically at Heel Strikes relative to the LEADING foot
        if len(peaks) > 1:
            for i in range(1, len(peaks)):
                start, end = peaks[i-1], peaks[i]
                
                # The foot that traveled further during the step is the leading foot
                l_travel = abs(seq[end, self.L_ANKLE, 2] - seq[start, self.L_ANKLE, 2])
                r_travel = abs(seq[end, self.R_ANKLE, 2] - seq[start, self.R_ANKLE, 2])
                
                leading_idx = self.L_ANKLE if l_travel > r_travel else self.R_ANKLE
                trailing_idx = self.R_ANKLE if leading_idx == self.L_ANKLE else self.L_ANKLE
                
                stance_x = seq[end, leading_idx, 0]
                swing_x = seq[end, trailing_idx, 0]
                
                # Direction of instability (from trailing to leading foot)
                direction = np.sign(stance_x - swing_x)
                
                # Calculate MoS at the exact frame of heel strike
                emos = (stance_x - xcom[end]) * direction
                emos_at_strikes.append(emos)

            metrics["mean_emos"] = np.mean(emos_at_strikes)
            metrics["variance_emos"] = np.var(emos_at_strikes)

        return metrics

    def process_dataset(self, filepath, labels_path):
        """Extracts metrics for all sequences."""
        print(f"\nProcessing dataset: {filepath}")
        with open(labels_path, "r") as f:
            key_to_severity = json.load(f)["key_to_severity"]
            
        data = self._load_dataset(filepath)
        grouped_sequences = self._get_severity_class_subsets(data, key_to_severity)

        metric_keys = [
            "sequence_length", "mean_bone_length_variance", "floating", "foot_skating",
            "mean_step_length", "variance_step_length", "mean_walking_speed", 
            "mean_vertical_foot_lifting", "mean_emos", "variance_emos", "mean_jerk"
        ]
        
        distributions = {"overall": {k: [] for k in metric_keys}}
        for severity in grouped_sequences.keys():
            distributions[severity] = {k: [] for k in metric_keys}

        # Process all sequences
        for severity, seq_list in grouped_sequences.items():
            for seq in seq_list:
                metrics = self._extract_sequence_metrics(seq)
                
                # Only append if the sequence actually took steps (is not NaN)
                if not np.isnan(metrics["mean_walking_speed"]):
                    for k in metric_keys:
                        distributions[severity][k].append(metrics[k])
                        distributions["overall"][k].append(metrics[k])
                        
        # Convert lists to np arrays
        for group in distributions.keys():
            for k in metric_keys:
                distributions[group][k] = np.array(distributions[group][k])
                
        return distributions

    def evaluate_and_cache(self, npz_path, labels_path, cache_output_path):
        """Process and save the raw distributions to a pickle file."""
        distributions = self.process_dataset(npz_path, labels_path)
        
        out_path = Path(cache_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, 'wb') as f:
            pickle.dump(distributions, f)
            
        print(f"[Evaluator] Saved evaluation distributions to: {cache_output_path}")
        return distributions
    

if __name__ == "__main__":
    evaluator = H36MEvaluator(fps=30)
    data_path = Path("thesis/data/raw/PD-GaM/h36m/h36m_3d_world_floorXZZplus_30f_or_longer.npz")
    labels_path = Path("thesis/data/metadata/pd_gam_labels.json") 
    output_path = Path("thesis/data/processed/evaluation/gt_h36m_distributions.pkl")

    print(f"--- Generating Ground Truth H36M Evaluation Metrics ---")
    print(f"Data path:   {data_path}")
    print(f"Labels path: {labels_path}")
    print(f"Output path: {output_path}")

    ground_truth_distributions = evaluator.evaluate_and_cache(
        npz_path=str(data_path),
        labels_path=str(labels_path),
        cache_output_path=str(output_path)
    )

    print("\n Ground Truth Extraction Complete!")
    for severity, metrics in ground_truth_distributions.items():
        count = len(metrics['sequence_length'])
        print(f"  -> Class '{severity}': {count} valid sequences processed.")