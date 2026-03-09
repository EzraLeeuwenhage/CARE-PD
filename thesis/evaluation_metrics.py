import torch
import numpy as np
from scipy.signal import find_peaks
import pickle
import json
from pathlib import Path


class SkeletonEvaluator:
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
        
        # Define a subset of major bones for constancy checking
        # edges from hips to knees and to ankles
        self.major_bones = [(0,1), (1,2), (2,3), (0,4), (4,5), (5,6)] 

    def _load_dataset(self, filepath):
        """Helper to load the dataset cleanly."""        
        print(f"Loading dataset from: {filepath}...")
        data = np.load(filepath, allow_pickle=True)
        # If npz contains a pickled dict, it's often inside a 0-d array
        if hasattr(data, 'files') and len(data.files) == 1 and data.files[0] == 'arr_0':
            data = data['arr_0'].item()
        return data

    def _get_severity_class_subsets(self, data, key_to_severity, target_class=None):
        """
        Groups flat .npz sequences by UPDRS_GAIT using the external labels registry.
        Returns a dictionary mapping {severity_class: [list of sequence tensors]}.
        """
        subsets = {}
        total_found = 0
        
        # data is a flat dictionary from .npz: {clip_id: tensor}
        for clip_id, tensor in data.items():
            # Skip if this clip isn't in our labels registry
            if clip_id not in key_to_severity:
                raise ValueError(f"Clip ID '{clip_id}' not found in severity labels registry.") 
                
            score = int(key_to_severity[clip_id])
            
            if target_class is not None and score != target_class:
                continue
                
            if score not in subsets:
                subsets[score] = []
                
            subsets[score].append(tensor)
            total_found += 1
                    
        print(f"Extracted {total_found} sequences across {len(subsets)} severity classes.")
        return subsets

    # ---------------------------------------------------------
    # PHYSICAL REALISM METRICS
    # ---------------------------------------------------------
    def _evaluate_physical_realism(self, seq_tensor):
        """
        seq_tensor: (Time, Vertices, Channels) e.g., (60, 17, 3)
        Returns dictionary of physical metrics.
        """
        if torch.is_tensor(seq_tensor):
            seq = seq_tensor.detach().cpu().numpy()
        else:
            seq = seq_tensor

        # A. Bone Length Variance (Lower is better)
        bone_variances = []
        for (j1, j2) in self.major_bones:
            # Calculate distance between joint 1 and joint 2 over all frames
            bone_lengths = np.linalg.norm(seq[:, j1, :] - seq[:, j2, :], axis=-1)
            bone_variances.append(np.var(bone_lengths))
            
        mean_bone_variance = np.mean(bone_variances)

        # B. Smoothness / Jitter / Jerk (Mean magnitudes of derivatives)
        velocity = np.diff(seq, axis=0) 
        acceleration = np.diff(velocity, axis=0)
        jerk = np.diff(acceleration, axis=0) # 3rd derivative of position
        
        jitter = np.mean(np.linalg.norm(acceleration, axis=-1))
        mean_jerk = np.mean(np.linalg.norm(jerk, axis=-1))

        # Heel Strike Detection for other measures
        ankle_dist = np.linalg.norm(seq[:, self.L_ANKLE, :] - seq[:, self.R_ANKLE, :], axis=-1)
        peaks, _ = find_peaks(ankle_dist, distance=8, prominence=0.02)
        
        heel_strike_y_range = 0.0
        mean_foot_skating_velocity = 0.0
        
        if len(peaks) > 0:
            # C. Ankle Strike Y-Range
            # Find the lowest ankle (the stance foot) at each strike
            l_ankle_y = seq[peaks, self.L_ANKLE, 1]
            r_ankle_y = seq[peaks, self.R_ANKLE, 1]
            stance_y = np.minimum(l_ankle_y, r_ankle_y)
            
            # The range (max - min) of the floor height across all steps in the sequence
            heel_strike_y_range = np.ptp(stance_y) 
            
            # TODO: velocity is maybe not as informative as actual foot skating distance over duration of the strike 

            # D. Mean Foot Skating (Horizontal velocity of stance foot at strike)
            # Calculate horizontal (X, Z) velocity and convert to meters per second
            horiz_vel = np.linalg.norm(np.diff(seq[:, :, [0, 2]], axis=0), axis=-1) * self.fps
            
            skating_velocities = []
            for p in peaks:
                if p == 0: continue # Skip if peak is exactly at frame 0 (no prior velocity available)
                
                # Identify which ankle is planted (the lower one)
                stance_idx = self.L_ANKLE if seq[p, self.L_ANKLE, 1] < seq[p, self.R_ANKLE, 1] else self.R_ANKLE
                
                # Get the horizontal velocity of that specific ankle exactly at the strike
                skating_velocities.append(horiz_vel[p-1, stance_idx])
                
            if skating_velocities:
                mean_foot_skating_velocity = np.mean(skating_velocities)

        return {
            "bone_length_variance": mean_bone_variance,
            "jitter": jitter,
            "mean_joint_jerk": mean_jerk,
            "heel_strike_y_range": heel_strike_y_range,
            "mean_foot_skating_velocity": mean_foot_skating_velocity
        }

    # ---------------------------------------------------------
    # CLINICAL GAIT FEATURES
    # ---------------------------------------------------------
    def _extract_pd_features(self, seq_tensor):
        """
        Extracts features exactly as described in CARE-PD and GaitGen papers.
        seq_tensor: (Time, Vertices, Channels).
        """
        if torch.is_tensor(seq_tensor):
            seq = seq_tensor.detach().cpu().numpy()
        else:
            seq = seq_tensor

        T = seq.shape[0]

        # --- Care-PD ---

        # Heel Strike Detection
        # "Euclidean distance between left and right ankle joints over time"
        ankle_dist = np.linalg.norm(seq[:, self.L_ANKLE, :] - seq[:, self.R_ANKLE, :], axis=-1)
        
        # "Local maxima at least 8 frames apart and prominence of 0.02"
        peaks, _ = find_peaks(ankle_dist, distance=8, prominence=0.02)
        num_steps = len(peaks)

        # If model hasn't learned to walk yet, return None
        if num_steps < 2:
            return None

        # 1. Cadence (Steps per minute)
        duration_mins = T / (self.fps * 60.0)
        cadence = num_steps / duration_mins

        # 2. Walking Speed (distance (meters?) per second)
        # "Total sacrum displacement between first and last heel strike, divided by time"
        first_strike, last_strike = peaks[0], peaks[-1]
        sacrum_displacement = np.linalg.norm(seq[last_strike, self.PELVIS, :] - seq[first_strike, self.PELVIS, :])
        time_elapsed = (last_strike - first_strike) / self.fps
        walking_speed = sacrum_displacement / time_elapsed if time_elapsed > 0 else 0

        # 3. Step Length
        # Distance along the walking axis (z is forward direction, index 2) at heel strikes

        # Step Length, Width, and Time
        step_lengths = []
        step_widths = []
        step_times = []

        for i in range(1, len(peaks)):
            prev_peak = peaks[i-1]
            curr_peak = peaks[i]
            
            # Distance along Z (walking axis) at time of strike
            z_dist = abs(seq[curr_peak, self.L_ANKLE, 2] - seq[curr_peak, self.R_ANKLE, 2])
            # Distance along X (mediolateral axis) at time of strike
            x_dist = abs(seq[curr_peak, self.L_ANKLE, 0] - seq[curr_peak, self.R_ANKLE, 0])
            # Duration between strikes
            t_dist = (curr_peak - prev_peak) / self.fps
            
            step_lengths.append(z_dist)
            step_widths.append(x_dist)
            step_times.append(t_dist)

        # 4. Estimated Margin of Stability (eMoS)
        # eXtrapolated Center of Mass (XCoM) using Pelvis. XCoM = Position + Velocity / natural_frequency
        pelvis_x = seq[:, self.PELVIS, 0]
        pelvis_y = np.mean(seq[:, self.PELVIS, 1]) # Leg length approximation
        pelvis_v_x = np.gradient(pelvis_x) * self.fps

        w0 = np.sqrt(9.81 / (pelvis_y + 1e-6)) # natural frequency w0 = sqrt(g / l)
        xcom = pelvis_x + (pelvis_v_x / w0)

        # Base of support (mediolateral, x-axis) approximated by tracking the stance ankle (the one closest to ground/lowest Y)
        stance_ankle_x = np.where(
            seq[:, self.L_ANKLE, 1] < seq[:, self.R_ANKLE, 1], 
            seq[:, self.L_ANKLE, 0], 
            seq[:, self.R_ANKLE, 0]
        )
        emos_array = np.abs(xcom - stance_ankle_x) # eMoS is distance between XCoM and base of support boundary

        # 5. Foot Lifting (Vertical range of ankle movement, y direction)
        l_ankle_lift = np.ptp(seq[:, self.L_ANKLE, 1]) # ptp calculates max - min (range)
        r_ankle_lift = np.ptp(seq[:, self.R_ANKLE, 1])
        foot_lifting = (l_ankle_lift + r_ankle_lift) / 2.0

        # --- GaitGen ---

        # Leg length normalization 
        # Approximate leg length as the mean Euclidean distance from Pelvis to Ankles over time
        l_leg_dist = np.linalg.norm(seq[:, self.PELVIS, :] - seq[:, self.L_ANKLE, :], axis=-1)
        r_leg_dist = np.linalg.norm(seq[:, self.PELVIS, :] - seq[:, self.R_ANKLE, :], axis=-1)
        leg_length = np.mean((l_leg_dist + r_leg_dist) / 2.0)

        # 6. Stooped Posture (ASMD)
        # "Vertical distance between the neck and sacrum joints at each time step, 
        # averaging these distances over all frames, and normalizing by leg length."
        vertical_dist = np.abs(seq[:, self.NECK, 1] - seq[:, self.PELVIS, 1])
        stoop_posture = np.mean(vertical_dist) / leg_length

        # 7. Arm Swing (AAMD)
        # "Euclidean distances between wrist and shoulder joints at each time step"
        l_arm_ext = np.linalg.norm(seq[:, self.L_WRIST, :] - seq[:, self.L_SHOULDER, :], axis=-1)
        r_arm_ext = np.linalg.norm(seq[:, self.R_WRIST, :] - seq[:, self.R_SHOULDER, :], axis=-1)
        
        # "finding the maximum and minimum distances over the sequence" (Range = max - min)
        l_arm_range = np.ptp(l_arm_ext)
        r_arm_range = np.ptp(r_arm_ext)
        
        # "normalizing by leg length, and selecting the minimum arm swing between the two arms"
        l_arm_norm = l_arm_range / leg_length
        r_arm_norm = r_arm_range / leg_length
        arm_swing = min(l_arm_norm, r_arm_norm)

        # 8. Joint Variances (For AVE calculation later)
        mean_pos = np.mean(seq, axis=0) 
        sq_distances = np.sum((seq - mean_pos) ** 2, axis=-1) 
        joint_variances = np.sum(sq_distances, axis=0) / (T - 1) # Shape: (17,)

        return {
            "cadence": cadence,
            "walking_speed": walking_speed,
            "step_length_mean": np.mean(step_lengths),
            "step_length_std": np.std(step_lengths),
            "step_width_mean": np.mean(step_widths),
            "step_width_std": np.std(step_widths),
            "step_time_mean": np.mean(step_times),
            "step_time_std": np.std(step_times),
            "emos_min": np.min(emos_array),
            "emos_std": np.std(emos_array),
            "foot_lifting": foot_lifting,
            "gaitgen_stoop_posture": stoop_posture,
            "gaitgen_arm_swing": arm_swing,
            "gaitgen_joint_variances": joint_variances
        }
    
    # ---------------------------------------------------------
    # DATASET DISTRIBUTION PROCESSING
    # ---------------------------------------------------------
    def process_dataset(self, filepath, labels_path="pd_gam_labels.json", output_path="dataset_distributions.pkl", specific_keys=None, severity_class=None):
        """
        Processes an .npz file of motion sequences to compute feature distributions.
        Filters by specific_keys (list of strings) OR severity_class (int/str), 
        otherwise processes all valid sequences.
        Saves the aggregated distributions and raw extracted data to a pickle file.
        """
        try:
            with open(labels_path, "r") as f:
                labels_registry = json.load(f)
            key_to_severity = labels_registry["key_to_severity"]
        except FileNotFoundError:
            print(f"Error: Labels file {labels_path} not found. Please run the metadata extraction script first.")
            return None
        
        data = self._load_dataset(filepath)
        if data is None:
            return None
        
        # Filter dataset on specific keys if provided
        if specific_keys is not None:
            data = {k: v for k, v in data.items() if k in specific_keys}
            
            if not data:
                print("No sequences found matching the provided specific_keys.")
                return None

        # filter dataset by severity class if provided
        grouped_sequences = self._get_severity_class_subsets(data, key_to_severity, target_class=severity_class)

        # Compile a flat list of all valid sequences for global physical metrics
        all_sequences = []
        for seq_list in grouped_sequences.values():
            all_sequences.extend(seq_list)
            
        if not all_sequences:
            print("No valid sequences found with UPDRS_GAIT scores and tensor data.")
            return None

        final_output = {
            "metadata": {
                "source_file": filepath,
                "total_sequences": len(all_sequences)
            },
            "global_physical_realism_raw": {},
            "per_class_pd_features_raw": {},
            "per_class_pd_features": {},
            "overall_pd_features_raw": {},
            "overall_pd_features": {}
        }
        
        print("\nComputing global physical realism metrics...")
        phys_bone_vars = []
        phys_jitters = []
        phys_jerks = []
        phys_heel_strike_y_ranges = []
        phys_foot_skating_velocities = []
        
        for seq in all_sequences:
            phys_metrics = self._evaluate_physical_realism(seq)
            phys_bone_vars.append(phys_metrics['bone_length_variance'])
            phys_jitters.append(phys_metrics['jitter'])
            phys_jerks.append(phys_metrics['mean_joint_jerk'])
            phys_heel_strike_y_ranges.append(phys_metrics['heel_strike_y_range'])
            phys_foot_skating_velocities.append(phys_metrics['mean_foot_skating_velocity'])

            
        final_output["global_physical_realism_raw"] = {
            "bone_length_variances": phys_bone_vars,
            "jitters": phys_jitters,
            "mean_joint_jerks": phys_jerks,
            "heel_strike_y_ranges": phys_heel_strike_y_ranges,
            "foot_skating_velocities": phys_foot_skating_velocities
        }

        print("\nComputing PD features per severity class and globally...")
        feature_keys = [
            "cadence", "walking_speed", "step_length_mean", "step_length_std", 
            "step_width_mean", "step_width_std", "step_time_mean", "step_time_std",
            "emos_min", "emos_std", "foot_lifting", "gaitgen_stoop_posture", 
            "gaitgen_arm_swing", "gaitgen_joint_variances"
        ]
        overall_features = {k: [] for k in feature_keys}
        total_valid_seqs = 0

        for severity, seq_list in grouped_sequences.items():
            print(f"  Processing Class {severity} ({len(seq_list)} sequences)...")
            class_features = {k: [] for k in feature_keys}
            
            valid_seqs = 0
            for seq in seq_list:
                features = self._extract_pd_features(seq)
                if features is not None:
                    valid_seqs += 1
                    total_valid_seqs += 1
                    for k, v in features.items():
                        class_features[k].append(v)
                        overall_features[k].append(v)
            
            if valid_seqs == 0:
                continue
            
            final_output["per_class_pd_features_raw"][severity] = class_features

            # Compute mean and std for class features, handle joint variances separately since they are arrays
            class_stats = {"count": valid_seqs}
            for feat_name, val_list in class_features.items():
                if feat_name == "gaitgen_joint_variances":
                    class_stats["gaitgen_joint_variances"] = np.mean(np.vstack(val_list), axis=0)
                else:
                    class_stats[f"{feat_name}_mean"] = np.mean(val_list)
                    class_stats[f"{feat_name}_std"] = np.std(val_list)
                    
            final_output["per_class_pd_features"][severity] = class_stats

        print("\nComputing overall dataset PD features...")
        final_output["overall_pd_features_raw"] = overall_features

        overall_stats = {"count": total_valid_seqs}
        
        if total_valid_seqs > 0:
            for feat_name, val_list in overall_features.items():
                if feat_name == "gaitgen_joint_variances":
                    overall_stats["gaitgen_joint_variances"] = np.mean(np.vstack(val_list), axis=0)
                else:
                    overall_stats[f"{feat_name}_mean"] = np.mean(val_list)
                    overall_stats[f"{feat_name}_std"] = np.std(val_list)
                    
        final_output["overall_pd_features"] = overall_stats

        # Save to disk using pickle (since we have NumPy arrays inside)
        with open(output_path, 'wb') as f:
            pickle.dump(final_output, f)
            
        print(f"\nSuccessfully saved structured distributions to: {output_path}")
        return final_output
    

if __name__ == "__main__":
    evaluator = SkeletonEvaluator(fps=30)
    SCRIPT_DIR = Path(__file__).parent.resolve()
    real_data_path = SCRIPT_DIR / ".." / "assets" / "datasets" / "h36m" / "PD-GaM" / "h36m_3d_world_floorXZZplus_30f_or_longer.npz"
    labels_file = SCRIPT_DIR / "pd_gam_labels.json"

    real_data_path = real_data_path.resolve()
    labels_file = labels_file.resolve()

    real_data_path_str = str(real_data_path)
    labels_file_str = str(labels_file)

    print(f"Data path resolved to: {real_data_path_str}")
    print(f"Labels path resolved to: {labels_file_str}")

    # Process Specific Patient Keys
    with open(labels_file, "r") as f:
        registry = json.load(f)
        
    # Get all sequence IDs that belong to patient 007
    patient_007_keys = [k for k in registry["key_to_severity"].keys() if k.startswith("007__")]
    
    print(f"Found {len(patient_007_keys)} keys for Patient 007.")

    patient_stats = evaluator.process_dataset(
        filepath=real_data_path,
        labels_path=labels_file,
        output_path="patient_007_distribution.pkl",
        specific_keys=patient_007_keys
    )