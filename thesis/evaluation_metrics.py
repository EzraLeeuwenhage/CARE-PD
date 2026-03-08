import torch
import numpy as np
from scipy.signal import find_peaks

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

    # ---------------------------------------------------------
    # PHYSICAL REALISM METRICS
    # ---------------------------------------------------------
    def evaluate_physical_realism(self, seq_tensor):
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

        # B. Smoothness / Jitter (Mean magnitude of acceleration)
        # 2nd derivative of positions over time
        velocity = np.diff(seq, axis=0)
        acceleration = np.diff(velocity, axis=0)
        jitter = np.mean(np.linalg.norm(acceleration, axis=-1))

        return {
            "bone_length_variance": mean_bone_variance,
            "jitter": jitter
        }

    # ---------------------------------------------------------
    # CLINICAL GAIT FEATURES
    # ---------------------------------------------------------
    def extract_pd_features(self, seq_tensor):
        """
        Extracts features exactly as described in CARE-PD and GaitGen papers.
        seq_tensor: (Time, Vertices, Channels)
        """
        if torch.is_tensor(seq_tensor):
            seq = seq_tensor.detach().cpu().numpy()
        else:
            seq = seq_tensor

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
        duration_mins = seq.shape[0] / (self.fps * 60.0)
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
        # TODO: for gaitgen AVE, ASMD and AAMD, implement class distribution versions (as in paper)

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
            "gaitgen_arm_swing": arm_swing
        }
    
    # ---------------------------------------------------------
    # GAITGEN AVE METRIC
    # ---------------------------------------------------------
    def compute_joint_variances(self, seq_tensor):
        """
        Equation 5: Computes the variance of local joint positions for a single sequence.
        seq_tensor: (Time, Vertices, Channels)
        Returns: Array of shape (Vertices,) containing the variance (\sigma) for each joint.
        """
        if torch.is_tensor(seq_tensor):
            seq = seq_tensor.detach().cpu().numpy()
        else:
            seq = seq_tensor
            
        T = seq.shape[0]
        
        # \bar{P}[j]: Mean position for each joint across all T frames
        mean_pos = np.mean(seq, axis=0) 
        
        # || P_t[j] - \bar{P}[j] ||^2 : Squared Euclidean distance from the mean
        sq_distances = np.sum((seq - mean_pos) ** 2, axis=-1) 
        
        # Sum over time and divide by (T - 1)
        joint_variances = np.sum(sq_distances, axis=0) / (T - 1)
        
        return joint_variances

    def calculate_ave(self, real_seq, gen_seq):
        """
        Equation 6: Calculates the overall Average Variance Error (AVE) 
        between a ground-truth sequence and a generated sequence.
        """
        # \sigma[j] (Ground truth variance)
        real_vars = self.compute_joint_variances(real_seq)
        
        # \hat{\sigma}[j] (Generated variance)
        gen_vars = self.compute_joint_variances(gen_seq)
        
        # || \sigma[j] - \hat{\sigma}[j] ||^2 : Squared difference per joint
        per_joint_ave = (real_vars - gen_vars) ** 2
        
        # Overall AVE: Average the errors across all joints
        overall_ave = np.mean(per_joint_ave)
        
        return overall_ave
    

if __name__ == "__main__":
    evaluator = SkeletonEvaluator(fps=30)
    real_data_path = "../assets/datasets/h36m/PD-GaM/h36m_3d_world_floorXZZplus_30f_or_longer.npz"
    
    # The specific sequence key you want to test
    sequence_key = "007__007-13-000661_wid00_3"
    
    print(f"Loading dataset from: {real_data_path}")
    try:
        data_dict = np.load(real_data_path, allow_pickle=True)
        
        if sequence_key not in data_dict:
            print(f"Error: Sequence '{sequence_key}' not found in the dataset.")
            print(f"Here are a few valid keys you can try instead: {data_dict.files[:5]}")
        else:
            real_seq = data_dict[sequence_key]
            print(f"Successfully loaded sequence '{sequence_key}'.")
            print(f"Sequence shape: {real_seq.shape} (Expected: Time, Vertices, Channels)")
            
            # Evaluate Physical Realism
            physical_metrics = evaluator.evaluate_physical_realism(real_seq)
            print("\n--- Physical Realism Metrics (Ground Truth) ---")
            for k, v in physical_metrics.items():
                print(f"  {k}: {v:.6f}")
                
            # Evaluate Clinical Gait Features
            clinical_features = evaluator.extract_pd_features(real_seq)
            print("\n--- Clinical Gait Features (Ground Truth) ---")
            if clinical_features is None:
                print("  Error: Could not extract features. Sequence might be too short or lack distinct heel strikes.")
            else:
                for k, v in clinical_features.items():
                    print(f"  {k}: {v:.6f}")
                    
    except FileNotFoundError:
        print(f"Error: Could not find the file at {real_data_path}. Make sure you are running the script from the CARE-PD root directory.")