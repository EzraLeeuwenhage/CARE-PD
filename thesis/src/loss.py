import torch
import torch.nn as nn

# Standard SMPL 24-joint index mapping
SMPL_24_JOINTS = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck",
    "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"
]

# Hardcoded kinematic importance weights based on gait impact and MPJAE variance
JOINT_MPJAE_WEIGHTS = {
    # Knees (highest error)
    "L_Knee": 4.5, "R_Knee": 4.5,

    # Hips and Ankles (large error)
    "L_Hip": 3.5, "R_Hip": 3.5,
    "L_Ankle": 3.0, "R_Ankle": 3.0,

    # Arm swing, root, and feet (still important walking joints)
    "Pelvis": 2.0, "Spine1": 1.8,
    "L_Foot": 1.8, "R_Foot": 1.8,
    "L_Elbow": 1.5, "R_Elbow": 1.5,
    "L_Shoulder": 1.2, "R_Shoulder": 1.2,

    # Low-movement torso, neck, head, and hands
    "Spine2": 0.5, "Spine3": 0.5,
    "Neck": 0.4, "Head": 0.4,
    "L_Collar": 0.4, "R_Collar": 0.4,
    "L_Wrist": 0.4, "R_Wrist": 0.4,
    "L_Hand": 0.3, "R_Hand": 0.3,
}


class KinematicWeightedPoseLoss(nn.Module):
    """Quadratic Bregman Divergence weighted by kinematic joint importance.
    
    Preserves E[u|x_t] minimizer while scaling gradients to prevent vanishing loss.
    """
    def __init__(self, scale_factor: float = 5.0, pose_dim: int = 3):
        super().__init__()
        self.scale_factor = scale_factor
        self.num_joints = 24
        self.pose_dim = pose_dim

        # Assemble weight vector across joints in exact SMPL order
        raw_weights = [JOINT_MPJAE_WEIGHTS[name] for name in SMPL_24_JOINTS]
        weight_tensor = torch.tensor(raw_weights, dtype=torch.float32)
        
        # Normalize so mean weight across all joints is exactly 1.0
        normalized_weights = weight_tensor / weight_tensor.mean()
        
        # Shape: [1, 1, 24, 1] to broadcast over [Batch, Time, Joints, Dim]
        self.register_buffer("weights", normalized_weights.view(1, 1, self.num_joints, 1))

    def forward(self, v_pred_masked: torch.Tensor, u_true: torch.Tensor, M_targ: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v_pred_masked: [B, T, J, D] predicted velocity
            u_true: [B, T, J, D] ground truth velocity target
            M_targ: [B, T] boolean mask of active generation frames
        """
        diff = v_pred_masked - u_true
        weighted_sq_err = (diff ** 2) * self.weights
        
        mask_4d = M_targ.unsqueeze(-1).unsqueeze(-1)
        valid_elements = mask_4d.sum() * self.num_joints * self.pose_dim + 1e-8
        
        unscaled_loss = (weighted_sq_err * mask_4d).sum() / valid_elements
        return unscaled_loss * self.scale_factor