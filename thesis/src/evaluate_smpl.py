import torch
import numpy as np

class SMPLEvaluator:
    def __init__(self):
        """Evaluator for 6D SMPL pose sequences using Geodesic Distance on SO(3)."""
        pass

    def _convert_6d_to_rmat(self, pose_6d_tensor):
        """Gram-Schmidt to convert (T, 6) to (T, 3, 3) rotation matrices.
        
        Account for NN predicting first two rows of 9D rotation matrix, due
        to original CARE-PD training data format, so stack vectors as rows to
        construct rotation matrix.
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

    def compute_mpjae(self, gt_6d, gen_6d):
        """Computes the Mean Per Joint Angular Error (MPJAE) using Geodesic Distance.

        Expects inputs of shape (T, J, 6). 
        Returns error in radians.
        """
        R_gt = self._convert_6d_to_rmat(gt_6d)   # (T, J, 3, 3)
        R_gen = self._convert_6d_to_rmat(gen_6d) # (T, J, 3, 3)

        # Truncate to the length of the shortest sequence to allow comparison
        min_frames = min(R_gt.shape[0], R_gen.shape[0])
        R_gt = R_gt[:min_frames]
        R_gen = R_gen[:min_frames]

        # Compute relative rotation matrix: R_rel = R_hat * R^T
        R_rel = torch.matmul(R_gen, R_gt.transpose(-1, -2))

        # Get trace (sum of diagonal elements) for each 3x3 matrix
        trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

        # Compute cosine of the angle and clamp value
        # Floating point errors can give values outside [-1, 1]
        cos_theta = (trace - 1.0) / 2.0
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

        # Compute geo distance and mean over all frames
        d_geo = torch.acos(cos_theta)
        mpjae = torch.mean(d_geo).item()
        
        return mpjae

    def evaluate_dataset(self, gt_npz_path, gen_npz_path):
        """Loads GT and Generated 6D datasets and computes overall MPJAE.

        Assumes keys match between datasets.
        """        
        gt_data = np.load(gt_npz_path, allow_pickle=True)
        gen_data = np.load(gen_npz_path, allow_pickle=True)
        
        # Resolve potentially nested dicts from npz
        gt_data = gt_data['arr_0'].item() if 'arr_0' in gt_data.files else {k: gt_data[k] for k in gt_data.files}
        gen_data = gen_data['arr_0'].item() if 'arr_0' in gen_data.files else {k: gen_data[k] for k in gen_data.files}

        common_keys = [k for k in gt_data.keys() if k in gen_data.keys() and not k.endswith('_trans')]
        
        if not common_keys:
            print("Error: No matching pose sequences found between GT and Gen datasets.")
            return

        if len(common_keys) < len(gt_data) or len(common_keys) < len(gen_data):
            print("Error: Not all keys are the same.")

        total_mpjae = []
        for key in common_keys:
            mpjae = self.compute_mpjae(gt_data[key], gen_data[key])
            total_mpjae.append(mpjae)

        overall_mpjae = np.mean(total_mpjae)
        
        print(f"Evaluated {len(common_keys)} matching sequences.")
        print(f"Overall Mean Per Joint Angular Error (MPJAE): {overall_mpjae:.6f} radians")
        
        return overall_mpjae