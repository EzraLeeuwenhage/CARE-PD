import math
import torch
import torch.nn.functional as F

from thesis.src.generate_prior import generate_label_prior
from thesis.src.model_conditional import ConditionalBaselineModel
from thesis.src.model_backbones import (
    JointBaselineBackbone,
    FlowHead,
    JumpHead,
    generate_x0,
    mask,
    add,
    sub, 
    mul,
    add_noise
)


def ctmc_jump_step(y_current, Q_pred, dt, num_classes):
    """Simulates a CTMC jump step over time interval dt using exact holding-time probabilities.
    
    TODO: cite holderrieth generator matching for using he exact exponential holding-time scheduler
    """
    Q_pred = Q_pred.float()
    
    # Ensure jump rates to other classes are non-negative
    # and zero out self-transition rate (diagonal of rate matrix Q)
    rates = F.relu(Q_pred)    
    mask = F.one_hot(y_current, num_classes=num_classes).bool()
    rates[mask] = 0.0
    
    # Total exit rate: lambda_i = sum_{j != i} Q_ij
    exit_rates = rates.sum(dim=-1, keepdim=True)
    
    # Exact continuous-time probabilities over interval dt
    stay_prob = torch.exp(-exit_rates * dt)
    jump_prob = 1.0 - stay_prob
    
    # Distribute jump probability across candidate states proportional to their rates
    jump_distribution = rates / (exit_rates + 1e-8)
    probs = jump_prob * jump_distribution
    
    # Insert self-transition probability along the diagonal
    probs[mask] = stay_prob.squeeze(-1)
    
    # Normalize to guard against floating-point rounding errors
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


class JointBaselineModel(ConditionalBaselineModel):
    """Multimodal joint generator model."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.alpha_trans = cfg['training'].get('alpha_pose_trans', 0.15) 
        self.alpha_label = cfg['training'].get('alpha_motion_label', 0.50)
        self.num_classes = cfg['model'].get('num_classes', 4)
        
        hidden_dim = cfg['model'].get('hidden_dim', 1024)
        class_embed_dim = cfg['model'].get('class_embed_dim', 64)
        time_embed_dim = cfg['model'].get('time_embed_dim', 64)
        
        self.backbone = JointBaselineBackbone(cfg, hidden_dim, class_embed_dim, time_embed_dim)
        self.flow_head = FlowHead(hidden_dim, self.backbone.seq_len, self.backbone.num_joints, self.backbone.pose_dim)
        self.jump_head = JumpHead(hidden_dim, num_classes=self.num_classes)

    def forward(self, x_tau_dict, t, y_tau):
        shared_latent = self.backbone(x_tau_dict, t, y_tau)
        u_theta = self.flow_head(shared_latent) 
        Q_theta = self.jump_head(shared_latent) 
        return u_theta, Q_theta

    def _compute_ar_rollout_loss(self, batch):
        x_1 = {'pose': batch['pose'], 'trans': batch['trans']}
        y_target = batch['severity']
        batch_size = y_target.shape[0]

        M_cond = batch['cond_mask'] 
        M_targ = ~M_cond

        x_0 = generate_x0(x_1, self.prefix_len, self.prior_noise_scale)
        tau = torch.rand(batch_size, 1, device=self.device)
        noise_std = self.cfg['training'].get('prefix_noise_std', 0.05)
        x_1_noisy = add_noise(x_1, noise_std)

        canvas_cond = mask(x_1_noisy, M_cond)
        canvas_targ = mask(add(mul(x_0, 1 - tau), mul(x_1, tau)), M_targ)
        x_tau = add(canvas_cond, canvas_targ)

        u_true = mask(sub(x_1, x_0), M_targ)

        # Discrete CTMC Jump Mixture Path
        mask_jump = torch.rand(batch_size, device=self.device) < tau.squeeze(-1)
        y_random = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        y_tau = torch.where(mask_jump, y_target, y_random)
        
        v_pred, Q_pred = self(x_tau, tau, y_tau)
        v_pred_masked = mask(v_pred, M_targ)
        
        _, _, num_joints, pose_dim = v_pred['pose'].shape
        _, _, trans_dim = v_pred['trans'].shape
        valid_pose_elements = M_targ.sum() * num_joints * pose_dim + 1e-8
        valid_trans_elements = M_targ.sum() * trans_dim + 1e-8

        # note: loss_pose computes micro-average (per element)
        # note: loss_label computes the macro-average (per sequence)
        # TODO: keep in mind that this causes variability in loss magnitudes between motion and label losses
        loss_pose = F.mse_loss(v_pred_masked['pose'], u_true['pose'], reduction='sum') / valid_pose_elements
        loss_trans = F.mse_loss(v_pred_masked['trans'], u_true['trans'], reduction='sum') / valid_trans_elements
        loss_label = F.cross_entropy(Q_pred, y_target)
        
        return loss_pose, loss_trans, loss_label

    def _compute_one_shot_loss(self, batch):
        x_1 = {'pose': batch['pose'], 'trans': batch['trans']}
        y_target = batch['severity']
        batch_size = y_target.shape[0]
        
        M_cond = batch['cond_mask']
        M_pad = batch['pad_mask']
        M_static = M_cond | M_pad
        M_targ = ~M_static

        x_0 = generate_x0(x_1, self.prefix_len, self.prior_noise_scale)
        tau = torch.rand(batch_size, 1, device=self.device)

        canvas_static = mask(x_1, M_static)
        canvas_targ = mask(add(mul(x_0, 1 - tau), mul(x_1, tau)), M_targ)
        x_tau = add(canvas_static, canvas_targ)

        u_true = mask(sub(x_1, x_0), M_targ)

        # Discrete CTMC Jump Mixture Path
        mask_jump = torch.rand(batch_size, device=self.device) < tau.squeeze(-1)
        y_random = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        y_tau = torch.where(mask_jump, y_target, y_random)

        v_pred, Q_pred = self(x_tau, tau, y_tau)
        v_pred_masked = mask(v_pred, M_targ)
        
        _, _, num_joints, pose_dim = v_pred['pose'].shape
        _, _, trans_dim = v_pred['trans'].shape
        valid_pose_elements = M_targ.sum() * num_joints * pose_dim + 1e-8
        valid_trans_elements = M_targ.sum() * trans_dim + 1e-8

        # see note on loss averaging in ar_rollout_loss method
        loss_pose = F.mse_loss(v_pred_masked['pose'], u_true['pose'], reduction='sum') / valid_pose_elements
        loss_trans = F.mse_loss(v_pred_masked['trans'], u_true['trans'], reduction='sum') / valid_trans_elements
        loss_label = F.cross_entropy(Q_pred, y_target)

        return loss_pose, loss_trans, loss_label

    def training_step(self, batch, batch_idx):
        gen_mode = self.cfg['model'].get('generation_mode', 'ar_rollout')
        
        if gen_mode == 'one_shot':
            loss_pose, loss_trans, loss_label = self._compute_one_shot_loss(batch)
        else:
            loss_pose, loss_trans, loss_label = self._compute_ar_rollout_loss(batch)

        loss_motion = ((1.0 - self.alpha_trans) * loss_pose) + (self.alpha_trans * loss_trans)
        loss_total = ((1.0 - self.alpha_label) * loss_motion) + (self.alpha_label * loss_label)

        self.log("train/loss_pose", loss_pose, on_step=False, on_epoch=True)
        self.log("train/loss_trans", loss_trans, on_step=False, on_epoch=True)
        
        self.log("train/loss_motion", loss_motion, on_step=False, on_epoch=True)
        self.log("train/loss_label", loss_label, on_step=False, on_epoch=True)
        self.log("train/loss_total", loss_total, on_step=False, on_epoch=True)
        return loss_total

    def solve_ODE(self, x_tau, y_tau, M_targ, num_steps=100, sample_labels=True):
        """Euler ODE Solver strictly masking velocity updates to prevent prefix drift + label sampling."""
        batch_size = x_tau['pose'].shape[0]
        dt = 1.0 / num_steps

        for step in range(num_steps):
            tau = step * dt
            tau_tensor = torch.full((batch_size, 1), tau, device=self.device)
            
            u_theta, Q_theta = self(x_tau, tau_tensor, y_tau)
            v_pred_masked = mask(u_theta, M_targ)
            
            x_tau = add(x_tau, mul(v_pred_masked, torch.tensor([dt], device=self.device)))
            
            if sample_labels:
                y_tau = ctmc_jump_step(y_tau, Q_theta, dt, self.num_classes)

        return x_tau, y_tau

    def _run_oneshot_inference(self, batch, num_steps=100, force_joint_conditioning=False, x_0=None, y_0=None, generator=None):
        """Generates the target sequence and discrete labels globally in a single pass."""
        true_dict = {'pose': batch['pose'], 'trans': batch['trans']}
        batch_size = batch['severity'].shape[0]
        
        M_cond = batch['cond_mask']
        M_pad = batch['pad_mask']
        M_static = M_cond | M_pad
        M_targ = ~M_static
        
        if x_0 is None:
            x_0 = generate_x0(true_dict, self.prefix_len, self.prior_noise_scale, generator=generator)

        x_tau = add(mask(true_dict, M_static), mask(x_0, M_targ))
        
        if force_joint_conditioning:
            y_tau = batch['severity'].clone()
            sample_labels = False
            y_0_prior = y_tau.clone()
        else:
            if y_0 is None:
                y_0 = generate_label_prior(batch_size, self.num_classes, self.device, generator=generator)
            y_tau = y_0.clone()
            sample_labels = True
            y_0_prior = y_0.clone()

        gen_dict, gen_labels = self.solve_ODE(x_tau, y_tau, M_targ, num_steps, sample_labels=sample_labels)
        
        return gen_dict['pose'], gen_dict['trans'], gen_labels, y_0_prior

    def _run_ar_inference(self, batch, num_steps=100, force_joint_conditioning=False, y_0=None, generator=None):
        """Generates the target sequence autoregressively and solves CTMC on the first window."""
        true_dict = {'pose': batch['pose'], 'trans': batch['trans']}
        batch_size = batch['severity'].shape[0]
        
        max_seq_length = batch['seq_len'].max().item()
        
        gen_pose = true_dict['pose'][:, :self.prefix_len]
        gen_trans = true_dict['trans'][:, :self.prefix_len]
        num_joints, pose_dim = gen_pose.shape[2], gen_pose.shape[3]
        
        M_cond = torch.zeros((batch_size, self.AR_window_size), dtype=torch.bool, device=self.device)
        M_cond[:, :self.prefix_len] = True
        M_targ = ~M_cond

        # Calculate NFEs per window to ensure fair comparison with one-shot generation
        total_target_frames = max_seq_length - self.prefix_len
        frames_per_window = self.AR_window_size - self.prefix_len
        num_windows = max(1, math.ceil(total_target_frames / frames_per_window))
        steps_per_window = max(1, num_steps // num_windows)

        if force_joint_conditioning:
            current_labels = batch['severity'].clone()
            sample_labels = False
            y_0_prior = current_labels.clone()
        else:
            if y_0 is None:
                y_0 = generate_label_prior(batch_size, self.num_classes, self.device, generator=generator)
            current_labels = y_0.clone()
            sample_labels = True
            y_0_prior = y_0.clone()
        
        while gen_pose.shape[1] < max_seq_length:
            window_pose = torch.zeros((batch_size, self.AR_window_size, num_joints, pose_dim), device=self.device)
            window_trans = torch.zeros((batch_size, self.AR_window_size, 3), device=self.device)

            window_pose[:, :self.prefix_len] = gen_pose[:, -self.prefix_len:]
            window_trans[:, :self.prefix_len] = gen_trans[:, -self.prefix_len:]
            x1_window = {'pose': window_pose, 'trans': window_trans}
            
            x0_window = generate_x0(x1_window, self.prefix_len, self.prior_noise_scale, generator=generator)
            x_tau = add(mask(x1_window, M_cond), mask(x0_window, M_targ))
            
            x1, current_labels = self.solve_ODE(x_tau, current_labels, M_targ, steps_per_window, sample_labels=sample_labels)
            
            # TODO: remember we have to freeze labels after the first generated window
            # Model can only be trained to flow noise to posterior, not to flow posterior again to posterior
            sample_labels = False

            gen_pose = torch.cat([gen_pose, x1['pose'][:, self.prefix_len:]], dim=1)
            gen_trans = torch.cat([gen_trans, x1['trans'][:, self.prefix_len:]], dim=1)
        
        return gen_pose[:, :max_seq_length], gen_trans[:, :max_seq_length], current_labels, y_0_prior

    def validation_step(self, batch, batch_idx):
        """Automated validation step for Joint Continuous Flow + CTMC."""
        seq_len = batch['seq_len']
        batch_size = batch['severity'].shape[0]
        
        if self.gen_mode == 'one_shot':
            outputs = self._run_oneshot_inference(batch, self.num_steps)
        elif self.gen_mode == 'ar_rollout':
            outputs = self._run_ar_inference(batch, self.num_steps)
        gen_pose, gen_trans, gen_labels, _ = outputs

        # Extract only the valid frames per sequence
        gt_pose_list, gen_pose_list, gt_trans_list, gen_trans_list = [], [], [], []
        
        for i in range(batch_size):
            l = seq_len[i].item()
            gt_pose_list.append(batch['pose'][i:i+1, :l])
            gen_pose_list.append(gen_pose[i:i+1, :l])
            gt_trans_list.append(batch['trans'][i:i+1, :l])
            gen_trans_list.append(gen_trans[i:i+1, :l])

        # Move pose to CPU, keep trans on GPU 
        gt_pose_flat = torch.cat(gt_pose_list, dim=1).cpu()
        gen_pose_flat = torch.cat(gen_pose_list, dim=1).cpu()
        gt_trans_flat = torch.cat(gt_trans_list, dim=1)
        gen_trans_flat = torch.cat(gen_trans_list, dim=1)

        # Compute evaluation metrics
        val_mpjae_rad = self.evaluator.compute_mpjae(gt_pose_flat, gen_pose_flat)
        val_mpjae_deg = val_mpjae_rad * (180.0 / math.pi)
        val_trans_mse = F.mse_loss(gen_trans_flat, gt_trans_flat)
        
        self.log("val/mpjae_deg", val_mpjae_deg, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/trans_mse", val_trans_mse, on_step=False, on_epoch=True, sync_dist=True)
        
        # Log categorical label accuracy
        accuracy = (gen_labels == batch['severity']).float().mean()
        self.log("val/label_accuracy", accuracy, on_step=False, on_epoch=True, sync_dist=True)
            
        return val_mpjae_deg