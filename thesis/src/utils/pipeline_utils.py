import numpy as np
import json
from pathlib import Path
import yaml
import torch
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from smplx.body_models import SMPL
from thesis.src.evaluate_h36m import H36MEvaluator
from thesis.src.evaluate_smpl import SMPLEvaluator
from thesis.src.evaluate_distributions import DistributionComparator
from thesis.src.generate_prior import generate_motion_prior_from_prefix
from thesis.src.utils.geometry_utils import forward_to_h36m

from thesis.src.utils.rendering.render_h36m_gif import render_three_way_gif
from thesis.src.utils.visualize_metrics.visualize_h36m_metric_dist import (
    plot_dataset_summary_stats, plot_pd_feature_violins, plot_pd_feature_comparison_plots,
    prepare_dataframe, prepare_combined_dataframe
)
from thesis.src.utils.visualize_metrics.visualize_smpl_metric_dist import (
    plot_smpl_mpjae, plot_arm_swing_metrics, plot_sparc_metrics
)


def validate_config(cfg):
    valid_reps = ["3D", "6D"]
    rep = cfg['data'].get('representation')
    assert rep in valid_reps, f"[Config Error] data.representation must be one of {valid_reps}, got {rep}"

def load_config(CONFIG_PATH="thesis/configs/baseline_3d.yaml"):
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    model_name = cfg['model']['name']
    cfg['paths']['output_dir'] = cfg['paths']['output_dir'].format(model_name=model_name)
    return cfg

def unpack_inference_outputs(outputs):
    """Safely unpacks variable-length inference outputs from conditional and joint models."""
    gen_pose = outputs[0]
    gen_trans = outputs[1]
    # Get severity tensor if it exists (Joint model)
    gen_severity = outputs[2] if len(outputs) > 2 else None
    # Get scalar severity value for logging/rendering
    y_0_prior = outputs[3] if len(outputs) > 3 else None
    return gen_pose, gen_trans, gen_severity, y_0_prior

def render_anchor_gifs(anchors, pl_module, is_joint_model, vis_dir, smpl_model, h36m_regressor, display_epoch):
    """
    Executes inference on anchor sequences, reconstructs the motion priors, 
    and renders 3-way comparison GIFs for WandB logging.
    """
    gif_paths = []
    
    for sev_val, anchor_data in anchors.items():
        gen = torch.Generator(device=pl_module.device).manual_seed(anchor_data["seed"])

        # Run inference based on generation mode and model type
        if pl_module.gen_mode == 'one_shot' and is_joint_model:
            outputs = pl_module._run_oneshot_inference(
                anchor_data["batch"],
                num_steps=pl_module.num_steps,
                x_0=anchor_data["x_0"],
                y_0=anchor_data["y_0"],
                generator=gen,
            )
        elif pl_module.gen_mode == 'one_shot' and not is_joint_model:
            outputs = pl_module._run_oneshot_inference(
                anchor_data["batch"],
                num_steps=pl_module.num_steps,
                x_0=anchor_data["x_0"],
                generator=gen,
            )
        elif pl_module.gen_mode == 'ar_rollout' and is_joint_model:
            outputs = pl_module._run_ar_inference(
                anchor_data["batch"],
                num_steps=pl_module.num_steps,
                y_0=anchor_data["y_0"],
                generator=gen,
            )
        elif pl_module.gen_mode == 'ar_rollout' and not is_joint_model:
            outputs = pl_module._run_ar_inference(
                anchor_data["batch"],
                num_steps=pl_module.num_steps,
                generator=gen,
            )

        gen_full_pose, gen_full_trans, gen_severity, _ = unpack_inference_outputs(outputs)
        gen_sev_val = gen_severity[0].item() if gen_severity is not None else None
        
        # Extract valid unpadded frames based on sequence length
        l = anchor_data["batch"]['seq_len'][0].item()
        gt_full_pose = anchor_data["batch"]['pose'][0, :l]
        gt_full_trans = anchor_data["batch"]['trans'][0, :l]
        gen_full_pose = gen_full_pose[0, :l]
        gen_full_trans = gen_full_trans[0, :l]

        # Reconstruct the exact Prior for visualization 
        if pl_module.gen_mode == 'one_shot':
            prior_full_pose = torch.cat([
                gt_full_pose[:pl_module.prefix_len], 
                anchor_data["x_0"]['pose'][0, pl_module.prefix_len:l]
            ], dim=0)
            
            prior_full_trans = torch.cat([
                gt_full_trans[:pl_module.prefix_len], 
                anchor_data["x_0"]['trans'][0, pl_module.prefix_len:l]
            ], dim=0)
        else:
            # For AR rollout, reconstruct the prior by generating it window-by-window with the same random seed
            prior_gen = torch.Generator(device=pl_module.device).manual_seed(anchor_data["seed"])
            
            prior_pose = gt_full_pose[:pl_module.prefix_len].unsqueeze(0)
            prior_trans = gt_full_trans[:pl_module.prefix_len].unsqueeze(0)
            
            curr_idx = pl_module.prefix_len
            while curr_idx < l:
                window_prefix_pose = gen_full_pose[curr_idx - pl_module.prefix_len : curr_idx].unsqueeze(0)
                window_prefix_trans = gen_full_trans[curr_idx - pl_module.prefix_len : curr_idx].unsqueeze(0)
                target_frames = min(pl_module.AR_window_size - pl_module.prefix_len, l - curr_idx)
                
                prior_dict = generate_motion_prior_from_prefix(
                    window_prefix_pose, window_prefix_trans, target_frames, 
                    s_scale=pl_module.prior_noise_scale, generator=prior_gen
                )
                
                prior_pose = torch.cat([prior_pose, prior_dict['pose']], dim=1)
                prior_trans = torch.cat([prior_trans, prior_dict['trans']], dim=1)
                curr_idx += target_frames
                
            prior_full_pose = prior_pose[0]
            prior_full_trans = prior_trans[0]
        
        seq_gt = forward_to_h36m(gt_full_pose, gt_full_trans, smpl_model, h36m_regressor, pl_module.device)
        seq_prior = forward_to_h36m(prior_full_pose, prior_full_trans, smpl_model, h36m_regressor, pl_module.device)
        seq_gen = forward_to_h36m(gen_full_pose, gen_full_trans, smpl_model, h36m_regressor, pl_module.device)
        
        gif_path = vis_dir / f"anchor_class_{sev_val}_epoch_{display_epoch}.gif"
        render_three_way_gif(seq_gt, seq_prior, seq_gen, sev_val, gif_path, gen_severity=gen_sev_val)
        gif_paths.append(gif_path)

    return gif_paths

def format_and_convert(data_dict, cfg, is_joint_model=False, save_to_disk=False):
    """Processes generation outputs entirely in-memory. Optionally saves arrays for final tests."""
    rep = cfg['data'].get('representation', '6D')
    out_dir = Path(cfg['paths']['output_dir'])

    h36m_dir = out_dir / "h36m"
    rep_dir = out_dir / f"{rep}_SMPL"
    if save_to_disk:
        for d in [h36m_dir, rep_dir]: d.mkdir(parents=True, exist_ok=True)

    gt_pose_dict, gen_pose_dict = {}, {}
    gt_h36m_dict, gen_h36m_dict = {}, {}
    gt_labels, gen_labels = {"key_to_severity": {}}, {"key_to_severity": {}}
    gen_sevs_list = data_dict["gen_severities"] if is_joint_model else data_dict["severities"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smpl_model = SMPL(model_path='thesis/data/care_pd_preprocessing/SMPL_NEUTRAL.pkl', num_betas=10).eval().to(device)
    h36m_regressor = torch.tensor(np.load('thesis/data/care_pd_preprocessing/J_regressor_h36m_correct.npy'), dtype=torch.float32).to(device)

    gt_h36m_all = [
        forward_to_h36m(pose_seq[0], trans_seq[0], smpl_model, h36m_regressor, device)
        for pose_seq, trans_seq in zip(data_dict["gt"]["pose"], data_dict["gt"]["trans"])
    ]
    gen_h36m_all = [
        forward_to_h36m(pose_seq[0], trans_seq[0], smpl_model, h36m_regressor, device)
        for pose_seq, trans_seq in zip(data_dict["gen"]["pose"], data_dict["gen"]["trans"])
    ]

    for i, gt_sev in enumerate(data_dict["severities"]):
        seq_key = f"seq_{i:03d}"

        gt_pose_dict[seq_key] = data_dict["gt"]["pose"][i][0].cpu().numpy()
        gt_pose_dict[f"{seq_key}_trans"] = data_dict["gt"]["trans"][i][0].cpu().numpy()
        gen_pose_dict[seq_key] = data_dict["gen"]["pose"][i][0].cpu().numpy()
        gen_pose_dict[f"{seq_key}_trans"] = data_dict["gen"]["trans"][i][0].cpu().numpy()

        gt_h36m_dict[seq_key] = gt_h36m_all[i]
        gen_h36m_dict[seq_key] = gen_h36m_all[i]

        gen_sev = gen_sevs_list[i]
        gt_labels["key_to_severity"][seq_key] = gt_sev
        gen_labels["key_to_severity"][seq_key] = gen_sev

    if save_to_disk:
        np.savez(rep_dir / f"ground_truth_{rep.lower()}.npz", **gt_pose_dict)
        np.savez(rep_dir / f"generated_{rep.lower()}.npz", **gen_pose_dict)
        np.savez(h36m_dir / "ground_truth_3d_world.npz", **gt_h36m_dict)
        np.savez(h36m_dir / "generated_3d_world.npz", **gen_h36m_dict)

        with open(h36m_dir / "gt_labels.json", 'w') as f: json.dump(gt_labels, f)
        with open(h36m_dir / "gen_labels.json", 'w') as f: json.dump(gen_labels, f)
        print(f"Saved Final Test Set datasets to disk at {out_dir}")

    return {
        "gt_pose_dict": gt_pose_dict, "gen_pose_dict": gen_pose_dict,
        "gt_h36m_dict": gt_h36m_dict, "gen_h36m_dict": gen_h36m_dict,
        "gt_key_to_severity": gt_labels["key_to_severity"],
        "gen_key_to_severity": gen_labels["key_to_severity"],
        "out_dir": out_dir, "prior_severities": data_dict.get("prior_severities", [])
    }

def evaluate_and_plot_distributions(memory_data, min_z_travel=0.5, is_joint_model=False, step_name="Validation"):
    """Unified Evaluation Engine: Evaluates dictionaries in memory and creates PNGs. Returns metrics dict and image paths."""
    out_dir = memory_data["out_dir"]
    vis_out_dir = out_dir / f"visualizations_{step_name.replace(' ', '_')}"
    vis_out_dir.mkdir(parents=True, exist_ok=True)

    # Evaluating
    h36m_eval = H36MEvaluator(fps=30, min_z_travel=min_z_travel)
    gt_h36m_data, _ = h36m_eval.evaluate_from_memory(memory_data["gt_h36m_dict"], memory_data["gt_key_to_severity"])
    gen_h36m_data, _ = h36m_eval.evaluate_from_memory(memory_data["gen_h36m_dict"], memory_data["gen_key_to_severity"])
    
    smpl_eval = SMPLEvaluator()
    smpl_summary, smpl_cache_data = smpl_eval.evaluate_from_memory(
        memory_data["gt_pose_dict"], memory_data["gen_pose_dict"], memory_data["gt_key_to_severity"]
    )

    comparator = DistributionComparator()
    h36m_dist_df = comparator._format_results_to_dataframe(comparator.compare(gt_h36m_data, gen_h36m_data))

    gt_comp, gen_comp = defaultdict(dict), defaultdict(dict)
    
    # Define aggregation targets for the text balloons
    target_sparc_joints = ['L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle']
    categories = [
        'Overall', 'Lower Body', 'Upper Body', 'Hips', 
        'Knees', 'Ankles', 'Shoulders', 'Left Body', 'Right Body'
    ]
    
    for sev_key, metrics in smpl_cache_data.get("raw_distributions", {}).items():
        c_key = "overall" if sev_key == "Overall" else sev_key.replace("Class ", "")
        
        # Arm Swing
        gt_comp[c_key]["Swing Asymmetry (SI)"] = np.array(metrics.get("GT_Symmetry_Index", []))
        gen_comp[c_key]["Swing Asymmetry (SI)"] = np.array(metrics.get("Gen_Symmetry_Index", []))
        
        # Standalone Knees
        gt_knees, gen_knees = [], []
        for j in ['L_Knee', 'R_Knee']:
            gt_knees.extend(metrics.get(f"GT_SPARC_{j}", []))
            gen_knees.extend(metrics.get(f"Gen_SPARC_{j}", []))
        gt_comp[c_key]["SPARC_Knees"] = np.array(gt_knees)
        gen_comp[c_key]["SPARC_Knees"] = np.array(gen_knees)
        
        # Lower Limbs Pooled
        gt_legs, gen_legs = [], []
        for j in target_sparc_joints:
            gt_legs.extend(metrics.get(f"GT_SPARC_{j}", []))
            gen_legs.extend(metrics.get(f"Gen_SPARC_{j}", []))
        gt_comp[c_key]["SPARC_Lower_Limbs"] = np.array(gt_legs)
        gen_comp[c_key]["SPARC_Lower_Limbs"] = np.array(gen_legs)

        # Broad Categories
        for cat in categories:
            gt_comp[c_key][f"SPARC_{cat}"] = np.array(metrics.get(f"GT_SPARC_{cat}", []))
            gen_comp[c_key][f"SPARC_{cat}"] = np.array(metrics.get(f"Gen_SPARC_{cat}", []))
        
    smpl_dist_df = comparator._format_results_to_dataframe(comparator.compare(gt_comp, gen_comp))

    # Plotting
    plot_dataset_summary_stats(prepare_dataframe(gt_h36m_data), vis_out_dir, prefix="gt_", dataset_label="Ground Truth Baseline")
    plot_pd_feature_violins(prepare_dataframe(gt_h36m_data), vis_out_dir, prefix="gt_", dataset_label="Ground Truth Baseline")
    plot_dataset_summary_stats(prepare_dataframe(gen_h36m_data), vis_out_dir, prefix="gen_", dataset_label=step_name)
    plot_pd_feature_violins(prepare_dataframe(gen_h36m_data), vis_out_dir, prefix="gen_", dataset_label=step_name)
    plot_pd_feature_comparison_plots(prepare_combined_dataframe(gt_h36m_data, gen_h36m_data), h36m_dist_df, vis_out_dir)
    
    plot_smpl_mpjae(smpl_cache_data, vis_out_dir)
    plot_arm_swing_metrics(smpl_cache_data, vis_out_dir, distances_df=smpl_dist_df)
    plot_sparc_metrics(smpl_cache_data, vis_out_dir, distances_df=smpl_dist_df)

    if is_joint_model:
        y_true = [v for k, v in memory_data["gt_key_to_severity"].items() if k.startswith("seq_")]
        y_pred = [v for k, v in memory_data["gen_key_to_severity"].items() if k.startswith("seq_")]
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]), annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
        plt.title('Actual Label vs Predicted Label Correlation', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
        plt.ylabel('Actual Label', fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(vis_out_dir / "label_confusion_matrix.png", dpi=300); plt.close()

        prior_sevs = memory_data.get("prior_severities", None)
        if prior_sevs:
            plt.figure(figsize=(6, 5))
            sns.heatmap(confusion_matrix(prior_sevs, y_pred, labels=[0, 1, 2, 3]), annot=True, fmt='d', cmap='Oranges', xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
            plt.title('Prior State vs Predicted Label Correlation', fontsize=12, fontweight='bold')
            plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
            plt.ylabel('Prior State (Jump Start)', fontsize=11, fontweight='bold')
            plt.tight_layout()
            plt.savefig(vis_out_dir / "prior_state_correlation_matrix.png", dpi=300); plt.close()

    # Return metrics dictionary for logging
    metrics_dict = {
        "eval_metrics/Mean_Hellinger_H36M": float(h36m_dist_df["Hellinger"].mean()),
        "eval_metrics/Mean_KS_H36M": float(h36m_dist_df["KS_Stat"].mean()),
        "eval_metrics/Mean_Hellinger_SMPL": float(smpl_dist_df["Hellinger"].mean()),
        "eval_metrics/Mean_KS_SMPL": float(smpl_dist_df["KS_Stat"].mean()),
        "physical_realism/mean_floating_gt": float(np.nanmean(gt_h36m_data["overall"]["floating"])),
        "physical_realism/mean_floating_gen": float(np.nanmean(gen_h36m_data["overall"]["floating"])),
        "physical_realism/mean_foot_disp_gt": float(np.nanmean(gt_h36m_data["overall"]["mean_stance_displacement"])),
        "physical_realism/mean_foot_disp_gen": float(np.nanmean(gen_h36m_data["overall"]["mean_stance_displacement"])),
    }
        
    return metrics_dict, vis_out_dir

def plot_physical_realism_tracking(val_epochs, floating_gt, floating_gen, foot_disp_gt, foot_disp_gen, out_dir):
    """Generates a Matplotlib tracked history plot over all epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(val_epochs, floating_gt, color='cornflowerblue', linestyle='--', linewidth=2.5, label='Ground Truth Baseline')
    axes[0].plot(val_epochs, floating_gen, color='salmon', linestyle='-', linewidth=2.5, label='Generated Model')
    axes[0].set_title("Floating over Epochs", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Epoch", fontweight='bold')
    axes[0].set_ylabel("Mean Floating (m)", fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(val_epochs, foot_disp_gt, color='cornflowerblue', linestyle='--', linewidth=2.5, label='Ground Truth Baseline')
    axes[1].plot(val_epochs, foot_disp_gen, color='salmon', linestyle='-', linewidth=2.5, label='Generated Model')
    axes[1].set_title("Foot Displacement (Skating) over Epochs", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Epoch", fontweight='bold')
    axes[1].set_ylabel("Mean Displacement (m)", fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    tracking_path = Path(out_dir) / "physical_realism_tracking.png"
    plt.savefig(tracking_path, dpi=300)
    plt.close()
    
    return tracking_path