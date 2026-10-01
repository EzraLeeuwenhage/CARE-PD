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
from thesis.src.utils.geometry_utils import batched_forward_to_h36m

from thesis.src.utils.rendering.render_h36m_gif import render_three_way_gif
from thesis.src.utils.visualize_metrics.visualize_h36m_metric_dist import (
    plot_dataset_summary_stats, plot_pd_feature_violins, plot_pd_feature_comparison_plots,
    prepare_dataframe, prepare_combined_dataframe
)
from thesis.src.utils.visualize_metrics.visualize_smpl_metric_dist import (
    plot_smpl_mpjae, 
    plot_clinical_metric_distributions
)
from thesis.src.utils.visualization_utils import plot_stationary_sequence_eval_metrics

# UTILS
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

# GIFS
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
                
                use_cumsum = getattr(pl_module, 'use_cumsum_prior', False)
                prior_dict = generate_motion_prior_from_prefix(
                    window_prefix_pose, window_prefix_trans, target_frames, 
                    prior_noise_scale=pl_module.prior_noise_scale, generator=prior_gen,
                    use_cumsum=use_cumsum
                )
                
                prior_pose = torch.cat([prior_pose, prior_dict['pose']], dim=1)
                prior_trans = torch.cat([prior_trans, prior_dict['trans']], dim=1)
                curr_idx += target_frames
                
            prior_full_pose = prior_pose[0]
            prior_full_trans = prior_trans[0]
        
        triplet_h36m = batched_forward_to_h36m(
            [gt_full_pose, prior_full_pose, gen_full_pose],
            [gt_full_trans, prior_full_trans, gen_full_trans],
            smpl_model, h36m_regressor, pl_module.device
        )
        seq_gt, seq_prior, seq_gen = triplet_h36m[0], triplet_h36m[1], triplet_h36m[2]
        
        gif_path = vis_dir / f"anchor_class_{sev_val}_epoch_{display_epoch}.gif"
        render_three_way_gif(seq_gt, seq_prior, seq_gen, sev_val, gif_path, gen_severity=gen_sev_val)
        gif_paths.append(gif_path)

    return gif_paths

# FORMAT CONVERSION
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

    gt_h36m_all = batched_forward_to_h36m(
        data_dict["gt"]["pose"], data_dict["gt"]["trans"], smpl_model, h36m_regressor, device
    )
    gen_h36m_all = batched_forward_to_h36m(
        data_dict["gen"]["pose"], data_dict["gen"]["trans"], smpl_model, h36m_regressor, device
    )

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

# EVALUATE AND PLOT
def evaluate_and_plot_distributions(memory_data, min_z_travel=0.5, is_joint_model=False, step_name="Validation"):
    """Main Orchestrator: Partitions datasets, executes mobile & stationary tracks, and aggregates logs."""
    out_dir = memory_data["out_dir"]
    vis_out_dir = out_dir / f"visualizations_{step_name.replace(' ', '_')}"
    vis_out_dir.mkdir(parents=True, exist_ok=True)

    # Partition data using SMPL root translations
    mobile_data, stat_data = partition_motion_data(memory_data, min_travel=min_z_travel, disp_mode="euclidean")

    # Execute Mobile Evaluation Track
    metrics_dict = evaluate_standard_track(mobile_data, vis_out_dir, step_name=step_name)

    # Execute Stationary Evaluation Track
    stationary_metrics = evaluate_stationary_track(stat_data, vis_out_dir)
    metrics_dict.update(stationary_metrics)

    # Confusion Matrices (Joint Model only)
    if is_joint_model:
        y_true = [v for k, v in memory_data["gt_key_to_severity"].items() if k.startswith("seq_")]
        y_pred = [v for k, v in memory_data["gen_key_to_severity"].items() if k.startswith("seq_")]
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]), annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
        plt.title('Actual Label vs Predicted Label Correlation', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
        plt.ylabel('Actual Label', fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(vis_out_dir / "label_confusion_matrix.png", dpi=300)
        plt.close()

    return metrics_dict, vis_out_dir

def partition_motion_data(memory_data, min_travel=0.5, disp_mode="euclidean"):
    """Partitions memory_data into standard and stationary subsets using SMPL root translations.
    
    Computes displacement on SMPL trans to maintain exact consistency with dataloader logic.
    """
    gt_poses = memory_data["gt_pose_dict"]
    gen_poses = memory_data["gen_pose_dict"]

    def _calc_disp(trans):
        if disp_mode == "euclidean":
            return float(np.linalg.norm(trans[-1] - trans[0]))
        elif disp_mode == "z_axis":
            return float(abs(trans[-1, 2] - trans[0, 2]))
        else:
            raise ValueError(f"Unknown displacement mode: {disp_mode}")

    # Identify valid sequence keys (excluding '_trans' suffix keys)
    seq_keys = [k for k in gt_poses.keys() if not k.endswith('_trans')]

    gt_standard_keys, gt_stat_keys = set(), set()
    gen_standard_keys, gen_stat_keys = set(), set()

    for k in seq_keys:
        # Check Ground Truth displacement
        trans_gt = gt_poses[f"{k}_trans"]
        if _calc_disp(trans_gt) >= min_travel:
            gt_standard_keys.add(k)
        else:
            gt_stat_keys.add(k)

        # Check Generated displacement independently
        if f"{k}_trans" in gen_poses:
            trans_gen = gen_poses[f"{k}_trans"]
            if _calc_disp(trans_gen) >= min_travel:
                gen_standard_keys.add(k)
            else:
                gen_stat_keys.add(k)

    def _filter_dict(src_dict, valid_keys):
        return {
            k: v for k, v in src_dict.items() 
            if (k in valid_keys or k.replace('_trans', '') in valid_keys)
        }

    standard_data = {
        "gt_pose_dict": _filter_dict(gt_poses, gt_standard_keys),
        "gen_pose_dict": _filter_dict(gen_poses, gen_standard_keys),
        "gt_h36m_dict": _filter_dict(memory_data["gt_h36m_dict"], gt_standard_keys),
        "gen_h36m_dict": _filter_dict(memory_data["gen_h36m_dict"], gen_standard_keys),
        "gt_key_to_severity": {k: memory_data["gt_key_to_severity"][k] for k in gt_standard_keys if k in memory_data["gt_key_to_severity"]},
        "gen_key_to_severity": {k: memory_data["gen_key_to_severity"][k] for k in gen_standard_keys if k in memory_data["gen_key_to_severity"]},
    }

    stat_data = {
        "gt_pose_dict": _filter_dict(gt_poses, gt_stat_keys),
        "gen_pose_dict": _filter_dict(gen_poses, gen_stat_keys),
        "gt_h36m_dict": _filter_dict(memory_data["gt_h36m_dict"], gt_stat_keys),
        "gen_h36m_dict": _filter_dict(memory_data["gen_h36m_dict"], gen_stat_keys),
        "gt_key_to_severity": {k: memory_data["gt_key_to_severity"][k] for k in gt_stat_keys if k in memory_data["gt_key_to_severity"]},
        "gen_key_to_severity": {k: memory_data["gen_key_to_severity"][k] for k in gen_stat_keys if k in memory_data["gen_key_to_severity"]},
    }

    print(f"  [Partition] GT: {len(gt_standard_keys)} standard, {len(gt_stat_keys)} Stationary | "
          f"Gen: {len(gen_standard_keys)} standard, {len(gen_stat_keys)} Stationary ({disp_mode} >= {min_travel}m)")

    return standard_data, stat_data

def evaluate_standard_track(standard_data, vis_out_dir, step_name="Validation"):
    """Executes the standard kinematic & clinical evaluation exclusively on standard sequences."""
    h36m_eval = H36MEvaluator(fps=30, min_z_travel=0.0)
    gt_h36m_data, _ = h36m_eval.evaluate_from_memory(standard_data["gt_h36m_dict"], standard_data["gt_key_to_severity"])
    gen_h36m_data, _ = h36m_eval.evaluate_from_memory(standard_data["gen_h36m_dict"], standard_data["gen_key_to_severity"])

    smpl_eval = SMPLEvaluator(fps=30)
    smpl_summary, smpl_cache_data = smpl_eval.evaluate_from_memory(
        standard_data["gt_pose_dict"], standard_data["gen_pose_dict"], standard_data["gt_key_to_severity"]
    )

    comparator = DistributionComparator()
    h36m_dist_df = comparator._format_results_to_dataframe(comparator.compare(gt_h36m_data, gen_h36m_data))

    gt_comp, gen_comp = defaultdict(dict), defaultdict(dict)
    clinical_metrics_map = [
        ("Ankle Bradykinesia", "GT_Ankle_Bradykinesia", "Gen_Ankle_Bradykinesia"),
        ("Spine Rigidity", "GT_Spine_Rigidity", "Gen_Spine_Rigidity"),
        ("Ankle SI", "GT_Ankle_SI", "Gen_Ankle_SI"),
        ("Wrist Smoothness AUC", "GT_Wrist_Smoothness_AUC", "Gen_Wrist_Smoothness_AUC"),
        ("Hand Smoothness AUC", "GT_Hand_Smoothness_AUC", "Gen_Hand_Smoothness_AUC"),
    ]

    for sev_key, metrics in smpl_cache_data.get("raw_distributions", {}).items():
        c_key = "overall" if sev_key.lower() == "overall" else sev_key.replace("Class ", "")
        for display_name, gt_k, gen_k in clinical_metrics_map:
            if gt_k in metrics and gen_k in metrics:
                gt_comp[c_key][display_name] = np.array(metrics[gt_k])
                gen_comp[c_key][display_name] = np.array(metrics[gen_k])

    smpl_dist_df = comparator._format_results_to_dataframe(comparator.compare(gt_comp, gen_comp))

    # Standard standard Plotting Suite
    plot_dataset_summary_stats(prepare_dataframe(gt_h36m_data), vis_out_dir, prefix="gt_", dataset_label="Ground Truth Baseline")
    plot_pd_feature_violins(prepare_dataframe(gt_h36m_data), vis_out_dir, prefix="gt_", dataset_label="Ground Truth Baseline")
    plot_dataset_summary_stats(prepare_dataframe(gen_h36m_data), vis_out_dir, prefix="gen_", dataset_label=step_name)
    plot_pd_feature_violins(prepare_dataframe(gen_h36m_data), vis_out_dir, prefix="gen_", dataset_label=step_name)
    plot_pd_feature_comparison_plots(prepare_combined_dataframe(gt_h36m_data, gen_h36m_data), h36m_dist_df, vis_out_dir)

    plot_smpl_mpjae(smpl_cache_data, vis_out_dir)
    plot_clinical_metric_distributions(smpl_cache_data, vis_out_dir, distances_df=smpl_dist_df)

    metrics_dict = {
        "eval_metrics/Mean_Norm_Wasserstein_H36M": float(h36m_dist_df["Norm_Wasserstein"].mean()),
        "eval_metrics/Mean_KS_H36M": float(h36m_dist_df["KS_Stat"].mean()),
        "eval_metrics/Mean_Norm_Wasserstein_SMPL": float(smpl_dist_df["Norm_Wasserstein"].mean()),
        "eval_metrics/Mean_KS_SMPL": float(smpl_dist_df["KS_Stat"].mean()),
        "physical_realism/mean_floating_gt": float(np.nanmean(gt_h36m_data["overall"]["floating"])),
        "physical_realism/mean_floating_gen": float(np.nanmean(gen_h36m_data["overall"]["floating"])),
        "physical_realism/mean_foot_disp_gt": float(np.nanmean(gt_h36m_data["overall"]["mean_stance_displacement"])),
        "physical_realism/mean_foot_disp_gen": float(np.nanmean(gen_h36m_data["overall"]["mean_stance_displacement"])),
    }

    return metrics_dict

def evaluate_stationary_track(stationary_data, vis_out_dir):
    """Evaluates the 4 stationary metrics and renders the 2x2 comparison overview.
    
    Gracefully handles empty Generated stationary data by bypassing the distance comparator.
    """
    n_gt_stat = len([k for k in stationary_data["gt_pose_dict"] if not k.endswith('_trans')])
    n_gen_stat = len([k for k in stationary_data["gen_pose_dict"] if not k.endswith('_trans')])
    has_gen_data = (n_gen_stat > 0)
    print(f"  [Stationary Track] Evaluating {n_gt_stat} GT vs {n_gen_stat} Generated stationary sequences...")

    h36m_eval = H36MEvaluator(fps=30, min_z_travel=0.0)
    stat_h36m_gt, _ = h36m_eval.evaluate_from_memory(stationary_data["gt_h36m_dict"], stationary_data["gt_key_to_severity"])
    stat_h36m_gen, _ = (h36m_eval.evaluate_from_memory(stationary_data["gen_h36m_dict"], stationary_data["gen_key_to_severity"]) 
                        if has_gen_data else ({}, {}))

    smpl_eval = SMPLEvaluator(fps=30)
    _, stat_smpl_cache = smpl_eval.evaluate_from_memory(
        stationary_data["gt_pose_dict"], 
        (stationary_data["gen_pose_dict"] if has_gen_data else stationary_data["gt_pose_dict"]), 
        stationary_data["gt_key_to_severity"]
    )

    stationary_records = []
    gt_stat_comp, gen_stat_comp = defaultdict(dict), defaultdict(dict)
    stat_raw = stat_smpl_cache.get("raw_distributions", {})

    stat_classes = ["overall"] + sorted([k for k in stat_h36m_gt.keys() if k != "overall"])

    for sev_key in stat_classes:
        cls_name = "Overall" if sev_key == "overall" else f"Class {sev_key}"
        c_comp = "overall" if sev_key == "overall" else str(sev_key)

        # Global Speed (H36M)
        v_gt = stat_h36m_gt[sev_key].get("mean_global_walking_speed", [])
        for x in v_gt: 
            stationary_records.append({"Severity Class": cls_name, "Source": "Ground Truth", "Metric": "mean_global_walking_speed", "Value": x})
        gt_stat_comp[c_comp]["Global Walking Speed"] = np.array(v_gt)

        if has_gen_data and sev_key in stat_h36m_gen:
            v_gen = stat_h36m_gen[sev_key].get("mean_global_walking_speed", [])
            for x in v_gen: 
                stationary_records.append({"Severity Class": cls_name, "Source": "Generated", "Metric": "mean_global_walking_speed", "Value": x})
            gen_stat_comp[c_comp]["Global Walking Speed"] = np.array(v_gen)

        # Mean Cadence (H36M)
        c_gt = stat_h36m_gt[sev_key].get("mean_cadence", [])
        for x in c_gt: 
            stationary_records.append({"Severity Class": cls_name, "Source": "Ground Truth", "Metric": "mean_cadence", "Value": x})
        gt_stat_comp[c_comp]["Mean Cadence"] = np.array(c_gt)

        if has_gen_data and sev_key in stat_h36m_gen:
            c_gen = stat_h36m_gen[sev_key].get("mean_cadence", [])
            for x in c_gen: 
                stationary_records.append({"Severity Class": cls_name, "Source": "Generated", "Metric": "mean_cadence", "Value": x})
            gen_stat_comp[c_comp]["Mean Cadence"] = np.array(c_gen)

        # Ankle Bradykinesia (SMPL)
        smpl_k = "Overall" if sev_key == "overall" else f"Class {sev_key}"
        ab_gt = stat_raw.get(smpl_k, {}).get("GT_Ankle_Bradykinesia", [])
        for x in ab_gt: 
            stationary_records.append({"Severity Class": cls_name, "Source": "Ground Truth", "Metric": "ankle_bradykinesia", "Value": x})
        gt_stat_comp[c_comp]["Ankle Bradykinesia"] = np.array(ab_gt)

        if has_gen_data:
            ab_gen = stat_raw.get(smpl_k, {}).get("Gen_Ankle_Bradykinesia", [])
            for x in ab_gen: 
                stationary_records.append({"Severity Class": cls_name, "Source": "Generated", "Metric": "ankle_bradykinesia", "Value": x})
            gen_stat_comp[c_comp]["Ankle Bradykinesia"] = np.array(ab_gen)

        # Wrist Smoothness AUC (SMPL)
        wj_gt = stat_raw.get(smpl_k, {}).get("GT_Wrist_Smoothness_AUC", [])
        for x in wj_gt: 
            stationary_records.append({"Severity Class": cls_name, "Source": "Ground Truth", "Metric": "wrist_smoothness_auc", "Value": x})
        gt_stat_comp[c_comp]["Wrist Smoothness AUC"] = np.array(wj_gt)

        if has_gen_data:
            wj_gen = stat_raw.get(smpl_k, {}).get("Gen_Wrist_Smoothness_AUC", [])
            for x in wj_gen: 
                stationary_records.append({"Severity Class": cls_name, "Source": "Generated", "Metric": "wrist_smoothness_auc", "Value": x})
            gen_stat_comp[c_comp]["Wrist Smoothness AUC"] = np.array(wj_gen)

    # Compute distances ONLY if Generated data is available
    stat_distances_df = None
    stat_metrics_logged = {}
    if has_gen_data:
        comparator = DistributionComparator()
        stat_results = comparator.compare(gt_stat_comp, gen_stat_comp)
        stat_distances_df = comparator._format_results_to_dataframe(stat_results)
        stat_metrics_logged = {
            "eval_stationary/Mean_Norm_Wasserstein": float(stat_distances_df["Norm_Wasserstein"].mean()),
            "eval_stationary/Mean_KS": float(stat_distances_df["KS_Stat"].mean())
        }
    else:
        print("  [Stationary Track] Generated stationary data is empty (e.g. initial baseline). Omitting distance calculation.")

    plot_stationary_sequence_eval_metrics(stationary_records, stat_distances_df, vis_out_dir)

    return stat_metrics_logged
