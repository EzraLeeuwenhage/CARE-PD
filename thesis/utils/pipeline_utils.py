import numpy as np
import json
from pathlib import Path
import yaml
import torch
from collections import defaultdict
from smplx.lbs import vertices2joints
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from thesis.utils.legacy_6d.sixD2smpl import build_smpl_pkl_from_6d_smpl, convert_6d_to_smpl
from thesis.utils.threeD2smpl import build_smpl_pkl_from_3d_smpl
from thesis.src.care_pd.smpl2h36m import convert_smpl_to_h36m
from thesis.src.evaluate_h36m import H36MEvaluator
from thesis.src.evaluate_smpl import SMPLEvaluator
from thesis.src.evaluate_distributions import DistributionComparator

from thesis.utils.visualize_h36m_metric_dist import (
    plot_dataset_summary_stats,
    plot_pd_feature_violins,
    plot_pd_feature_comparison_plots,
    prepare_dataframe,
    prepare_combined_dataframe
)
from thesis.utils.visualize_smpl_metric_dist import (
    plot_smpl_mpjae,
    plot_arm_swing_metrics,
    plot_sparc_metrics
)

def validate_config(cfg):
    valid_reps = ["3D", "6D"]
    rep = cfg['data'].get('representation')
    assert rep in valid_reps, f"[Config Error] data.representation must be one of {valid_reps}, got {rep}"
    
    valid_seq_types = ["fixed", "masked"]
    seq_type = cfg['windowing'].get('sequence_type', 'fixed')
    assert seq_type in valid_seq_types, f"[Config Error] windowing.sequence_type must be {valid_seq_types}"
    print(">>> Config Validation Passed.")

def load_config(CONFIG_PATH="thesis/configs/baseline.yaml"):
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    model_name = cfg['model']['name']
    cfg['paths']['output_dir'] = cfg['paths']['output_dir'].format(model_name=model_name)
    return cfg

def import_pipeline_components(representation):
    if representation == '6D':
        print(f"\n[ROUTER] Initializing 6D Pipeline...")
        from thesis.src.legacy_6d.model import ConditionalBaselineModel, JointBaselineModel
        from thesis.src.legacy_6d.dataloader import get_dataloader
        from thesis.src.legacy_6d.callbacks import EpochAndValPrintCallback, WandBEvaluationCallback
        from thesis.src.legacy_6d.sample import generate_trajectories
    elif representation == '3D':
        print(f"\n[ROUTER] Initializing 3D Pipeline...")
        from thesis.src.model import ConditionalBaselineModel, JointBaselineModel
        from thesis.src.dataloader import get_dataloader
        from thesis.src.callbacks import EpochAndValPrintCallback, WandBEvaluationCallback
        from thesis.src.sample import generate_trajectories
    else:
        raise ValueError(f"Unknown representation '{representation}'")

    return (ConditionalBaselineModel, JointBaselineModel, get_dataloader, 
            EpochAndValPrintCallback, WandBEvaluationCallback, generate_trajectories)

def format_and_convert(data_dict, cfg, rep='3D', is_joint_model=False):
    out_dir = Path(cfg['paths']['output_dir'])

    smpl_dir = out_dir / "SMPL"
    h36m_dir = out_dir / "h36m"
    rep_dir = out_dir / f"{rep}_SMPL"
    
    for d in [smpl_dir, h36m_dir, rep_dir]: d.mkdir(parents=True, exist_ok=True)
    
    gt_pkl, gen_pkl = smpl_dir / "ground_truth.pkl", smpl_dir / "generated.pkl"
    gt_h36m, gen_h36m = h36m_dir / "ground_truth_3d_world.npz", h36m_dir / "generated_3d_world.npz"
    gt_npz, gen_npz = rep_dir / f"ground_truth_{rep.lower()}.npz", rep_dir / f"generated_{rep.lower()}.npz"

    gt_dict, gen_dict = {}, {}
    gt_labels, gen_labels = {"key_to_severity": {}}, {"key_to_severity": {}}
    
    gen_severities_list = data_dict["gen_severities"] if is_joint_model else data_dict["severities"]

    print(f"Formatting and caching raw {rep} sequences...")
    for i, gt_sev in enumerate(data_dict["severities"]):
        seq_key = f"seq_{i:03d}"

        # Handles both 6D and 3D gracefully
        gt_dict[seq_key] = data_dict["gt"]["pose"][i].cpu().numpy() if torch.is_tensor(data_dict["gt"]["pose"]) else data_dict["gt"]["pose"][i]
        gt_dict[f"{seq_key}_trans"] = data_dict["gt"]["trans"][i].cpu().numpy() if torch.is_tensor(data_dict["gt"]["trans"]) else data_dict["gt"]["trans"][i]
        
        gen_dict[seq_key] = data_dict["gen"]["pose"][i].cpu().numpy() if torch.is_tensor(data_dict["gen"]["pose"]) else data_dict["gen"]["pose"][i]
        gen_dict[f"{seq_key}_trans"] = data_dict["gen"]["trans"][i].cpu().numpy() if torch.is_tensor(data_dict["gen"]["trans"]) else data_dict["gen"]["trans"][i]
        
        gen_sev = gen_severities_list[i]
        
        # Registry mapping for SMPLEvaluator (Matches Seq keys)
        gt_labels["key_to_severity"][seq_key] = gt_sev
        gen_labels["key_to_severity"][seq_key] = gen_sev
        
        # Registry mapping for H36MEvaluator (Matches converted .pkl keys)
        gt_labels["key_to_severity"][f"GT__gt_{i:03d}"] = gt_sev
        gen_labels["key_to_severity"][f"GEN__gen_{i:03d}"] = gen_sev

    np.savez(gt_npz, **gt_dict)
    np.savez(gen_npz, **gen_dict)
    
    if gt_h36m.exists() and gt_pkl.exists():
        print("Ground Truth H36M data already exists.")
    else:
        print("Formatting Ground Truth to SMPL...")
        if rep == '6D': build_smpl_pkl_from_6d_smpl(data_dict["gt"]["pose"], data_dict["gt"]["trans"], str(gt_pkl), "GT", "gt")
        else: build_smpl_pkl_from_3d_smpl(data_dict["gt"]["pose"], data_dict["gt"]["trans"], str(gt_pkl), "GT", "gt")
        
        print("Converting Ground Truth SMPL -> H36M (This takes a moment)...")
        convert_smpl_to_h36m(str(gt_pkl), str(gt_h36m.parent), gt_h36m.name)
    
    print("Formatting Generated data to SMPL...")
    if rep == '6D': build_smpl_pkl_from_6d_smpl(data_dict["gen"]["pose"], data_dict["gen"]["trans"], str(gen_pkl), "GEN", "gen")
    else: build_smpl_pkl_from_3d_smpl(data_dict["gen"]["pose"], data_dict["gen"]["trans"], str(gen_pkl), "GEN", "gen")
        
    print("Converting Generated SMPL -> H36M...")
    convert_smpl_to_h36m(str(gen_pkl), str(gen_h36m.parent), gen_h36m.name)
        
    gt_labels_path, gen_labels_path = h36m_dir / "gt_labels.json", h36m_dir / "gen_labels.json"
    with open(gt_labels_path, 'w') as f: json.dump(gt_labels, f)
    with open(gen_labels_path, 'w') as f: json.dump(gen_labels, f)
        
    return {
        "gt_data": gt_npz, "gen_data": gen_npz, "gt_h36m": gt_h36m,
        "gen_h36m": gen_h36m, "gt_labels": gt_labels_path, "gen_labels": gen_labels_path, 
        "out_dir": out_dir, "prior_severities": data_dict.get("prior_severities", [])
    }

def evaluate_pipeline(paths, is_joint_model=False, min_z_travel=0.5):
    evaluator = H36MEvaluator(fps=30, min_z_travel=min_z_travel)
    gt_h36m_data = evaluator.evaluate_and_cache(str(paths["gt_h36m"]), str(paths["gt_labels"]), str(paths["out_dir"] / "evaluation" / "gt_h36m_distributions.pkl"))
    gen_h36m_data = evaluator.evaluate_and_cache(str(paths["gen_h36m"]), str(paths["gen_labels"]), str(paths["out_dir"] / "evaluation" / "gen_h36m_distributions.pkl"), synthetic=True)

    smpl_evaluator = SMPLEvaluator()
    smpl_eval_path = paths["out_dir"] / "evaluation" / "smpl_mpjae_evaluation.json"
    smpl_evaluator.evaluate_and_cache(
        gt_npz_path=paths["gt_data"], gen_npz_path=paths["gen_data"],
        labels_path=paths["gen_labels"], cache_output_path=str(smpl_eval_path)
    )

    print("\nGenerating and saving Final Test Set visualizations...")
    vis_out_dir = Path(f"thesis/visualizations/{paths['out_dir'].name}")
    vis_out_dir.mkdir(parents=True, exist_ok=True)
    
    comparator = DistributionComparator()
    h36m_results = comparator.compare(gt_h36m_data, gen_h36m_data)
    h36m_dist_df = comparator._format_results_to_dataframe(h36m_results)

    gt_df = prepare_dataframe(gt_h36m_data)
    gen_df = prepare_dataframe(gen_h36m_data)
    combined_df = prepare_combined_dataframe(gt_h36m_data, gen_h36m_data)
    
    plot_dataset_summary_stats(gt_df, vis_out_dir, prefix="gt_", dataset_label="Ground Truth Baseline")
    plot_pd_feature_violins(gt_df, vis_out_dir, prefix="gt_", dataset_label="Ground Truth Baseline")
    plot_dataset_summary_stats(gen_df, vis_out_dir, prefix="gen_", dataset_label="Generated Test Set")
    plot_pd_feature_violins(gen_df, vis_out_dir, prefix="gen_", dataset_label="Generated Test Set")
    plot_pd_feature_comparison_plots(combined_df, h36m_dist_df, vis_out_dir)

    if is_joint_model:
        with open(paths["gt_labels"], 'r') as f: gt_lbls = json.load(f)["key_to_severity"]
        with open(paths["gen_labels"], 'r') as f: gen_lbls = json.load(f)["key_to_severity"]
        y_true = [v for k, v in gt_lbls.items() if k.startswith("seq_")]
        y_pred = [v for k, v in gen_lbls.items() if k.startswith("seq_")]
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]), annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
        plt.title('Joint Model Label Confusion Matrix'); plt.tight_layout()
        plt.savefig(vis_out_dir / "label_confusion_matrix.png", dpi=300); plt.close()

        prior_sevs = paths.get("prior_severities", None)
        if prior_sevs:
            plt.figure(figsize=(6, 5))
            sns.heatmap(confusion_matrix(prior_sevs, y_pred, labels=[0, 1, 2, 3]), annot=True, fmt='d', cmap='Oranges', xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
            plt.title('Prior State vs Predicted Label Correlation'); plt.tight_layout()
            plt.savefig(vis_out_dir / "prior_state_correlation_matrix.png", dpi=300); plt.close()
    
    with open(smpl_eval_path, 'r') as f: smpl_json = json.load(f)
    gt_comp, gen_comp = defaultdict(dict), defaultdict(dict)
    target_sparc_joints = ['L_Knee', 'R_Knee']
    
    for sev_key, metrics in smpl_json.get("raw_distributions", {}).items():
        c_key = "overall" if sev_key == "Overall" else sev_key.replace("Class ", "")
        gt_comp[c_key]["Swing Asymmetry (SI)"] = np.array(metrics.get("GT_Symmetry_Index", []))
        gen_comp[c_key]["Swing Asymmetry (SI)"] = np.array(metrics.get("Gen_Symmetry_Index", []))
        
        gt_knees, gen_knees = [], []
        for j in target_sparc_joints:
            gt_knees.extend(metrics.get(f"GT_SPARC_{j}", []))
            gen_knees.extend(metrics.get(f"Gen_SPARC_{j}", []))
        gt_comp[c_key]["SPARC_Knees"] = np.array(gt_knees)
        gen_comp[c_key]["SPARC_Knees"] = np.array(gen_knees)
        
    smpl_dist_df = comparator._format_results_to_dataframe(comparator.compare(gt_comp, gen_comp))
    plot_smpl_mpjae(smpl_json, vis_out_dir)
    plot_arm_swing_metrics(smpl_json, vis_out_dir, distances_df=smpl_dist_df)
    plot_sparc_metrics(smpl_json, vis_out_dir, distances_df=smpl_dist_df)
    print(f"Final Test visual artifacts saved permanently to: {vis_out_dir}")

# 6D to H36M conversion
def forward_6d_to_h36m(pose_6d, trans, smpl_model, h36m_regressor, device):
    smpl_pose = convert_6d_to_smpl(pose_6d) 
    T = smpl_pose.shape[0]
    
    global_orient = torch.as_tensor(smpl_pose[:, 0:1, :], dtype=torch.float32, device=device).reshape(T, -1)
    body_pose     = torch.as_tensor(smpl_pose[:, 1:24, :], dtype=torch.float32, device=device).reshape(T, -1)
    world_trans_t = torch.as_tensor(trans, dtype=torch.float32, device=device)
    
    betas, zero_pose, zero_hand = torch.zeros((T, 10)).to(device), torch.zeros((T, 3)).to(device), torch.zeros((T, 15, 3)).to(device)

    with torch.no_grad():
        out = smpl_model(betas=betas, body_pose=body_pose, global_orient=global_orient, jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose, left_hand_pose=zero_hand, right_hand_pose=zero_hand, expression=betas)
        vertices_world = out.vertices + world_trans_t[:, None, :]
        h36m_joints = vertices2joints(h36m_regressor, vertices_world)
    return h36m_joints.cpu().numpy()

def batched_6d_to_h36m(pose_6d, trans, smpl_model, h36m_regressor, device, chunk_size=256):
    N, T, J, _ = pose_6d.shape
    out_h36m = np.zeros((N, T, 17, 3), dtype=np.float32)
    for i in range(0, N, chunk_size):
        p_chunk, t_chunk = pose_6d[i:i+chunk_size].to(device), trans[i:i+chunk_size].to(device)
        B = p_chunk.shape[0]
        h36m_flat = forward_6d_to_h36m(p_chunk.reshape(B * T, J, 6), t_chunk.reshape(B * T, 3), smpl_model, h36m_regressor, device)
        out_h36m[i:i+chunk_size] = h36m_flat.reshape(B, T, 17, 3)
    return out_h36m

# 3D to H36M conversion
def forward_3d_to_h36m(pose_3d, trans, smpl_model, h36m_regressor, device):
    T = pose_3d.shape[0]
    
    global_orient = torch.as_tensor(pose_3d[:, 0:1, :], dtype=torch.float32, device=device).reshape(T, -1)
    body_pose     = torch.as_tensor(pose_3d[:, 1:24, :], dtype=torch.float32, device=device).reshape(T, -1)
    world_trans_t = torch.as_tensor(trans, dtype=torch.float32, device=device)
    
    betas, zero_pose, zero_hand = torch.zeros((T, 10)).to(device), torch.zeros((T, 3)).to(device), torch.zeros((T, 15, 3)).to(device)

    with torch.no_grad():
        out = smpl_model(betas=betas, body_pose=body_pose, global_orient=global_orient, jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose, left_hand_pose=zero_hand, right_hand_pose=zero_hand, expression=betas)
        vertices_world = out.vertices + world_trans_t[:, None, :]
        h36m_joints = vertices2joints(h36m_regressor, vertices_world)
    return h36m_joints.cpu().numpy()

def batched_3d_to_h36m(pose_3d, trans, smpl_model, h36m_regressor, device, chunk_size=256):
    N, T, J, _ = pose_3d.shape
    out_h36m = np.zeros((N, T, 17, 3), dtype=np.float32)
    for i in range(0, N, chunk_size):
        p_chunk, t_chunk = pose_3d[i:i+chunk_size], trans[i:i+chunk_size]
        B = p_chunk.shape[0]
        h36m_flat = forward_3d_to_h36m(p_chunk.reshape(B * T, J, 3), t_chunk.reshape(B * T, 3), smpl_model, h36m_regressor, device)
        out_h36m[i:i+chunk_size] = h36m_flat.reshape(B, T, 17, 3)
    return out_h36m