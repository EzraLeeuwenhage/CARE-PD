import numpy as np
import json
from pathlib import Path
import yaml
import torch
from collections import defaultdict
from smplx.lbs import vertices2joints

from thesis.utils.sixD2smpl import build_smpl_pkl_from_6d_smpl, convert_6d_to_smpl
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


def load_config(CONFIG_PATH="thesis/configs/baseline.yaml"):
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)

    model_name = cfg['model']['name']
    cfg['paths']['output_dir'] = cfg['paths']['output_dir'].format(model_name=model_name)
    return cfg


def format_and_convert(data_dict, cfg, is_joint_model=False):
    out_dir = Path(cfg['paths']['output_dir'])

    smpl_dir = out_dir / "SMPL"
    h36m_dir = out_dir / "h36m"
    sixd_dir = out_dir / "6D_SMPL"
    
    smpl_dir.mkdir(parents=True, exist_ok=True)
    h36m_dir.mkdir(parents=True, exist_ok=True)
    sixd_dir.mkdir(parents=True, exist_ok=True)
    
    gt_pkl = smpl_dir / "ground_truth.pkl"
    gen_pkl = smpl_dir / "generated.pkl"
    
    gt_h36m = h36m_dir / "ground_truth_3d_world.npz"
    gen_h36m = h36m_dir / "generated_3d_world.npz"
    
    gt_6d_npz = sixd_dir / "ground_truth_6d.npz"
    gen_6d_npz = sixd_dir / "generated_6d.npz"

    gt_dict, gen_dict = {}, {}
    gt_labels, gen_labels = {"key_to_severity": {}}, {"key_to_severity": {}}
    
    gen_severities_list = data_dict["gen_severities"] if is_joint_model else data_dict["severities"]

    print("Formatting and caching raw 6D sequences...")
    for i, gt_sev in enumerate(data_dict["severities"]):
        seq_key = f"seq_{i:03d}"

        # Save 6D SMPL sequences to NPZ files for both ground truth and generated data
        gt_dict[seq_key] = data_dict["gt"]["pose"][i].numpy()
        gt_dict[f"{seq_key}_trans"] = data_dict["gt"]["trans"][i].numpy()
        gen_dict[seq_key] = data_dict["gen"]["pose"][i].numpy()
        gen_dict[f"{seq_key}_trans"] = data_dict["gen"]["trans"][i].numpy()
        
        gen_sev = gen_severities_list[i]
        
        # Registry mapping for SMPLEvaluator (Matches 6D Seq keys)
        gt_labels["key_to_severity"][seq_key] = gt_sev
        gen_labels["key_to_severity"][seq_key] = gen_sev
        
        # Registry mapping for H36MEvaluator (Matches converted .pkl keys)
        gt_labels["key_to_severity"][f"GT__gt_{i:03d}"] = gt_sev
        gen_labels["key_to_severity"][f"GEN__gen_{i:03d}"] = gen_sev

    np.savez(gt_6d_npz, **gt_dict)
    np.savez(gen_6d_npz, **gen_dict)
    
    if gt_h36m.exists() and gt_pkl.exists():
        print("Ground Truth H36M data already exists.")
    else:
        print("Formatting Ground Truth to SMPL...")
        build_smpl_pkl_from_6d_smpl(data_dict["gt"]["pose"], data_dict["gt"]["trans"], str(gt_pkl), "GT", "gt")
        print("Converting Ground Truth SMPL -> H36M (This takes a moment)...")
        convert_smpl_to_h36m(str(gt_pkl), str(gt_h36m.parent), gt_h36m.name)
    
    print("Formatting Generated data to SMPL...")
    build_smpl_pkl_from_6d_smpl(data_dict["gen"]["pose"], data_dict["gen"]["trans"], str(gen_pkl), "GEN", "gen")
    print("Converting Generated SMPL -> H36M...")
    convert_smpl_to_h36m(str(gen_pkl), str(gen_h36m.parent), gen_h36m.name)
        
    gt_labels_path = h36m_dir / "gt_labels.json"
    gen_labels_path = h36m_dir / "gen_labels.json"
    
    with open(gt_labels_path, 'w') as f: json.dump(gt_labels, f)
    with open(gen_labels_path, 'w') as f: json.dump(gen_labels, f)
        
    return {
        "gt_6d": gt_6d_npz, "gen_6d": gen_6d_npz, "gt_h36m": gt_h36m,
        "gen_h36m": gen_h36m, "gt_labels": gt_labels_path,
        "gen_labels": gen_labels_path, "out_dir": out_dir
    }


def evaluate_pipeline(paths):
    evaluator = H36MEvaluator(fps=30)
    gt_h36m_data = evaluator.evaluate_and_cache(
        npz_path=str(paths["gt_h36m"]),
        labels_path=str(paths["gt_labels"]),
        cache_output_path=str(paths["out_dir"] / "evaluation" / "gt_h36m_distributions.pkl")
    )
    gen_h36m_data = evaluator.evaluate_and_cache(
        npz_path=str(paths["gen_h36m"]),
        labels_path=str(paths["gen_labels"]),
        cache_output_path=str(paths["out_dir"] / "evaluation" / "gen_h36m_distributions.pkl"),
        synthetic=True
    )

    smpl_evaluator = SMPLEvaluator()
    smpl_eval_path = paths["out_dir"] / "evaluation" / "smpl_mpjae_evaluation.json"
    smpl_evaluator.evaluate_and_cache(
        gt_npz_path=paths["gt_6d"],
        gen_npz_path=paths["gen_6d"],
        labels_path=paths["gen_labels"],
        cache_output_path=str(smpl_eval_path),
        verbose=True
    )

    print("\nGenerating and saving Final Test Set visualizations...")
    model_folder = paths["out_dir"].name
    vis_out_dir = Path(f"thesis/visualizations/{model_folder}")
    vis_out_dir.mkdir(parents=True, exist_ok=True)
    
    comparator = DistributionComparator()
    
    # H36M Distribution Plots
    h36m_results = comparator.compare(gt_h36m_data, gen_h36m_data)
    h36m_dist_df = comparator._format_results_to_dataframe(h36m_results)
    
    gen_df = prepare_dataframe(gen_h36m_data)
    combined_df = prepare_combined_dataframe(gt_h36m_data, gen_h36m_data)
    
    plot_dataset_summary_stats(gen_df, vis_out_dir, prefix="gen_", dataset_label="Final Test Set")
    plot_pd_feature_violins(gen_df, vis_out_dir, prefix="gen_", dataset_label="Final Test Set")
    plot_pd_feature_comparison_plots(combined_df, h36m_dist_df, vis_out_dir)
    
    # SMPL Distribution Plots
    with open(smpl_eval_path, 'r') as f:
        smpl_json = json.load(f)
        
    gt_comp, gen_comp = defaultdict(dict), defaultdict(dict)
    target_sparc_joints = ['L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle']
    
    for sev_key, metrics in smpl_json.get("raw_distributions", {}).items():
        c_key = "overall" if sev_key == "Overall" else sev_key.replace("Class ", "")
        
        gt_comp[c_key]["Swing Asymmetry (SI)"] = np.array(metrics.get("GT_Symmetry_Index", []))
        gen_comp[c_key]["Swing Asymmetry (SI)"] = np.array(metrics.get("Gen_Symmetry_Index", []))
        
        gt_legs, gen_legs = [], []
        for j in target_sparc_joints:
            gt_legs.extend(metrics.get(f"GT_SPARC_{j}", []))
            gen_legs.extend(metrics.get(f"Gen_SPARC_{j}", []))
        gt_comp[c_key]["SPARC_Lower_Limbs"] = np.array(gt_legs)
        gen_comp[c_key]["SPARC_Lower_Limbs"] = np.array(gen_legs)
        
    smpl_dist_df = comparator._format_results_to_dataframe(comparator.compare(gt_comp, gen_comp))
    
    # Pass the loaded dictionary instead of a path to plot directly
    plot_smpl_mpjae(smpl_json, vis_out_dir)
    plot_arm_swing_metrics(smpl_json, vis_out_dir, distances_df=smpl_dist_df)
    plot_sparc_metrics(smpl_json, vis_out_dir, distances_df=smpl_dist_df)
    
    print(f"Final Test visual artifacts saved permanently to: {vis_out_dir}")


def forward_6d_to_h36m(pose_6d, trans, smpl_model, h36m_regressor, device):
    """
    Directly converts a single sequence of 6D poses and translations to 3D H36M coordinates.
    Executes entirely in memory without intermediate files.
    
    Args:
        pose_6d: Tensor of shape (T, 24, 6)
        trans: Tensor of shape (T, 3)
    Returns:
        np.ndarray of shape (T, 17, 3) representing 3D H36M joints
    """
    # 1. Convert 6D back to standard SMPL Axis-Angle
    smpl_pose = convert_6d_to_smpl(pose_6d)  # Returns numpy array (T, 24, 3)
    
    T = smpl_pose.shape[0]
    
    # 2. Extract global orientation and body pose, format for SMPL layer
    global_orient = torch.tensor(smpl_pose[:, 0:1, :], dtype=torch.float32).reshape(T, -1).to(device)
    body_pose     = torch.tensor(smpl_pose[:, 1:24, :], dtype=torch.float32).reshape(T, -1).to(device)
    world_trans_t = trans.clone().detach().to(dtype=torch.float32, device=device)
    
    # 3. Create neutral shape/expression placeholders
    betas = torch.zeros((T, 10), dtype=torch.float32).to(device)
    zero_pose = torch.zeros((T, 3), dtype=torch.float32).to(device)
    zero_hand = torch.zeros((T, 15, 3), dtype=torch.float32).to(device)

    # 4. Forward pass through the SMPL body model
    with torch.no_grad():
        out = smpl_model(betas=betas, body_pose=body_pose, global_orient=global_orient,
                         jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose,
                         left_hand_pose=zero_hand, right_hand_pose=zero_hand,
                         expression=betas)
        
        # Apply global translation to the generated vertices
        vertices_world = out.vertices + world_trans_t[:, None, :]
        
        # 5. Regress H36M 3D joint locations from the vertices
        h36m_joints = vertices2joints(h36m_regressor, vertices_world)
        
    return h36m_joints.cpu().numpy()