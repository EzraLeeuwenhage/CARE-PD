# Visualizing sequences

python -m thesis.src.care_pd.viz_seqs -n "thesis/data/raw/PD-GaM/h36m/h36m_3d_world_floorXZZplus_30f_or_longer.npz" -f h36m

python -m thesis.src.care_pd.viz_seqs -n thesis/data/processed/baseline_model/h36m/ground_truth_3d_world.npz -f h36m -l thesis\data\processed\baseline_model\h36m\gt_labels.json -k GT__gt_18539

python -m thesis.src.care_pd.viz_seqs --auto ConditionalModel-MLP-Baseline/h36m -f h36m --use-lbl -k GT__gt_001