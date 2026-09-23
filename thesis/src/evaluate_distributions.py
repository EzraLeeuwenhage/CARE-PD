import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ks_2samp, wasserstein_distance
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm


class DistributionComparator:
    def __init__(self, metrics_to_compare=None):
        """
        Args:
            metrics_to_compare (list): List of string metric names to compare. 
                                       If None, computes for all available metrics.
        """
        self.metrics = metrics_to_compare

    def compute_wasserstein(self, p_samples, q_samples):
        """Computes the 1-Wasserstein distance."""
        p_samples = p_samples[~np.isnan(p_samples)]
        q_samples = q_samples[~np.isnan(q_samples)]

        if len(p_samples) == 0 or len(q_samples) == 0:
            return np.nan

        return float(wasserstein_distance(p_samples, q_samples))

    def compute_standardized_wasserstein(self, p_samples, q_samples):
        """Computes 1-Wasserstein normalized by ground-truth standard deviation (z-score distance)."""
        p_clean = p_samples[~np.isnan(p_samples)]
        q_clean = q_samples[~np.isnan(q_samples)]

        if len(p_clean) == 0 or len(q_clean) == 0:
            return np.nan

        w_dist = float(wasserstein_distance(p_clean, q_clean))
        std_gt = float(np.std(p_clean))
        return w_dist / (std_gt + 1e-6) if std_gt > 1e-6 else 0.0

    def compute_ks(self, p_samples, q_samples):
        """Computes the Kolmogorov-Smirnov statistic."""
        p_samples = p_samples[~np.isnan(p_samples)]
        q_samples = q_samples[~np.isnan(q_samples)]
        
        if len(p_samples) == 0 or len(q_samples) == 0:
            return np.nan
            
        stat, _ = ks_2samp(p_samples, q_samples)
        return stat

    def compare(self, gt_data, gen_data):
        """Runs full comparison between ground truth and generated data dicts."""
        results = {}
        common_severities = [k for k in gt_data.keys() if k in gen_data.keys()]

        for sev in common_severities:
            results[sev] = {}
            metrics_to_eval = self.metrics if self.metrics else gt_data[sev].keys()

            for metric in metrics_to_eval:
                if metric not in gt_data[sev] or metric not in gen_data[sev]:
                    continue

                gt_samples = gt_data[sev][metric]
                gen_samples = gen_data[sev][metric]

                results[sev][metric] = {
                    "KS_Stat": self.compute_ks(gt_samples, gen_samples),
                    "Wasserstein": self.compute_wasserstein(gt_samples, gen_samples),
                    "Norm_Wasserstein": self.compute_standardized_wasserstein(gt_samples, gen_samples)
                }
                
        return results

    # ----------------------------------------
    # HEATMAP FOR DISTANCES
    # ----------------------------------------
    def _format_results_to_dataframe(self, results):
        rows = []
        keys = ["overall"] + sorted([k for k in results.keys() if k != "overall"])
        for sev in keys:
            for metric, distances in results[sev].items():
                rows.append({
                    "Severity": "Overall" if sev == "overall" else f"Class {sev}",
                    "Metric": metric,
                    "KS_Stat": distances["KS_Stat"],
                    "Wasserstein": distances["Wasserstein"],
                    "Norm_Wasserstein": distances["Norm_Wasserstein"]
                })
        return pd.DataFrame(rows)
        
    def plot_distance_heatmaps(self, results, save_dir=None):
        """Renders a dual heatmap for KS and Wasserstein distances using semantic coloring."""
        # Prepare data for heatmap visuals
        results = self._format_results_to_dataframe(results) if isinstance(results, dict) else results
        ks_pivot = results.pivot(index='Metric', columns='Severity', values='KS_Stat')
        w_pivot = results.pivot(index='Metric', columns='Severity', values='Wasserstein')

        # Sort columns on severity class
        cols = ["Overall"] + [f"Class {i}" for i in range(4) if f"Class {i}" in ks_pivot.columns]
        ks_pivot = ks_pivot[cols]
        w_pivot = w_pivot[cols]

        # Define colormap for KS overview
        colors = ['#85e085', '#ffe680', '#ffb366', '#ff6666'] 
        cmap_ks = ListedColormap(colors)
        bounds_ks = [0.0, 0.10, 0.20, 0.40, 1.0]
        norm_ks = BoundaryNorm(bounds_ks, cmap_ks.N)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        sns.heatmap(ks_pivot, annot=True, fmt=".2f", cmap=cmap_ks, norm=norm_ks, ax=axes[0], 
                    cbar=False, linewidths=1, linecolor='white')
        axes[0].set_title("Kolmogorov-Smirnov (KS) Statistic", fontsize=14, fontweight='bold')
        axes[0].set_ylabel("")
        axes[0].set_xlabel("")

        sns.heatmap(w_pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=axes[1], 
                    cbar_kws={'label': '1-Wasserstein Distance'}, linewidths=1, linecolor='white')
        axes[1].set_title("1-Wasserstein Distance (Physical Feature Units)", fontsize=14, fontweight='bold')
        axes[1].set_ylabel("")
        axes[1].set_xlabel("")

        plt.suptitle("Distribution Distances: Semantic Evaluation", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_dir:
            out_path = Path(save_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path / "03_distance_heatmaps.png", dpi=300)
            print(f"Saved distance heatmap to {out_path / '03_distance_heatmaps.png'}")
        else:
            plt.show()
        plt.close()

    # ----------------------------------------
    # VISUALIZATIONS OF COMPUTATIONS
    # ----------------------------------------

    def plot_ks_ecdf(self, p_samples, q_samples, metric_name, sev_class, save_dir=None):
        """Visualizes the Empirical CDFs and the maximum vertical gap (KS Statistic)."""
        p_samples = p_samples[~np.isnan(p_samples)]
        q_samples = q_samples[~np.isnan(q_samples)]

        # Compute empirical CDF values
        x_p, y_p = np.sort(p_samples), np.arange(1, len(p_samples) + 1) / len(p_samples)
        x_q, y_q = np.sort(q_samples), np.arange(1, len(q_samples) + 1) / len(q_samples)

        # Evaluate both ECDFs across all unique points to find the max gap
        x_all = np.sort(np.concatenate([x_p, x_q]))
        cdf_p = np.searchsorted(x_p, x_all, side='right') / len(x_p)
        cdf_q = np.searchsorted(x_q, x_all, side='right') / len(x_q)
        
        gaps = np.abs(cdf_p - cdf_q)
        max_idx = np.argmax(gaps)
        ks_stat = gaps[max_idx]
        ks_x = x_all[max_idx]

        plt.figure(figsize=(8, 5))
        plt.step(x_p, y_p, label='Ground Truth (GT)', where='post', color='cornflowerblue', linewidth=2)
        plt.step(x_q, y_q, label='Generated (Gen)', where='post', color='salmon', linewidth=2)
        
        # get maximum gap
        plt.plot([ks_x, ks_x], [cdf_p[max_idx], cdf_q[max_idx]], color='red', linestyle='--', linewidth=2, 
                 label=f'KS Statistic (Max Gap): {ks_stat:.4f}')

        # Visually show the tail limitation if severe 
        max_gt = np.max(x_p)
        if np.max(x_q) > max_gt * 1.5:
            plt.axvspan(max_gt, np.max(x_q), color='grey', alpha=0.15, label='Tail Blindspot (Ignored by KS)')

        plt.title(f'KS Statistic (ECDF Max Gap): {metric_name}\nClass: {sev_class}', fontweight='bold')
        plt.xlabel(metric_name)
        plt.ylabel('Cumulative Probability')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_dir:
            out_path = Path(save_dir) / "KS_ECDF"
            out_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path / f"ks_{metric_name}_cls_{sev_class}.png", dpi=300)
        else:
            plt.show()
        plt.close()

    def plot_wasserstein_transport(self, p_samples, q_samples, metric_name, sev_class, save_dir=None):
        """Visualizes the 1-Wasserstein distance as the integrated area between empirical CDFs."""
        p_samples = p_samples[~np.isnan(p_samples)]
        q_samples = q_samples[~np.isnan(q_samples)]

        if len(p_samples) == 0 or len(q_samples) == 0:
            return

        x_p = np.sort(p_samples)
        x_q = np.sort(q_samples)

        x_all = np.sort(np.unique(np.concatenate([x_p, x_q])))
        cdf_p = np.searchsorted(x_p, x_all, side='right') / len(x_p)
        cdf_q = np.searchsorted(x_q, x_all, side='right') / len(x_q)

        w_dist = float(wasserstein_distance(p_samples, q_samples))

        plt.figure(figsize=(8, 5))
        plt.step(x_all, cdf_p, label='Ground Truth (GT)', where='post', color='cornflowerblue', linewidth=2)
        plt.step(x_all, cdf_q, label='Generated (Gen)', where='post', color='salmon', linewidth=2)

        plt.fill_between(x_all, cdf_p, cdf_q, step='post', color='mediumaquamarine', alpha=0.4,
                         label=f'Transport Work Area (W1={w_dist:.4f})')

        plt.title(f'1-Wasserstein Optimal Transport Area: {metric_name}\nClass: {sev_class}', fontweight='bold')
        plt.xlabel(metric_name)
        plt.ylabel('Cumulative Probability')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_dir:
            out_path = Path(save_dir) / "Wasserstein_Transport"
            out_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path / f"wasserstein_{metric_name}_cls_{sev_class}.png", dpi=300)
        else:
            plt.show()
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and visualize distribution distances.")
    parser.add_argument("--save_computation_plots", action="store_true")
    parser.add_argument("--save_dir", type=str, default="thesis/visualizations/evaluate_distributions_tests")
    parser.add_argument("--classes", type=str, nargs='+', default=["all"])
    args = parser.parse_args()

    gt_path = Path("thesis/data/processed/baseline_model/evaluation/gt_h36m_distributions_new_metrics.pkl")
    gen_path = Path("thesis/data/processed/baseline_model/evaluation/gen_h36m_distributions_new_metrics.pkl")
    
    if not gt_path.exists() or not gen_path.exists():
        print(f"Error: Could not find one of the .pkl files.\nGT: {gt_path}\nGEN: {gen_path}")
        exit()

    with open(gt_path, 'rb') as f:
        gt_data = pickle.load(f)
    with open(gen_path, 'rb') as f:
        gen_data = pickle.load(f)

    pd_features = [
        "mean_step_length", "mean_step_asymmetry", "mean_active_walking_speed", 
        "mean_global_walking_speed", "mean_cadence", "max_ankle_clearance", 
        "mean_emos"
    ]

    print(f"Comparing {len(pd_features)} PD features between GT and Generated Data...")
    
    # Compute metrics
    comparator = DistributionComparator(metrics_to_compare=pd_features)
    results = comparator.compare(gt_data, gen_data)

    eval_distance_dir = Path("thesis/data/processed/baseline_model/evaluation/")
    comparator.plot_distance_heatmaps(results, save_dir=eval_distance_dir)
    
    # Visualizations of computation mechanism
    if args.save_computation_plots:
        target_classes = list(gt_data.keys()) if "all" in args.classes else args.classes
        save_dir = Path(args.save_dir) if args.save_computation_plots else None
        
        print(f"\nGenerating diagnostic plots for classes: {target_classes}")
        for severity in target_classes:
            if severity not in gt_data or severity not in gen_data:
                print(f"Skipping class '{severity}' (not found in data).")
                continue
                
            for metric in pd_features:                    
                gt_samples = gt_data[severity][metric]
                gen_samples = gen_data[severity][metric]

                comparator.plot_ks_ecdf(gt_samples, gen_samples, metric, severity, save_dir)
                comparator.plot_wasserstein_transport(gt_samples, gen_samples, metric, severity, save_dir)
                
        print(f"Saved diagnostic plots to: {save_dir}")