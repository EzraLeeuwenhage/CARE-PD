import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from thesis.src.dataloader import get_dataloader
from thesis.src.utils.geometry_utils import pose_to_rmat
from thesis.src.utils.visualization_utils import plot_sequence_length_distributions

# -------------------------------------------------------------------------
# CONSTANTS & SIGNAL PARAMETERS
# -------------------------------------------------------------------------
FPS = 30
MAX_FREQ = 15.0
SPARC_THRESHOLD = 0.01

def compute_angular_speed(seq_pose: np.ndarray) -> np.ndarray:
    """Converts 3D axis-angle sequence to instantaneous angular speed (a_t) across 24 joints."""
    if isinstance(seq_pose, np.ndarray):
        seq_pose_tensor = torch.tensor(seq_pose, dtype=torch.float32)
    else:
        seq_pose_tensor = seq_pose

    # R_mat shape: (T, 24, 3, 3)
    rot_mats = pose_to_rmat(seq_pose_tensor).cpu().numpy()
    T, J, _, _ = rot_mats.shape
    if T < 2:
        return np.full((1, J), np.nan)

    R_t = rot_mats[:-1]
    R_next = rot_mats[1:]

    # R_rel = R_t^T * R_{t+1} on SO(3)
    R_t_T = np.swapaxes(R_t, -1, -2)
    R_rel = np.matmul(R_t_T, R_next)

    # Lie algebra logarithmic map to tangent vector
    R_rel_flat = R_rel.reshape(-1, 3, 3)
    rotvecs = Rotation.from_matrix(R_rel_flat).as_rotvec().reshape(T - 1, J, 3)

    # Instantaneous angular speed: ||omega_t|| in rad/s
    omega_t = rotvecs * FPS
    return np.linalg.norm(omega_t, axis=-1)


def compute_parameterized_sparc(
    a_sig: np.ndarray, 
    fps: int = 30, 
    padlevel: int = 4, 
    fc_max: float = 15.0, 
    amp_th: float = 0.01, 
    detrend: bool = False,
    fixed_nfft: int = None
):
    """
    Computes SPARC arc length and returns the normalized spectrum.
    If fixed_nfft is provided, it overrides padlevel to ensure standard grid 
    sizes across different sequence lengths (required for averaging spectra).
    """
    # Literature detrending for SPARC: absolute deviation from the mean
    if detrend:
        a_sig = np.abs(a_sig - np.mean(a_sig))
        
    nfft = fixed_nfft if fixed_nfft else int(pow(2, np.ceil(np.log2(len(a_sig))) + padlevel))
    
    A = np.abs(np.fft.rfft(a_sig, n=nfft))
    A_0 = A[0] if A[0] > 0 else 1e-6
    # A_norm = A / A_0
    
    f_vec = np.fft.rfftfreq(nfft, d=1.0 / fps)

    A_norm = (A / A_0) * f_vec  # Scales amplitude by frequency
    
    valid = f_vec <= fc_max
    f_sub = f_vec[valid]
    A_sub = A_norm[valid]
    
    # Adaptive Cutoff
    above_th_idxs = np.where(A_sub >= amp_th)[0]
    if len(above_th_idxs) > 0:
        idx_c = above_th_idxs[-1]
        fc_adj = f_sub[idx_c]
    else:
        idx_c = 0
        fc_adj = f_sub[0]
        
    if fc_adj == 0:
        return 0.0, f_vec, A_norm, fc_adj
        
    f_int = f_sub[:idx_c + 1]
    A_int = A_sub[:idx_c + 1]
    
    # Discrete integral
    dx = np.diff(f_int) / fc_adj
    dy = np.diff(A_int)
    arc_length = -float(np.sum(np.sqrt(dx ** 2 + dy ** 2)))
    
    return arc_length, f_vec, A_norm, fc_adj


# -------------------------------------------------------------------------
# UPDATED Q1: A(0) NORMALIZATION VS. A(0) + DETRENDING NORMALIZATION
# -------------------------------------------------------------------------
# QUESTION ANSWERED:
#   Does removing the static DC offset via Beck et al. detrending (|a(t) - mean|)
#   prevent the DC component from collapsing in stationary gaits, and does it
#   stop the adaptive cutoff (fc) from pegging at 15.0 Hz?
#
# INTERPRETATION GUIDE:
#   - Orig SPARC-Gyro: a(t) = ||omega_t||, normalized by A(0) = sum(a).
#   - Detrended SPARC: a_detrend(t) = |a(t) - mean(a)|, normalized by A_detrend(0) = sum(|a - mean|).
#   - If Detrended pegging drops significantly below the 96-99% baseline,
#     detrending successfully restores the adaptive cutoff mechanism.
def analyze_denominator_explosion(a_t_dict, labels_dict, out_dir):
    print("\n" + "=" * 78)
    print("Q1: A(0) NORMALIZATION VS. A(0) + DETRENDING ANALYSIS")
    print("=" * 78)

    results = []
    knee_idx = 5  # R_Knee

    for key, a_t in a_t_dict.items():
        sev = labels_dict.get(key.split('_down')[0], -1)
        if sev not in [0, 3]:
            continue

        a = a_t[:, knee_idx]
        if len(a) < 2:
            continue

        # Delegate SPARC subroutine calls to parameterized helper
        _, _, _, fc_orig = compute_parameterized_sparc(
            a, fps=FPS, padlevel=4, fc_max=MAX_FREQ, amp_th=SPARC_THRESHOLD, detrend=False
        )
        _, _, _, fc_detrend = compute_parameterized_sparc(
            a, fps=FPS, padlevel=4, fc_max=MAX_FREQ, amp_th=SPARC_THRESHOLD, detrend=True
        )

        results.append({
            "Severity": f"Class {sev}",
            "A0_Orig": np.sum(a),
            "A0_Detrend": np.sum(np.abs(a - np.mean(a))),
            "Cutoff_Orig_Hz": fc_orig,
            "Cutoff_Detrend_Hz": fc_detrend
        })

    df = pd.DataFrame(results)

    # 1. Compare A(0) magnitudes across methods and severity classes
    mean_a0_orig_c0 = df[df["Severity"] == "Class 0"]["A0_Orig"].mean()
    mean_a0_orig_c3 = df[df["Severity"] == "Class 3"]["A0_Orig"].mean()
    mean_a0_detr_c0 = df[df["Severity"] == "Class 0"]["A0_Detrend"].mean()
    mean_a0_detr_c3 = df[df["Severity"] == "Class 3"]["A0_Detrend"].mean()

    print(f"Mean A(0) [Raw Sum]          | Class 0: {mean_a0_orig_c0:8.2f} | Class 3: {mean_a0_orig_c3:8.2f}")
    print(f"Mean A(0) [Detrended MAD]    | Class 0: {mean_a0_detr_c0:8.2f} | Class 3: {mean_a0_detr_c3:8.2f}")

    # 2. Check 15.0 Hz cutoff pegging rate
    pegged_orig_c0 = (df[df["Severity"] == "Class 0"]["Cutoff_Orig_Hz"] >= 14.9).mean() * 100
    pegged_orig_c3 = (df[df["Severity"] == "Class 3"]["Cutoff_Orig_Hz"] >= 14.9).mean() * 100
    pegged_detr_c0 = (df[df["Severity"] == "Class 0"]["Cutoff_Detrend_Hz"] >= 14.9).mean() * 100
    pegged_detr_c3 = (df[df["Severity"] == "Class 3"]["Cutoff_Detrend_Hz"] >= 14.9).mean() * 100

    print(f"\nSequences Pegging at Cutoff Ceiling (15.0 Hz):")
    print(f"  * Class 0 (Orig A(0) Norm)        : {pegged_orig_c0:5.1f}%")
    print(f"  * Class 0 (Detrended A(0) Norm)   : {pegged_detr_c0:5.1f}%")
    print(f"  * Class 3 (Orig A(0) Norm)        : {pegged_orig_c3:5.1f}%")
    print(f"  * Class 3 (Detrended A(0) Norm)   : {pegged_detr_c3:5.1f}%")

    # 3. Generate diagnostic visual distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # A(0) Distributions
    sns.kdeplot(data=df, x="A0_Orig", hue="Severity", fill=True, common_norm=False, ax=axes[0, 0])
    axes[0, 0].set_title("Original A(0) Distribution [Raw Integral]", fontweight='bold')
    axes[0, 0].set_xlabel("A(0) Amplitude")

    sns.kdeplot(data=df, x="A0_Detrend", hue="Severity", fill=True, common_norm=False, ax=axes[0, 1])
    axes[0, 1].set_title("Detrended A(0) Distribution [MAD Integral]", fontweight='bold')
    axes[0, 1].set_xlabel("A(0) Amplitude")

    # Cutoff Frequency Distributions
    sns.kdeplot(data=df, x="Cutoff_Orig_Hz", hue="Severity", fill=True, common_norm=False, ax=axes[1, 0])
    axes[1, 0].axvline(15.0, color='red', linestyle='--', label='15.0 Hz Ceiling')
    axes[1, 0].set_title("Cutoff Frequency: Original SPARC-Gyro", fontweight='bold')
    axes[1, 0].set_xlabel("Cutoff Frequency (Hz)")
    axes[1, 0].legend()

    sns.kdeplot(data=df, x="Cutoff_Detrend_Hz", hue="Severity", fill=True, common_norm=False, ax=axes[1, 1])
    axes[1, 1].axvline(15.0, color='red', linestyle='--', label='15.0 Hz Ceiling')
    axes[1, 1].set_title("Cutoff Frequency: Detrended A(0) SPARC", fontweight='bold')
    axes[1, 1].set_xlabel("Cutoff Frequency (Hz)")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(out_dir / "q1_denominator_explosion.png", dpi=300)
    plt.close()


# -------------------------------------------------------------------------
# Q2: MULTI-BAND SPECTRAL ANALYSIS (PSD vs SPARC vs LINEAR AUC)
# -------------------------------------------------------------------------
# QUESTION ANSWERED:
#   Are healthy walking harmonics distinct from high-frequency pathology/jitter,
#   and can we separate shuffling/FOG from relaxed and higher-frequency gait?
#
# INTERPRETATION GUIDE:
#   - FOG/Shuffling: 0 - 0.7 Hz
#   - Relaxed Gait: 0.7 - 1.5 Hz
#   - Fast Gait: 1.5 - 5.5 Hz
#   - Jitter: 5.5 - 8.0 Hz
def analyze_spectral_bands(
    a_t_dict,
    labels_dict,
    out_dir,
    f_fog: float = 0.7,
    f_rel: float = 1.5,
    f_fast: float = 5.5,
    f_max: float = 8.0,
    sparc_amp_th: float = 0.07
):
    print("\n" + "=" * 125)
    print(f"Q2: 4-BAND SPECTRAL METRICS COMPARISON (MAX FREQ: {f_max:.1f} Hz | THRESHOLD: {sparc_amp_th})")
    print("=" * 125)

    psd_spectra_by_class = {0: [], 1: [], 2: [], 3: []}
    sparc_orig_spectra_by_class = {0: [], 1: [], 2: [], 3: []}
    sparc_detr_spectra_by_class = {0: [], 1: [], 2: [], 3: []}
    band_results = []
    knee_idx = 5

    nfft = 2048
    f = np.fft.rfftfreq(nfft, d=1.0 / FPS)

    # Define the 4 new frequency bands[cite: 11]
    mask_fog = (f >= 0) & (f <= f_fog)
    mask_rel = (f > f_fog) & (f <= f_rel)
    mask_fast = (f > f_rel) & (f <= f_fast)
    mask_jit = (f > f_fast) & (f <= f_max)

    for key, a_t in a_t_dict.items():
        sev = labels_dict.get(key.split('_down')[0], -1)

        a = a_t[:, knee_idx]
        if len(a) < 2:
            continue

        # 1. PSD (Standard zero-mean, squared magnitude)[cite: 11]
        a_psd_detrend = a - np.mean(a)
        A_psd = np.abs(np.fft.rfft(a_psd_detrend, n=nfft))
        PSD = (A_psd ** 2) / nfft
        PSD_norm = PSD / (np.sum(PSD) + 1e-8)
        psd_spectra_by_class[sev].append(PSD_norm)

        psd_fog_pwr = float(np.sum(PSD_norm[mask_fog]))
        psd_rel_pwr = float(np.sum(PSD_norm[mask_rel]))
        psd_fast_pwr = float(np.sum(PSD_norm[mask_fast]))
        psd_jit_pwr = float(np.sum(PSD_norm[mask_jit]))

        # 2. Original SPARC (No detrending, A(0) norm)[cite: 11]
        sparc_orig, _, A_norm_orig, fc_orig = compute_parameterized_sparc(
            a_sig=a, fps=FPS, fc_max=f_max, amp_th=sparc_amp_th, detrend=False, fixed_nfft=nfft
        )
        sparc_orig_spectra_by_class[sev].append(A_norm_orig)

        # 3. Detrended SPARC (|a - mean| MAD norm)[cite: 11]
        sparc_detr, _, A_norm_detr, fc_detr = compute_parameterized_sparc(
            a_sig=a, fps=FPS, fc_max=f_max, amp_th=sparc_amp_th, detrend=True, fixed_nfft=nfft
        )
        sparc_detr_spectra_by_class[sev].append(A_norm_detr)

        # 4. Method 4: Linear Magnitude Band AUC & Jitter Total Variation
        lin_fog_auc = float(np.sum(A_norm_orig[mask_fog]))
        lin_rel_auc = float(np.sum(A_norm_orig[mask_rel]))
        lin_fast_auc = float(np.sum(A_norm_orig[mask_fast]))
        lin_jit_auc = float(np.sum(A_norm_orig[mask_jit]))

        # Total Variation per frequency band
        tv_fog = float(np.sum(np.abs(np.diff(A_norm_orig[mask_fog]))))
        tv_rel = float(np.sum(np.abs(np.diff(A_norm_orig[mask_rel]))))
        tv_fast = float(np.sum(np.abs(np.diff(A_norm_orig[mask_fast]))))
        tv_jit = float(np.sum(np.abs(np.diff(A_norm_orig[mask_jit]))))

        band_results.append({
            "Severity": sev,
            "PSD_FOG": psd_fog_pwr,
            "PSD_Rel": psd_rel_pwr,
            "PSD_Fast": psd_fast_pwr,
            "PSD_Jit": psd_jit_pwr,
            "SPARC_Orig": sparc_orig,
            "SPARC_Detrend": sparc_detr,
            "Cutoff_Orig": fc_orig,
            "Cutoff_Detrend": fc_detr,
            "Lin_FOG": lin_fog_auc,
            "Lin_Rel": lin_rel_auc,
            "Lin_Fast": lin_fast_auc,
            "Lin_Jit": lin_jit_auc,
            "TV_FOG": tv_fog,
            "TV_Rel": tv_rel,
            "TV_Fast": tv_fast,
            "TV_Jit": tv_jit,
        })

    df = pd.DataFrame(band_results)
    grouped = df.groupby("Severity").mean()

    # Terminal Printout for PSD
    print(f" {'Class':<5} | {'PSD FOG':<10} | {'PSD Relaxed':<12} | {'PSD Fast':<10} | {'PSD Jitter':<10}")
    print("-" * 65)
    for sev in [0, 1, 2, 3]:
        if sev in grouped.index:
            print(f"  {sev:<4} | {grouped.loc[sev, 'PSD_FOG']:<10.4f} | {grouped.loc[sev, 'PSD_Rel']:<12.4f} \
                  | {grouped.loc[sev, 'PSD_Fast']:<10.4f} | {grouped.loc[sev, 'PSD_Jit']:<10.4f}")

    # Terminal Printout for Linear AUC & TV
    print(f"\n {'Class':<5} | {'Lin FOG':<10} | {'Lin Relaxed':<12} | {'Lin Fast':<10} | {'Lin Jitter':<10}")
    print("-" * 80)
    for sev in [0, 1, 2, 3]:
        if sev in grouped.index:
            print(f"  {sev:<4} | {grouped.loc[sev, 'Lin_FOG']:<10.2f} | {grouped.loc[sev, 'Lin_Rel']:<12.2f} \
                  | {grouped.loc[sev, 'Lin_Fast']:<10.2f} | {grouped.loc[sev, 'Lin_Jit']:<10.2f}")

    # Terminal Printout for Total Variation
    print(f"\n {'Class':<5} | {'TV FOG':<10} | {'TV Relaxed':<12} | {'TV Fast':<10} | {'TV Jitter':<10}")
    print("-" * 65)
    for sev in [0, 1, 2, 3]:
        if sev in grouped.index:
            print(f"  {sev:<4} | {grouped.loc[sev, 'TV_FOG']:<10.2f} | {grouped.loc[sev, 'TV_Rel']:<12.2f} \
                  | {grouped.loc[sev, 'TV_Fast']:<10.2f} | {grouped.loc[sev, 'TV_Jit']:<10.2f}")

    colors = ['#85e085', '#ffe680', '#ffb366', '#ff6666']

    # Helper function to plot the 4 background spans
    def add_band_spans():
        plt.axvspan(0, f_fog, color='gray', alpha=0.1, label='Shuffling/FOG')
        plt.axvspan(f_fog, f_rel, color='green', alpha=0.1, label='Relaxed Gait')
        plt.axvspan(f_rel, f_fast, color='blue', alpha=0.1, label='Fast Gait')
        plt.axvspan(f_fast, f_max, color='red', alpha=0.1, label='Jitter Band')
        plt.axvline(f_fog, color='black', linestyle=':', alpha=0.5)
        plt.axvline(f_rel, color='black', linestyle=':', alpha=0.5)
        plt.axvline(f_fast, color='black', linestyle=':', alpha=0.5)

    # ---------------------------------------------------------
    # PLOT 1: PSD (Area-based on Squared Magnitude)
    # ---------------------------------------------------------
    plt.figure(figsize=(11, 6))
    for sev in [0, 1, 2, 3]:
        if len(psd_spectra_by_class[sev]) > 0 and sev in grouped.index:
            avg_psd = np.mean(psd_spectra_by_class[sev], axis=0)
            plt.plot(
                f[f <= f_max], 
                avg_psd[f <= f_max], 
                label=f'Class {sev}', 
                color=colors[sev], 
                linewidth=2
            )

    add_band_spans()

    psd_stats_lines = ["Per-Class PSD Area (FOG | Relaxed | Fast | Jitter):"]
    for sev in [0, 1, 2, 3]:
        if sev in grouped.index:
            p_f = grouped.loc[sev, "PSD_FOG"]
            p_r = grouped.loc[sev, "PSD_Rel"]
            p_fa = grouped.loc[sev, "PSD_Fast"]
            p_j = grouped.loc[sev, "PSD_Jit"]
            psd_stats_lines.append(f"  Class {sev}: {p_f:.3f} | {p_r:.3f} | {p_fa:.3f} | {p_j:.3f}")
    
    plt.gca().text(
        0.02, 0.95, "\n".join(psd_stats_lines),
        transform=plt.gca().transAxes,
        fontsize=9, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray')
    )

    plt.title("Area Under Curve: Mean PSD by Severity\n(4-Band Segmentation)", fontweight='bold')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Power / Bin")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    psd_file = out_dir / "q2_PSD_averages_4bands.png"
    plt.savefig(psd_file, dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 2: Original SPARC Spectrum (Arc Length)
    # ---------------------------------------------------------
    plt.figure(figsize=(11, 6))
    for sev in [0, 1, 2, 3]:
        if len(sparc_orig_spectra_by_class[sev]) > 0 and sev in grouped.index:
            avg_orig = np.mean(sparc_orig_spectra_by_class[sev], axis=0)
            mean_fc = grouped.loc[sev, "Cutoff_Orig"]

            plt.plot(
                f[f <= f_max], 
                avg_orig[f <= f_max], 
                label=f'Class {sev}', 
                color=colors[sev], 
                linewidth=2
            )

            fc_idx = np.argmin(np.abs(f - mean_fc))
            plt.plot(
                mean_fc, avg_orig[fc_idx],
                marker='X', markersize=9, color=colors[sev],
                markeredgecolor='black', markeredgewidth=1.2, zorder=5
            )

    add_band_spans()
    plt.axhline(sparc_amp_th, color='purple', linestyle='--', linewidth=1.5, label=f'SPARC Threshold ({sparc_amp_th})')

    orig_stats_lines = ["Per-Class SPARC (Orig) & Cutoff:"]
    for sev in [0, 1, 2, 3]:
        if sev in grouped.index:
            so = grouped.loc[sev, "SPARC_Orig"]
            fc = grouped.loc[sev, "Cutoff_Orig"]
            orig_stats_lines.append(f"  Class {sev}: SPARC={so:.3f} | fc={fc:.2f}Hz")

    plt.gca().text(
        0.02, 0.95, "\n".join(orig_stats_lines),
        transform=plt.gca().transAxes,
        fontsize=9, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray')
    )

    plt.title(f"Original SPARC Spectrum [A(0) Normalization]\n(Threshold: {sparc_amp_th}, 'X' denotes mean Cutoff fc)", fontweight='bold')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Linear Magnitude")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    sparc_orig_file = out_dir / f"q2_SPARC_averages_orig_th{sparc_amp_th}_maxf_{f_max}_4bands.png"
    plt.savefig(sparc_orig_file, dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 3: Detrended SPARC Spectrum (MAD Normalization)
    # ---------------------------------------------------------
    plt.figure(figsize=(11, 6))
    for sev in [0, 1, 2, 3]:
        if len(sparc_detr_spectra_by_class[sev]) > 0 and sev in grouped.index:
            avg_detr = np.mean(sparc_detr_spectra_by_class[sev], axis=0)
            mean_fc = grouped.loc[sev, "Cutoff_Detrend"]

            plt.plot(
                f[f <= f_max], 
                avg_detr[f <= f_max], 
                label=f'Class {sev}', 
                color=colors[sev], 
                linewidth=2
            )

            fc_idx = np.argmin(np.abs(f - mean_fc))
            plt.plot(
                mean_fc, avg_detr[fc_idx],
                marker='X', markersize=9, color=colors[sev],
                markeredgecolor='black', markeredgewidth=1.2, zorder=5
            )

    add_band_spans()
    plt.axhline(sparc_amp_th, color='purple', linestyle='--', linewidth=1.5, label=f'SPARC Threshold ({sparc_amp_th})')

    detr_stats_lines = ["Per-Class SPARC (Detr) & Cutoff:"]
    for sev in [0, 1, 2, 3]:
        if sev in grouped.index:
            sd = grouped.loc[sev, "SPARC_Detrend"]
            fc = grouped.loc[sev, "Cutoff_Detrend"]
            detr_stats_lines.append(f"  Class {sev}: SPARC={sd:.3f} | fc={fc:.2f}Hz")

    plt.gca().text(
        0.02, 0.95, "\n".join(detr_stats_lines),
        transform=plt.gca().transAxes,
        fontsize=9, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray')
    )

    plt.title(f"Detrended SPARC Spectrum [MAD Normalization]\n(Threshold: {sparc_amp_th}, 'X' denotes mean Cutoff fc)", fontweight='bold')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Linear Magnitude")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    sparc_detr_file = out_dir / f"q2_SPARC_averages_detrend_th{sparc_amp_th}_maxf_{f_max}_4bands.png"
    plt.savefig(sparc_detr_file, dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 4: Linear Spectral Band AUC and Jitter TV
    # ---------------------------------------------------------
    plt.figure(figsize=(11, 6))
    for sev in [0, 1, 2, 3]:
        if len(sparc_orig_spectra_by_class[sev]) > 0 and sev in grouped.index:
            avg_orig = np.mean(sparc_orig_spectra_by_class[sev], axis=0)
            plt.plot(
                f[f <= f_max], 
                avg_orig[f <= f_max], 
                label=f'Class {sev}', 
                color=colors[sev], 
                linewidth=2
            )

    add_band_spans()

    lin_stats_lines = ["Per-Class Linear AUC (FOG | Relaxed | Fast | Jitter):"]
    for sev in [0, 1, 2, 3]:
        if sev in grouped.index:
            l_f = grouped.loc[sev, "Lin_FOG"]
            l_r = grouped.loc[sev, "Lin_Rel"]
            l_fa = grouped.loc[sev, "Lin_Fast"]
            l_j = grouped.loc[sev, "Lin_Jit"]
            lin_stats_lines.append(f"  Class {sev}: {l_f:5.2f} | {l_r:5.2f} | {l_fa:5.2f} | {l_j:5.2f}")

    lin_stats_lines.append("\nPer-Class Total Variation (FOG | Relaxed | Fast | Jitter):")
    for sev in [0, 1, 2, 3]:
        if sev in grouped.index:
            t_f = grouped.loc[sev, "TV_FOG"]
            t_r = grouped.loc[sev, "TV_Rel"]
            t_fa = grouped.loc[sev, "TV_Fast"]
            t_j = grouped.loc[sev, "TV_Jit"]
            lin_stats_lines.append(f"  Class {sev}: {t_f:5.2f} | {t_r:5.2f} | {t_fa:5.2f} | {t_j:5.2f}")

    plt.gca().text(
        0.02, 0.95, "\n".join(lin_stats_lines),
        transform=plt.gca().transAxes,
        fontsize=8.5, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray')
    )

    plt.title("Linear Magnitude Spectrum: 4-Band Segmentation & Jitter Complexity", fontweight='bold')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Linear Magnitude")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    lin_auc_file = out_dir / "q2_linear_complexity_4bands.png"
    plt.savefig(lin_auc_file, dpi=300)
    plt.close()

    print(f"\nSaved PSD plot to:               {psd_file}")
    print(f"Saved SPARC (Orig) plot to:      {sparc_orig_file}")
    print(f"Saved SPARC (Detrended) plot to: {sparc_detr_file}")
    print(f"Saved Linear Band AUC plot to:   {lin_auc_file}")


# -------------------------------------------------------------------------
# ADAPTED Q3: NUMERICAL GRID CONVERGENCE STUDY
# -------------------------------------------------------------------------
# QUESTION ANSWERED:
#   At what zero-padding factor does the discrete chord summation converge 
#   to the continuous DTFT arc length within numerical tolerance?
#
# INTERPRETATION GUIDE:
#   - Check the mean step delta: |Pad(k) - Pad(k-1)|.
#   - When the step delta drops below ~0.05, the discrete summation has converged.
#   - This justifies padlevel=4 as a mathematically rigorous choice in your methodology.
def analyze_padding_convergence(a_t_dict, labels_dict, out_dir):
    print("\n" + "=" * 78)
    print("Q3: NUMERICAL GRID CONVERGENCE STUDY (padlevel 0 -> 5)")
    print("=" * 78)

    pad_levels = [0, 1, 2, 3, 4, 5]
    knee_idx = 5
    sample_keys = list(a_t_dict.keys())[:150]

    def _calc_sparc(a_sig, pad):
        nfft = int(pow(2, np.ceil(np.log2(len(a_sig))) + pad))
        A = np.abs(np.fft.rfft(a_sig, n=nfft))
        A_norm = A / (A[0] + 1e-6)
        f_vec = np.fft.rfftfreq(nfft, d=1.0 / FPS)

        valid = f_vec <= MAX_FREQ
        f_sub = f_vec[valid]
        A_sub = A_norm[valid]

        dx = np.diff(f_sub) / MAX_FREQ
        dy = np.diff(A_sub)
        return -float(np.sum(np.sqrt(dx ** 2 + dy ** 2)))

    scores_by_pad = {p: [] for p in pad_levels}

    for key in sample_keys:
        a = a_t_dict[key][:, knee_idx]
        if len(a) < 2:
            continue
        for p in pad_levels:
            scores_by_pad[p].append(_calc_sparc(a, pad=p))

    df = pd.DataFrame(scores_by_pad)
    mean_scores = df.mean()

    print(f"{'Pad Level':<12} | {'N_FFT Multiplier':<18} | {'Mean SPARC':<12} | {'Step Delta |Δ|':<15}")
    print("-" * 65)

    prev_val = None
    deltas = []
    for p in pad_levels:
        val = mean_scores[p]
        delta_str = f"{abs(val - prev_val):.4f}" if prev_val is not None else "---"
        if prev_val is not None:
            deltas.append(abs(val - prev_val))
        print(f"  padlevel={p:<2} | {2**p:<18}x | {val:<12.4f} | {delta_str:<15}")
        prev_val = val

    # Plot convergence curve
    plt.figure(figsize=(8, 5))
    plt.plot(pad_levels, [mean_scores[p] for p in pad_levels], marker='o', linewidth=2, color='navy')
    plt.title("SPARC Discretization Convergence vs. Zero-Padding", fontweight='bold')
    plt.xlabel("Zero-Padding Level (padlevel)")
    plt.ylabel("Mean SPARC Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_file = out_dir / "q3_padding_convergence.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"\nSaved convergence curve to: {out_file}")


def compute_freq_weighted_sparc(
    a_sig: np.ndarray, 
    fps: int = 30, 
    padlevel: int = 4, 
    fc_max: float = 15.0, 
    fixed_nfft: int = None
):
    """
    Computes Frequency-Weighted SPARC without an amplitude threshold.
    The integration is strictly bounded by fc_max to ignore processing noise.
    """
    nfft = fixed_nfft if fixed_nfft else int(pow(2, np.ceil(np.log2(len(a_sig))) + padlevel))
    
    A = np.abs(np.fft.rfft(a_sig, n=nfft))
    A_0 = A[0] if A[0] > 0 else 1e-6
    f_vec = np.fft.rfftfreq(nfft, d=1.0 / fps)
    
    # Apply frequency weighting (Spectral Acceleration)
    A_weighted = (A / A_0) * f_vec
    
    # Hard cutoff based on frequency, not amplitude
    valid = f_vec <= fc_max
    f_sub = f_vec[valid]
    A_sub = A_weighted[valid]
    
    if len(f_sub) < 2:
        return 0.0, f_vec, A_weighted
        
    # Discrete integral
    dx = np.diff(f_sub) / fc_max  # Normalize horizontal steps by the evaluated bandwidth
    dy = np.diff(A_sub)
    arc_length = -float(np.sum(np.sqrt(dx ** 2 + dy ** 2)))
    
    return arc_length, f_vec, A_weighted


# -------------------------------------------------------------------------
# Q4: FREQ WEIGHTED SPARC FOR SPECIFIC JOINT GROUPS
# -------------------------------------------------------------------------
def analyze_cross_joint_distributions(
    a_t_dict,
    labels_dict,
    out_dir,
    fc_max: float = 8.0
):
    print("\n" + "=" * 100)
    print(f"Q3: CROSS-JOINT FREQUENCY DISTRIBUTIONS & WEIGHTED SPARC (MAX FREQ: {fc_max:.1f} Hz)")
    print("=" * 100)

    # Key SMPL joints for PD analysis
    target_joints = {
        'Pelvis': 0,
        'R_Knee': 5,
        'R_Ankle': 8,
        'R_Wrist': 21
    }

    nfft = 2048
    f = np.fft.rfftfreq(nfft, d=1.0 / FPS)
    colors = ['#85e085', '#ffe680', '#ffb366', '#ff6666']
    
    joint_results = []

    for joint_name, j_idx in target_joints.items():
        print(f"\nProcessing {joint_name} (Index {j_idx})...")
        
        unnorm_psd_by_class = {0: [], 1: [], 2: [], 3: []}
        sparc_fw_by_class = {0: [], 1: [], 2: [], 3: []}
        sparc_fw_spectra_by_class = {0: [], 1: [], 2: [], 3: []}
        
        for key, a_t in a_t_dict.items():
            sev = labels_dict.get(key.split('_down')[0], -1)
            if sev not in [0, 1, 2, 3]:
                continue
                
            a = a_t[:, j_idx]
            if len(a) < 2:
                continue
                
            # 1. Unnormalized PSD (to see true physical energy distribution)
            a_detrend = a - np.mean(a)
            A_raw = np.abs(np.fft.rfft(a_detrend, n=nfft))
            PSD_unnorm = (A_raw ** 2) / nfft
            unnorm_psd_by_class[sev].append(PSD_unnorm)
            
            # 2. Frequency-Weighted SPARC (No Amp Threshold)
            sparc_fw, _, A_weighted = compute_freq_weighted_sparc(
                a_sig=a, fps=FPS, fc_max=fc_max, fixed_nfft=nfft
            )
            sparc_fw_by_class[sev].append(sparc_fw)
            sparc_fw_spectra_by_class[sev].append(A_weighted)
            
        # Compile joint stats
        for sev in [0, 1, 2, 3]:
            if len(sparc_fw_by_class[sev]) > 0:
                mean_sparc_fw = np.mean(sparc_fw_by_class[sev])
                joint_results.append({
                    "Joint": joint_name,
                    "Severity": sev,
                    "FW_SPARC": mean_sparc_fw
                })
                
        # ---------------------------------------------------------
        # PLOT: Unnormalized PSD per Joint
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 5))
        for sev in [0, 1, 2, 3]:
            if len(unnorm_psd_by_class[sev]) > 0:
                avg_psd = np.mean(unnorm_psd_by_class[sev], axis=0)
                plt.plot(
                    f[f <= fc_max], 
                    avg_psd[f <= fc_max], 
                    label=f'Class {sev}', 
                    color=colors[sev], 
                    linewidth=2
                )

        plt.title(f"Unnormalized PSD: {joint_name} (Physical Energy Distribution)", fontweight='bold')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Raw Spectral Power")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_file = out_dir / f"q4_unnorm_PSD_{joint_name}_fmax_{fc_max}.png"
        plt.savefig(plot_file, dpi=300)
        plt.close()

        # ---------------------------------------------------------
        # PLOT: Frequency-Weighted Spectrum & Arc Length per Joint
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 5))
        for sev in [0, 1, 2, 3]:
            if len(sparc_fw_spectra_by_class[sev]) > 0:
                avg_fw = np.mean(sparc_fw_spectra_by_class[sev], axis=0)
                mean_sparc = np.mean(sparc_fw_by_class[sev])
                plt.plot(
                    f[f <= fc_max], 
                    avg_fw[f <= fc_max], 
                    label=f'Class {sev} (SPARC: {mean_sparc:.2f})', 
                    color=colors[sev], 
                    linewidth=2
                )

        fw_stats_lines = ["Per-Class FW-SPARC (Arc Length):"]
        for sev in [0, 1, 2, 3]:
            if len(sparc_fw_by_class[sev]) > 0:
                score = np.mean(sparc_fw_by_class[sev])
                fw_stats_lines.append(f"  Class {sev}: SPARC={score:.4f}")

        plt.gca().text(
            0.02, 0.95, "\n".join(fw_stats_lines),
            transform=plt.gca().transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray')
        )

        plt.title(f"Frequency-Weighted Spectrum: {joint_name}\n[A_norm * f up to {fc_max:.1f} Hz]", fontweight='bold')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Weighted Linear Magnitude")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        sparc_plot_file = out_dir / f"q3_FW_SPARC_{joint_name}.png"
        plt.savefig(sparc_plot_file, dpi=300)
        plt.close()

    # Print comparative FW-SPARC table
    df = pd.DataFrame(joint_results)
    pivot_df = df.pivot(index="Severity", columns="Joint", values="FW_SPARC")
    
    print(f"\n {'Class':<5} | {'Pelvis':<12} | {'R_Knee':<12} | {'R_Ankle':<12} | {'R_Wrist':<12}")
    print("-" * 65)
    for sev in [0, 1, 2, 3]:
        if sev in pivot_df.index:
            p = pivot_df.loc[sev, 'Pelvis']
            k = pivot_df.loc[sev, 'R_Knee']
            a = pivot_df.loc[sev, 'R_Ankle']
            w = pivot_df.loc[sev, 'R_Wrist']
            print(f"  {sev:<4} | {p:<12.4f} | {k:<12.4f} | {a:<12.4f} | {w:<12.4f}")


# -------------------------------------------------------------------------
# PLOTTING ALL JOINT SPECTRA: VISUALLY DEDUCE PATTERN OF PATHOLOGY
# -------------------------------------------------------------------------
def plot_all_joints_spectral_profiles(
    a_t_dict,
    labels_dict,
    out_dir,
    fc_max: float = 15.0,
    f_fog: float = 0.7,
    f_rel: float = 1.5,
    f_fast: float = 5.5
):
    print("\n" + "=" * 100)
    print(f"GENERATING DUAL-NORMALIZED SPECTRAL PROFILES FOR ALL 24 JOINTS (MAX FREQ: {fc_max:.1f} Hz)")
    print("=" * 100)

    smpl_joints = {
        0: 'Pelvis', 1: 'L_Hip', 2: 'R_Hip', 3: 'Spine_1',
        4: 'L_Knee', 5: 'R_Knee', 6: 'Spine_2', 7: 'L_Ankle',
        8: 'R_Ankle', 9: 'Spine_3', 10: 'L_Foot', 11: 'R_Foot',
        12: 'Neck', 13: 'L_Collar', 14: 'R_Collar', 15: 'Head',
        16: 'L_Shoulder', 17: 'R_Shoulder', 18: 'L_Elbow', 19: 'R_Elbow',
        20: 'L_Wrist', 21: 'R_Wrist', 22: 'L_Hand', 23: 'R_Hand'
    }

    nfft = 2048
    f = np.fft.rfftfreq(nfft, d=1.0 / FPS)
    colors = ['#85e085', '#ffe680', '#ffb366', '#ff6666']

    joint_dir = out_dir / "all_joints_spectral_profiles"
    joint_dir.mkdir(parents=True, exist_ok=True)

    def _add_band_background(ax):
        ax.axvspan(0, f_fog, color='gray', alpha=0.08, label='FOG/Shuffling')
        ax.axvspan(f_fog, f_rel, color='green', alpha=0.08, label='Relaxed Gait')
        ax.axvspan(f_rel, f_fast, color='blue', alpha=0.08, label='Fast Gait')
        ax.axvspan(f_fast, fc_max, color='red', alpha=0.08, label='Jitter Band')
        ax.axvline(f_fog, color='black', linestyle=':', alpha=0.3)
        ax.axvline(f_rel, color='black', linestyle=':', alpha=0.3)
        ax.axvline(f_fast, color='black', linestyle=':', alpha=0.3)

    for j_idx, joint_name in tqdm(smpl_joints.items(), desc="Plotting Joint Profiles"):
        phys_amp_by_class = {0: [], 1: [], 2: [], 3: []}
        auc_norm_psd_by_class = {0: [], 1: [], 2: [], 3: []}

        for key, a_t in a_t_dict.items():
            sev = labels_dict.get(key.split('_down')[0], -1)
            if sev not in [0, 1, 2, 3]:
                continue

            a = a_t[:, j_idx]
            if len(a) < 2:
                continue

            T = len(a)
            a_detrend = a - np.mean(a)
            A_raw = np.abs(np.fft.rfft(a_detrend, n=nfft))

            # 1. Physical Amplitude Spectrum: normalized by sequence length T (rad/s)
            A_physical = (2.0 / T) * A_raw
            phys_amp_by_class[sev].append(A_physical)

            # 2. Relative PSD: normalized by total spectral area/sum (Unitless proportion)
            PSD_raw = (A_raw ** 2) / nfft
            PSD_norm = PSD_raw / (np.sum(PSD_raw) + 1e-8)
            auc_norm_psd_by_class[sev].append(PSD_norm)

        valid_f = f <= fc_max
        fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

        # ---------------------------------------------------------
        # LEFT: Absolute Scale (Physical Amplitude in rad/s)
        # ---------------------------------------------------------
        for sev in [0, 1, 2, 3]:
            if len(phys_amp_by_class[sev]) > 0:
                avg_phys = np.mean(phys_amp_by_class[sev], axis=0)
                axes[0].plot(
                    f[valid_f], 
                    avg_phys[valid_f], 
                    label=f'Class {sev} (n={len(phys_amp_by_class[sev])})', 
                    color=colors[sev], 
                    linewidth=2
                )

        _add_band_background(axes[0])
        axes[0].set_title(
            f"Physical Amplitude Spectrum: {joint_name} (Joint {j_idx})\n"
            f"[Normalized by Sequence Length T → Absolute Scale in rad/s]",
            fontweight='bold', fontsize=11
        )
        axes[0].set_xlabel("Frequency (Hz)")
        axes[0].set_ylabel("Harmonic Amplitude (rad/s)")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        # ---------------------------------------------------------
        # RIGHT: Relative Proportion (Normalized by Total AUC)
        # ---------------------------------------------------------
        for sev in [0, 1, 2, 3]:
            if len(auc_norm_psd_by_class[sev]) > 0:
                avg_psd = np.mean(auc_norm_psd_by_class[sev], axis=0)
                axes[1].plot(
                    f[valid_f], 
                    avg_psd[valid_f], 
                    label=f'Class {sev}', 
                    color=colors[sev], 
                    linewidth=2
                )

        _add_band_background(axes[1])
        axes[1].set_title(
            f"Relative Energy Distribution: {joint_name} (Joint {j_idx})\n"
            f"[Normalized by Total AUC → Shape / Signal-to-Noise Ratio]",
            fontweight='bold', fontsize=11
        )
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].set_ylabel("Fractional Power / Bin")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = joint_dir / f"joint_{j_idx:02d}_{joint_name}_dual_profile.png"
        plt.savefig(plot_file, dpi=300)
        plt.close()

    print(f"\nSaved all 24 dual-profile joint figures to:\n  {joint_dir.resolve()}")



# =========================================================================
# PLOT FINAL JOINT AND METRIC DECISIONS FOR METHOD VERIFICATION
# =========================================================================
def analyze_q6_clinical_metrics(
    a_t_dict,
    labels_dict,
    out_dir,
    fps: int = 30,
    nfft: int = 2048
):
    print("\n" + "=" * 115)
    print("Q6: CLINICAL METRICS EVALUATION (BRADYKINESIA & SMOOTHNESS)")
    print("=" * 115)

    q6_dir = out_dir / "q6_clinical_metrics"
    q6_dir.mkdir(parents=True, exist_ok=True)

    f = np.fft.rfftfreq(nfft, d=1.0 / fps)
    colors = ['#85e085', '#ffe680', '#ffb366', '#ff6666']
    classes = [0, 1, 2, 3]

    # Clinical frequency band masks
    mask_voluntary = (f >= 0.5) & (f <= 3.0)
    mask_jitter = (f > 3.0) & (f <= 8.0)
    mask_brady = (f >= 0.5) & (f <= 8.0)

    # Integration helper (compatible across NumPy versions)
    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

    def _calc_robinson_si(x_l, x_r, eps=1e-8):
        """Robinson's Symmetry Index: 2 * |L - R| / (L + R) * 100%"""
        return (2.0 * np.abs(x_l - x_r) / (x_l + x_r + eps)) * 100.0

    def _add_band_background(ax, max_f=8.0):
        ax.axvspan(0.5, 3.0, color='green', alpha=0.08, label='Voluntary Locomotion (0.5-3 Hz)')
        ax.axvspan(3.0, max_f, color='red', alpha=0.08, label='Tremor/Jitter/FOG (3-8 Hz)')
        ax.axvline(0.5, color='gray', linestyle=':', alpha=0.4)
        ax.axvline(3.0, color='black', linestyle=':', alpha=0.5)
        ax.axvline(max_f, color='gray', linestyle=':', alpha=0.4)

    # =========================================================================
    # PART 1: BRADYKINESIA ANALYSIS (PHYSICAL AMPLITUDE SPECTRUM, 0.5 - 8.0 Hz)
    # =========================================================================
    print("\n--- Computing Bradykinesia Physical AUC (0.5 - 8.0 Hz) ---")

    # 1A. Bilateral Extremities (Ankles & Shoulders) -> Combined L/R Plot + Robinson SI
    bilateral_groups = {
        "Ankles": {"L": (7, "L_Ankle"), "R": (8, "R_Ankle")},
        "Shoulders": {"L": (16, "L_Shoulder"), "R": (17, "R_Shoulder")}
    }

    for group_name, j_info in bilateral_groups.items():
        spectra_data = {"L": {c: [] for c in classes}, "R": {c: [] for c in classes}}
        seq_level_metrics = []

        for key, a_t in a_t_dict.items():
            sev = labels_dict.get(key.split('_down')[0], -1)
            if sev not in classes:
                continue

            row = {"Severity": sev}
            for sk in ["L", "R"]:
                j_idx, j_name = j_info[sk]
                a = a_t[:, j_idx]
                if len(a) < 2:
                    row[f"AUC_{sk}"] = np.nan
                    continue

                T = len(a)
                a_detrend = a - np.mean(a)
                A_raw = np.abs(np.fft.rfft(a_detrend, n=nfft))
                A_phys = (2.0 / T) * A_raw  # Harmonic amplitude in rad/s
                spectra_data[sk][sev].append(A_phys)

                # Physical AUC over 0.5 - 8.0 Hz
                row[f"AUC_{sk}"] = float(trapz_fn(A_phys[mask_brady], f[mask_brady]))

            if not np.isnan(row.get("AUC_L", np.nan)) and not np.isnan(row.get("AUC_R", np.nan)):
                row["SI"] = _calc_robinson_si(row["AUC_L"], row["AUC_R"])

            seq_level_metrics.append(row)

        df_metrics = pd.DataFrame(seq_level_metrics)
        grouped = df_metrics.groupby("Severity").mean()

        # Terminal Report
        print(f"\n[Bradykinesia] Joint Group: {group_name}")
        print(f" {'Class':<5} | {'Mean AUC (Left)':<17} | {'Mean AUC (Right)':<17} | {'Robinson SI (%)':<15}")
        print("-" * 62)
        for c in classes:
            if c in grouped.index:
                al = grouped.loc[c, "AUC_L"]
                ar = grouped.loc[c, "AUC_R"]
                si = grouped.loc[c, "SI"]
                print(f"  {c:<4} | {al:<17.4f} | {ar:<17.4f} | {si:<15.2f}%")

        # Plot Bilateral Overlay
        plt.figure(figsize=(11, 6))
        valid_f = (f >= 0) & (f <= 8.0)

        for c in classes:
            if len(spectra_data["L"][c]) > 0:
                mean_l = np.mean(spectra_data["L"][c], axis=0)
                plt.plot(f[valid_f], mean_l[valid_f], color=colors[c], linestyle='-', linewidth=2.0,
                         label=f'Class {c} (Left - Solid)')
            if len(spectra_data["R"][c]) > 0:
                mean_r = np.mean(spectra_data["R"][c], axis=0)
                plt.plot(f[valid_f], mean_r[valid_f], color=colors[c], linestyle='--', linewidth=2.0,
                         label=f'Class {c} (Right - Dashed)')

        _add_band_background(plt.gca(), max_f=8.0)

        stats_box = [f"{group_name} Bradykinesia Stats (0.5-8.0 Hz):", "Class | AUC(L) | AUC(R) | Robinson SI"]
        for c in classes:
            if c in grouped.index:
                al = grouped.loc[c, "AUC_L"]
                ar = grouped.loc[c, "AUC_R"]
                si = grouped.loc[c, "SI"]
                stats_box.append(f"  {c}   | {al:6.3f} | {ar:6.3f} | {si:5.1f}%")

        plt.gca().text(
            0.02, 0.95, "\n".join(stats_box),
            transform=plt.gca().transAxes,
            fontsize=8.5, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.88, edgecolor='gray')
        )

        plt.title(f"Bradykinesia Profile: {group_name}\n[Physical Amplitude Spectrum: Absolute Scale in rad/s]",
                  fontweight='bold', fontsize=11)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Harmonic Amplitude (rad/s)")
        plt.legend(loc="upper right", fontsize=8.5)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(q6_dir / f"bradykinesia_{group_name.lower()}_physical_spectrum.png", dpi=300)
        plt.close()

    # 1B. Axial Rigidity Joints (Spine 2 & Spine 3) -> Separate Figures
    spine_joints = [(6, "Spine_2"), (9, "Spine_3")]
    spine_results = []

    for j_idx, j_name in spine_joints:
        spectra_data = {c: [] for c in classes}
        seq_level_metrics = []

        for key, a_t in a_t_dict.items():
            sev = labels_dict.get(key.split('_down')[0], -1)
            if sev not in classes:
                continue

            a = a_t[:, j_idx]
            if len(a) < 2:
                continue

            T = len(a)
            a_detrend = a - np.mean(a)
            A_raw = np.abs(np.fft.rfft(a_detrend, n=nfft))
            A_phys = (2.0 / T) * A_raw
            spectra_data[sev].append(A_phys)

            auc_val = float(trapz_fn(A_phys[mask_brady], f[mask_brady]))
            seq_level_metrics.append({"Severity": sev, "AUC": auc_val})

        df_spine = pd.DataFrame(seq_level_metrics)
        grouped_spine = df_spine.groupby("Severity").mean()

        for c in classes:
            if c in grouped_spine.index:
                spine_results.append({
                    "Joint": j_name,
                    "Severity": c,
                    "AUC": grouped_spine.loc[c, "AUC"]
                })

        # Separate Figure for each spine joint
        plt.figure(figsize=(10, 5.5))
        valid_f = (f >= 0) & (f <= 8.0)

        for c in classes:
            if len(spectra_data[c]) > 0:
                mean_spec = np.mean(spectra_data[c], axis=0)
                plt.plot(f[valid_f], mean_spec[valid_f], color=colors[c], linewidth=2.0,
                         label=f'Class {c}')

        _add_band_background(plt.gca(), max_f=8.0)

        stats_box = [f"{j_name} Bradykinesia Stats (0.5-8.0 Hz):", "Class | Physical AUC"]
        for c in classes:
            if c in grouped_spine.index:
                val = grouped_spine.loc[c, "AUC"]
                stats_box.append(f"  {c}   |   {val:8.4f}")

        plt.gca().text(
            0.02, 0.95, "\n".join(stats_box),
            transform=plt.gca().transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.88, edgecolor='gray')
        )

        plt.title(f"Bradykinesia Profile: {j_name} (Joint {j_idx})\n[Physical Amplitude Spectrum: Absolute Scale in rad/s]",
                  fontweight='bold', fontsize=11)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Harmonic Amplitude (rad/s)")
        plt.legend(loc="upper right", fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(q6_dir / f"bradykinesia_{j_name.lower()}_physical_spectrum.png", dpi=300)
        plt.close()

    # Terminal Report for Spine Joints
    df_spine_all = pd.DataFrame(spine_results)
    pivot_spine = df_spine_all.pivot(index="Severity", columns="Joint", values="AUC")
    print(f"\n[Bradykinesia] Axial Spine Joints (Mean Physical AUC)")
    print(f" {'Class':<5} | {'Spine_2 (Joint 6)':<20} | {'Spine_3 (Joint 9)':<20}")
    print("-" * 52)
    for c in classes:
        if c in pivot_spine.index:
            s2 = pivot_spine.loc[c, "Spine_2"] if "Spine_2" in pivot_spine.columns else np.nan
            s3 = pivot_spine.loc[c, "Spine_3"] if "Spine_3" in pivot_spine.columns else np.nan
            print(f"  {c:<4} | {s2:<20.4f} | {s3:<20.4f}")

    # =========================================================================
    # PART 2: SMOOTHNESS ANALYSIS (RELATIVE ENERGY SPECTRUM, 3.0 - 8.0 Hz)
    # =========================================================================
    print("\n--- Computing Smoothness Relative Jitter AUC (3.0 - 8.0 Hz) ---")

    # Distal extremities evaluated independently without Robinson SI
    smooth_joints = [
        (20, "L_Wrist"),
        (21, "R_Wrist"),
        (22, "L_Hand"),
        (23, "R_Hand")
    ]

    smoothness_results = []

    for j_idx, j_name in smooth_joints:
        spectra_data = {c: [] for c in classes}
        seq_level_metrics = []

        for key, a_t in a_t_dict.items():
            sev = labels_dict.get(key.split('_down')[0], -1)
            if sev not in classes:
                continue

            a = a_t[:, j_idx]
            if len(a) < 2:
                continue

            a_detrend = a - np.mean(a)
            A_raw = np.abs(np.fft.rfft(a_detrend, n=nfft))
            PSD_raw = (A_raw ** 2) / nfft
            PSD_norm = PSD_raw / (np.sum(PSD_raw) + 1e-8)  # Relative unitless distribution
            spectra_data[sev].append(PSD_norm)

            # Smoothness / Jitter AUC over 3.0 - 8.0 Hz
            jitter_auc = float(np.sum(PSD_norm[mask_jitter]))
            seq_level_metrics.append({"Severity": sev, "Jitter_AUC": jitter_auc})

        df_smooth = pd.DataFrame(seq_level_metrics)
        grouped_smooth = df_smooth.groupby("Severity").mean()

        for c in classes:
            if c in grouped_smooth.index:
                smoothness_results.append({
                    "Joint": j_name,
                    "Severity": c,
                    "Jitter_AUC": grouped_smooth.loc[c, "Jitter_AUC"]
                })

        # Separate Figure for each extremity joint
        plt.figure(figsize=(10, 5.5))
        valid_f = (f >= 0) & (f <= 8.0)

        for c in classes:
            if len(spectra_data[c]) > 0:
                mean_psd = np.mean(spectra_data[c], axis=0)
                plt.plot(f[valid_f], mean_psd[valid_f], color=colors[c], linewidth=2.0,
                         label=f'Class {c}')

        _add_band_background(plt.gca(), max_f=8.0)

        stats_box = [f"{j_name} Jitter Fraction (3.0-8.0 Hz):", "Class | Jitter AUC"]
        for c in classes:
            if c in grouped_smooth.index:
                val = grouped_smooth.loc[c, "Jitter_AUC"]
                stats_box.append(f"  {c}   |   {val:8.4f}")

        plt.gca().text(
            0.02, 0.95, "\n".join(stats_box),
            transform=plt.gca().transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.88, edgecolor='gray')
        )

        plt.title(f"Smoothness Degradation: {j_name} (Joint {j_idx})\n[Relative Energy Distribution: Pathological Jitter (3-8 Hz)]",
                  fontweight='bold', fontsize=11)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Fractional Power / Bin")
        plt.legend(loc="upper right", fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(q6_dir / f"smoothness_{j_name.lower()}_relative_energy.png", dpi=300)
        plt.close()

    # Terminal Report for Smoothness Joints
    df_smooth_all = pd.DataFrame(smoothness_results)
    pivot_smooth = df_smooth_all.pivot(index="Severity", columns="Joint", values="Jitter_AUC")
    print(f"\n[Smoothness] Distal Extremity Joints (Mean Jitter AUC, 3.0 - 8.0 Hz)")
    print(f" {'Class':<5} | {'L_Wrist':<12} | {'R_Wrist':<12} | {'L_Hand':<12} | {'R_Hand':<12}")
    print("-" * 62)
    for c in classes:
        if c in pivot_smooth.index:
            lw = pivot_smooth.loc[c, "L_Wrist"] if "L_Wrist" in pivot_smooth.columns else np.nan
            rw = pivot_smooth.loc[c, "R_Wrist"] if "R_Wrist" in pivot_smooth.columns else np.nan
            lh = pivot_smooth.loc[c, "L_Hand"] if "L_Hand" in pivot_smooth.columns else np.nan
            rh = pivot_smooth.loc[c, "R_Hand"] if "R_Hand" in pivot_smooth.columns else np.nan
            print(f"  {c:<4} | {lw:<12.4f} | {rw:<12.4f} | {lh:<12.4f} | {rh:<12.4f}")

    print(f"\nAll Q6 diagnostic plots and tables saved to:\n  {q6_dir.resolve()}\n")


def analyze_q6_clinical_metrics_upgraded(
    a_t_dict,
    labels_dict,
    out_dir=None,
    fps: int = 30,
    nfft: int = 2048,
    show_outliers: bool = False
):
    print("\n" + "=" * 115)
    print("UPGRADED Q6: CONDENSED CLINICAL METRICS (WORSE-SIDE + BILATERAL ASYMMETRY)")
    print("=" * 115)

    # Route output directly to visualizations/q6_clinical_metrics_upgraded
    if out_dir is not None and "visualizations" in Path(out_dir).parts:
        q6_dir = Path(out_dir) / "FINAL_clinical_metrics"
    else:
        q6_dir = Path("thesis/visualizations/FINAL_clinical_metrics")
    q6_dir.mkdir(parents=True, exist_ok=True)

    RAD2DEG = 180.0 / np.pi
    f = np.fft.rfftfreq(nfft, d=1.0 / fps)
    colors = ['#85e085', '#ffe680', '#ffb366', '#ff6666']
    classes = [0, 1, 2, 3]

    # Clinical frequency band definitions
    mask_voluntary = (f >= 0.5) & (f <= 3.0)
    mask_jitter = (f > 3.0) & (f <= 8.0)
    mask_brady = (f >= 0.5) & (f <= 8.0)
    valid_f = (f >= 0.0) & (f <= 8.0)

    # Integration helper across NumPy versions
    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

    def _calc_robinson_si(x_l, x_r, eps=1e-8):
        """Calculates Robinson's Symmetry Index: 2 * |L - R| / (L + R) * 100%"""
        return (2.0 * np.abs(x_l - x_r) / (x_l + x_r + eps)) * 100.0

    def _add_band_background(ax, max_f=8.0):
        ax.axvspan(0.5, 3.0, color='green', alpha=0.08, label='Voluntary Locomotion Band (0.5–3.0 Hz)')
        ax.axvspan(3.0, max_f, color='red', alpha=0.08, label='Tremor / Freezing / Jitter Band (3.0–8.0 Hz)')
        ax.axvline(0.5, color='gray', linestyle=':', alpha=0.4)
        ax.axvline(3.0, color='black', linestyle=':', alpha=0.5)
        ax.axvline(max_f, color='gray', linestyle=':', alpha=0.4)

    # Storage for spectra of the selected worse sides across severity classes
    worse_spectra = {
        "ankle": {c: [] for c in classes},
        "spine": {c: [] for c in classes},
        "wrist": {c: [] for c in classes},
        "hand": {c: [] for c in classes},
    }

    per_sequence_rows = []

    # SMPL Joint Indices:
    # Ankles: L=7, R=8 | Spine: Spine2=6, Spine3=9
    # Wrists: L=20, R=21 | Hands: L=22, R=23
    for key, a_t in a_t_dict.items():
        sev = labels_dict.get(key.split('_down')[0], -1)
        if sev not in classes:
            continue

        T = a_t.shape[0]
        if T < 2:
            continue

        # Precompute FFTs for relevant joints
        relevant_joints = [6, 7, 8, 9, 20, 21, 22, 23]
        a_phys_all = {}
        psd_norm_all = {}

        for j in relevant_joints:
            sig = a_t[:, j]
            sig_detrend = sig - np.mean(sig)
            A_raw = np.abs(np.fft.rfft(sig_detrend, n=nfft))
            
            # Physical Amplitude Spectrum in deg/s
            a_phys_all[j] = (2.0 / T) * A_raw * RAD2DEG
            
            # Relative Power Spectral Density (Normalized by total variance)
            psd_raw = (A_raw ** 2) / nfft
            psd_norm_all[j] = psd_raw / (np.sum(psd_raw) + 1e-8)

        # -------------------------------------------------------------
        # 1. BRADYKINESIA: Physical Amplitude AUC (0.5 - 8.0 Hz) in deg/s·Hz
        # -------------------------------------------------------------
        # Ankles (L=7, R=8)
        auc_l_ank = float(trapz_fn(a_phys_all[7][mask_brady], f[mask_brady]))
        auc_r_ank = float(trapz_fn(a_phys_all[8][mask_brady], f[mask_brady]))
        worse_ank_auc = min(auc_l_ank, auc_r_ank)
        worse_ank_spec = a_phys_all[7] if auc_l_ank <= auc_r_ank else a_phys_all[8]
        ankle_si = _calc_robinson_si(auc_l_ank, auc_r_ank)
        worse_spectra["ankle"][sev].append(worse_ank_spec)

        # Axial Spine Rigidity (Spine2=6, Spine3=9)
        auc_s2 = float(trapz_fn(a_phys_all[6][mask_brady], f[mask_brady]))
        auc_s3 = float(trapz_fn(a_phys_all[9][mask_brady], f[mask_brady]))
        axial_auc = 0.5 * (auc_s2 + auc_s3)
        spine_spec = 0.5 * (a_phys_all[6] + a_phys_all[9])
        worse_spectra["spine"][sev].append(spine_spec)

        # -------------------------------------------------------------
        # 2. SMOOTHNESS / TREMOR: Relative Jitter AUC (3.0 - 8.0 Hz) in %
        # -------------------------------------------------------------
        def _calc_jitter_pct_trapz(psd_signal):
            jitter_area = trapz_fn(psd_signal[mask_jitter], f[mask_jitter])
            total_area = trapz_fn(psd_signal, f) + 1e-8
            return float(jitter_area / total_area) * 100.0

        # Wrists (L=20, R=21)
        jit_l_wri = _calc_jitter_pct_trapz(psd_norm_all[20])
        jit_r_wri = _calc_jitter_pct_trapz(psd_norm_all[21])
        worse_wri_jit = max(jit_l_wri, jit_r_wri)
        worse_wri_spec = psd_norm_all[20] if jit_l_wri >= jit_r_wri else psd_norm_all[21]
        worse_spectra["wrist"][sev].append(worse_wri_spec)

        # Hands (L=22, R=23)
        jit_l_hnd = _calc_jitter_pct_trapz(psd_norm_all[22])
        jit_r_hnd = _calc_jitter_pct_trapz(psd_norm_all[23])
        worse_hnd_jit = max(jit_l_hnd, jit_r_hnd)
        worse_hnd_spec = psd_norm_all[22] if jit_l_hnd >= jit_r_hnd else psd_norm_all[23]
        worse_spectra["hand"][sev].append(worse_hnd_spec)

        per_sequence_rows.append({
            "Key": key,
            "Severity": sev,
            "Worse_Ankle_AUC": worse_ank_auc,
            "Axial_Spine_AUC": axial_auc,
            "Worse_Wrist_Jitter": worse_wri_jit,
            "Worse_Hand_Jitter": worse_hnd_jit,
            "Ankle_SI": ankle_si
        })

    df = pd.DataFrame(per_sequence_rows)
    mean_stats = df.groupby("Severity").mean(numeric_only=True)
    std_stats = df.groupby("Severity").std(numeric_only=True)

    # =========================================================================
    # TERMINAL REPORT
    # =========================================================================
    print(f"\n{'='*95}")
    print(f" CONDENSED CLINICAL METRICS SUMMARY (PER-CLASS MEANS ± STD)")
    print(f"{'='*95}")
    header = (
        f"{'Class':<5} | {'Worse Ankle AUC (°/s·Hz)':<25} | "
        f"{'Axial Spine AUC (°/s·Hz)':<26} | {'Ankle SI (%)':<13}"
    )
    print(header)
    print("-" * len(header))
    for c in classes:
        if c in mean_stats.index:
            wa = f"{mean_stats.loc[c, 'Worse_Ankle_AUC']:.2f}±{std_stats.loc[c, 'Worse_Ankle_AUC']:.2f}"
            ax = f"{mean_stats.loc[c, 'Axial_Spine_AUC']:.2f}±{std_stats.loc[c, 'Axial_Spine_AUC']:.2f}"
            asi = f"{mean_stats.loc[c, 'Ankle_SI']:.1f}±{std_stats.loc[c, 'Ankle_SI']:.1f}%"
            print(f" {c:<4} | {wa:<25} | {ax:<26} | {asi:<13}")

    print("\n" + "-" * 75)
    header_jitter = f"{'Class':<5} | {'Worse Wrist Jitter (3–8 Hz)':<28} | {'Worse Hand Jitter (3–8 Hz)':<26}"
    print(header_jitter)
    print("-" * 75)
    for c in classes:
        if c in mean_stats.index:
            ww = f"{mean_stats.loc[c, 'Worse_Wrist_Jitter']:.2f}±{std_stats.loc[c, 'Worse_Wrist_Jitter']:.2f}%"
            wh = f"{mean_stats.loc[c, 'Worse_Hand_Jitter']:.2f}±{std_stats.loc[c, 'Worse_Hand_Jitter']:.2f}%"
            print(f" {c:<4} | {ww:<28} | {wh:<26}")
    print("-" * 75)

    # =========================================================================
    # VISUALIZATION PIPELINE
    # =========================================================================
    
    # 1. Worse Ankle Bradykinesia Spectrum
    plt.figure(figsize=(10, 5.5))
    for c in classes:
        if len(worse_spectra["ankle"][c]) > 0:
            avg_curve = np.mean(worse_spectra["ankle"][c], axis=0)
            plt.plot(f[valid_f], avg_curve[valid_f], color=colors[c], linewidth=2.2, label=f'Class {c}')
    _add_band_background(plt.gca(), max_f=8.0)
    stats_box = ["Worse Ankle Bradykinesia (0.5–8.0 Hz):", "Class | Mean AUC (deg/s·Hz)"]
    for c in classes:
        if c in mean_stats.index:
            stats_box.append(f"  {c}   |   {mean_stats.loc[c, 'Worse_Ankle_AUC']:8.2f}")
    plt.gca().text(0.02, 0.95, "\n".join(stats_box), transform=plt.gca().transAxes,
                   fontsize=9, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
    plt.title("Bradykinesia Profile: Worse Ankle (Joints 7 & 8)\n"
              "[Physical Amplitude Spectrum Normalized by T → Absolute Scale in deg/s]",
              fontweight='bold', fontsize=11)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Harmonic Amplitude (deg/s)")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(q6_dir / "01_worse_ankle_bradykinesia_spectrum.png", dpi=300)
    plt.close()

    # 2. Axial Spine Rigidity Spectrum
    plt.figure(figsize=(10, 5.5))
    for c in classes:
        if len(worse_spectra["spine"][c]) > 0:
            avg_curve = np.mean(worse_spectra["spine"][c], axis=0)
            plt.plot(f[valid_f], avg_curve[valid_f], color=colors[c], linewidth=2.2, label=f'Class {c}')
    _add_band_background(plt.gca(), max_f=8.0)
    stats_box = ["Axial Spine Rigidity (0.5–8.0 Hz):", "Class | Mean AUC (deg/s·Hz)"]
    for c in classes:
        if c in mean_stats.index:
            stats_box.append(f"  {c}   |   {mean_stats.loc[c, 'Axial_Spine_AUC']:8.2f}")
    plt.gca().text(0.02, 0.95, "\n".join(stats_box), transform=plt.gca().transAxes,
                   fontsize=9, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
    plt.title("Axial Rigidity Profile: Trunk Spine (Mean of Spine 2 & Spine 3)\n"
              "[Physical Amplitude Spectrum Normalized by T → Absolute Scale in deg/s]",
              fontweight='bold', fontsize=11)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Harmonic Amplitude (deg/s)")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(q6_dir / "02_axial_spine_rigidity_spectrum.png", dpi=300)
    plt.close()

    # 3. Worse Wrist Smoothness Degradation Spectrum
    plt.figure(figsize=(10, 5.5))
    for c in classes:
        if len(worse_spectra["wrist"][c]) > 0:
            avg_curve = np.mean(worse_spectra["wrist"][c], axis=0)
            plt.plot(f[valid_f], avg_curve[valid_f], color=colors[c], linewidth=2.2, label=f'Class {c}')
    _add_band_background(plt.gca(), max_f=8.0)
    stats_box = ["Worse Wrist Jitter Energy (3.0–8.0 Hz):", "Class | Relative Jitter AUC (%)"]
    for c in classes:
        if c in mean_stats.index:
            stats_box.append(f"  {c}   |   {mean_stats.loc[c, 'Worse_Wrist_Jitter']:6.2f}%")
    plt.gca().text(0.02, 0.95, "\n".join(stats_box), transform=plt.gca().transAxes,
                   fontsize=9, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
    plt.title("Smoothness Degradation: Worse Wrist (Joints 20 & 21)\n"
              "[Relative Energy Distribution: Pathological Jitter (3.0–8.0 Hz)]",
              fontweight='bold', fontsize=11)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Fractional Power / Bin")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(q6_dir / "03_worse_wrist_smoothness_spectrum.png", dpi=300)
    plt.close()

    # 4. Worse Hand Smoothness Degradation Spectrum
    plt.figure(figsize=(10, 5.5))
    for c in classes:
        if len(worse_spectra["hand"][c]) > 0:
            avg_curve = np.mean(worse_spectra["hand"][c], axis=0)
            plt.plot(f[valid_f], avg_curve[valid_f], color=colors[c], linewidth=2.2, label=f'Class {c}')
    _add_band_background(plt.gca(), max_f=8.0)
    stats_box = ["Worse Hand Jitter Energy (3.0–8.0 Hz):", "Class | Relative Jitter AUC (%)"]
    for c in classes:
        if c in mean_stats.index:
            stats_box.append(f"  {c}   |   {mean_stats.loc[c, 'Worse_Hand_Jitter']:6.2f}%")
    plt.gca().text(0.02, 0.95, "\n".join(stats_box), transform=plt.gca().transAxes,
                   fontsize=9, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
    plt.title("Smoothness Degradation: Worse Hand (Joints 22 & 23)\n"
              "[Relative Energy Distribution: Pathological Jitter (3.0–8.0 Hz)]",
              fontweight='bold', fontsize=11)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Fractional Power / Bin")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(q6_dir / "04_worse_hand_smoothness_spectrum.png", dpi=300)
    plt.close()

    # 5. Bilateral Motor Asymmetry (Robinson SI Boxplot for Ankles)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ankle_box_data = [df[df['Severity'] == c]['Ankle_SI'].dropna().values for c in classes]
    bp0 = ax.boxplot(ankle_box_data, labels=[f"Class {c}" for c in classes],
                     patch_artist=True, showfliers=show_outliers, widths=0.45)
    for patch, col in zip(bp0['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax.set_title("Lower Extremity Asymmetry: Ankles\n"
                 "[Robinson's Symmetry Index: 2 * |L - R| / (L + R) * 100%]",
                 fontweight='bold', fontsize=12)
    ax.set_ylabel("Robinson Symmetry Index (%)", fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(q6_dir / "05_bilateral_asymmetry_progression.png", dpi=300)
    plt.close()

    # 6. Master Distribution Dashboard of all 5 Core Metrics
    metric_fields = [
        ("Worse_Ankle_AUC", "Worse Ankle AUC (deg/s·Hz)", "Lower = More Severe"),
        ("Axial_Spine_AUC", "Axial Spine AUC (deg/s·Hz)", "Lower = More Severe"),
        ("Ankle_SI", "Ankle Robinson SI (%)", "Higher = More Asymmetric"),
        ("Worse_Wrist_Jitter", "Worse Wrist Jitter Energy (%)", "Higher = More Severe"),
        ("Worse_Hand_Jitter", "Worse Hand Jitter Energy (%)", "Higher = More Severe")
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()

    for idx, (col_name, plot_label, clinical_interp) in enumerate(metric_fields):
        ax = axes_flat[idx]
        box_data = [df[df['Severity'] == c][col_name].dropna().values for c in classes]
        bp = ax.boxplot(box_data, labels=[f"Class {c}" for c in classes],
                        patch_artist=True, showfliers=show_outliers, widths=0.45)
        for patch, col in zip(bp['boxes'], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)
        ax.set_title(f"{plot_label}\n({clinical_interp})", fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    # Turn off the unused 6th subplot cell
    axes_flat[5].axis('off')

    fig.suptitle("Master Clinical Gait Metrics: Distribution Progression by Severity",
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(q6_dir / "06_condensed_clinical_metrics_dashboard.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nAll upgraded Q6 clinical figures successfully saved to:\n  {q6_dir.resolve()}\n")
    return df


if __name__ == "__main__":
    out_dir = Path("thesis/visualizations/analyses/SPARC/")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset configuration using dataloaders
    raw_smpl_path = Path("thesis/data/raw/PD-GaM/3D_SMPL/PD-GaM_3D_SMPL_rot_trans_canonical.npz")
    labels_path = Path("thesis/data/metadata/pd_gam_labels.json")

    cfg = {
        'data': {
            'smpl_path': raw_smpl_path,
            'severity_labels_path': labels_path,
            'patient_prefix': 'all',
        },
        'windowing': {
            'total_window_size': 60,
            'prefix_length': 15,
            'step_size': 45,
            'max_sequence_len': 200,   # High ceiling so sequences are not sliced
            'full_seq_step_size': 200,
            'min_z_travel': 0.5,
            'filter_z_travel': True,   # filter out stationary Class 3 sequences
        },
        'training': {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 0,
            'eval_split': 0.0,         # 100% of the dataset included in train pool
            'test_split': 0.0,
            'overfit_severity_class': -1,
        },
        'model': {
            'generation_mode': 'one_shot',  # Instructs get_dataloader to use FullSequenceSMPLDataset
        }
    }

    print("\n--- Initializing FullSequenceSMPLDataset via DataLoader ---")
    loader = get_dataloader(cfg, mode='train')

    # # Visualize exact chunk length distributions output by the dataloader
    # plot_sequence_length_distributions(loader.dataset, out_dir=out_dir)
    
    # -------------------------------------------------------------
    # DIAGNOSTIC: DATASET SIZE & UNIQUE KEY VERIFICATION
    # -------------------------------------------------------------
    raw_dataset = loader.dataset
    unique_keys_in_indices = set(k for k, _ in raw_dataset.window_indices)
    
    print("\n" + "=" * 78)
    print("STANDALONE DATALOADER DIAGNOSTICS")
    print("=" * 78)
    print(f"Total raw keys in .npz:             {len(raw_dataset.pose_data)}")
    print(f"Keys retained in valid_keys:        {len(raw_dataset.valid_keys)}")
    print(f"Unique keys in window_indices:      {len(unique_keys_in_indices)}")
    print(f"Total samples (len(dataset)):       {len(raw_dataset)}")
    print(f"Total batches (len(loader)):        {len(loader)}")
    print(f"Expansion difference:               {len(raw_dataset) - len(unique_keys_in_indices)} duplicate window(s)")
    print("=" * 78 + "\n")

    # 2. Extract full unpadded sequences
    a_t_dict = {}
    labels_dict = {}

    print(f"Extracting angular speeds for {len(loader)} ground-truth sequences...")
    for batch in tqdm(loader, desc="Processing Sequences"):
        key = batch['key'][0]
        actual_len = batch['seq_len'][0].item()
        
        # Slicing [:actual_len] strips out any padding applied up to max_sequence_len
        pose_3d = batch['pose'][0, :actual_len].numpy()
        sev = batch['severity'][0].item()

        a_t_dict[key] = compute_angular_speed(pose_3d)
        labels_dict[key] = sev

    # 3. Execution
    # analyze_denominator_explosion(a_t_dict, labels_dict, out_dir)
    # analyze_spectral_bands(a_t_dict, labels_dict, out_dir)
    # analyze_padding_convergence(a_t_dict, labels_dict, out_dir)
    # analyze_cross_joint_distributions(a_t_dict, labels_dict, out_dir)
    # plot_all_joints_spectral_profiles(a_t_dict, labels_dict, out_dir)

    # analyze_q6_clinical_metrics(a_t_dict, labels_dict, out_dir)
    analyze_q6_clinical_metrics_upgraded(a_t_dict, labels_dict, out_dir, show_outliers=False)

    print("\n" + "=" * 78)
    print(f"All diagnostic analyses complete. Visualizations saved to:\n  {out_dir.resolve()}")
    print("=" * 78 + "\n")