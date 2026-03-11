# Dataset & Feature Documentation

## 1. Dataset Overview

This project uses the **PD-GaM gait dataset**, a video-based dataset designed for analyzing gait patterns associated with **Parkinson's Disease (PD)**.

| Attribute | Value |
|---|---|
| **Dataset Name** | PD-GaM |
| **Original Modality** | RGB Video |
| **Original FPS** | 25 FPS |
| **Normalized FPS** | 30 FPS |
| **# Subjects** | 30 |
| **# Walking Sequences** | 1701 |
| **Total Duration** | 186:22 (min:sec) |
| **Mean Age** | 54.1 ± 8.1 years |
| **Sex Distribution** | 56.7% male |
| **Clinical Annotation** | UPDRS-Gait score (0–3) |

### Clinical Annotation

Each walking sequence includes a corresponding gait severity score from the **Unified Parkinson's Disease Rating Scale (UPDRS)**.

| Score | Interpretation |
|---|---|
| **0** | Normal gait |
| **1** | Slight gait impairment |
| **2** | Moderate gait impairment |
| **3** | Severe gait impairment |

These scores allow the dataset to be used for:

- Clinical severity prediction
- Gait abnormality analysis
- Parkinsonian motion modeling

---

# 2. Data Normalization

Two normalization procedures were applied to ensure comparability across subjects.

## Frame Rate Normalization

All sequences were **resampled from 25 FPS to 30 FPS** to standardize temporal resolution across recordings.

## Spatial Normalization

All spatial coordinates are **normalized by the patient's leg length**.

This converts all spatial measurements into **dimensionless ratios relative to body size**.

### Example Interpretations

| Metric Value | Meaning |
|---|---|
| Walking speed = 0.8 | 0.8 leg-lengths per second |
| Step length = 0.5 | Step equals 50% of the patient's leg length |
| Foot height = 0.1 | Foot lifted 10% of leg length |

This normalization ensures:

- Comparability across **different patient heights**
- More meaningful **biomechanical interpretation**
- **Scale-invariant machine learning features**

---

# 3. Computed Gait Features

The project extracts a comprehensive set of **biomechanical and clinical gait features** from the pose sequences.

These features fall into two categories:

1. **Physical Realism Metrics**
2. **Clinical Parkinsonian Gait Metrics**

---

# 4. Physical Realism Features

These features evaluate whether the reconstructed or generated skeleton motion is **physically plausible and biomechanically consistent**.

They are independent of Parkinson's severity and instead measure **motion quality and realism**.

---

## 4.1 Bone Length Variance

### Computation
Variance of the Euclidean distance between connected joints (e.g., **hip → knee**) across all frames.

### Purpose
Ensures skeletal structure remains constant during motion.

### Interpretation

- Near zero → physically consistent skeleton  
- High values → distorted or unstable skeleton

### Units
leg-length^2


---

## 4.2 Jitter (Acceleration Magnitude)

### Computation
Mean magnitude of the **second derivative of joint positions** over time.

### Purpose
Measures high-frequency instability or vibration in the motion.

### Interpretation

- Low jitter → smooth motion  
- High jitter → glitchy or vibrating skeleton

### Units
leg-length / frames^2