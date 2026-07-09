# Dataset & Feature Documentation

## 1. Dataset Overview (PD-GaM)

A video-based gait dataset designed for analyzing **Parkinson's Disease (PD)** motion patterns.

| Attribute | Value |
|---|---|
| Original Modality | RGB Video |
| FPS (Normalized) | 30 FPS |
| # Subjects / Sequences | 30 / 1701 |
| Sex Distribution | 56.7% Male |
| Mean Age | 54.1 ± 8.1 years |

### Clinical Annotation (UPDRS-Gait)

Each sequence is labeled with a **Unified Parkinson's Disease Rating Scale (UPDRS) gait score**.

| Score | Interpretation |
|---|---|
| 0 | Normal gait |
| 1 | Slight gait impairment |
| 2 | Moderate gait impairment |
| 3 | Severe gait impairment |

### Data distribution

Contains 1701 motion sequences:
- 783 of score 0
- 635 of score 1
- 248 of score 2
- 35 of score 3

But walk with id "004__004-12-105182_wid04_5" (of score 3) is discarded because it is less than 30 frames (has 20).

Total sequences with > 300 frames: 82
Breakdown by Severity Class (UPDRS_GAIT):
- Class 0: 0 sequences
- Class 1: 4 sequences
- Class 2: 78 sequences
- Class 3: 0 sequences
