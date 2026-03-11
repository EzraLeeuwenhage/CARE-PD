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

---

# 2. Data Normalization & Units

To ensure comparability across different patients, the data undergoes specific normalization procedures.

### Temporal Normalization

All sequences are resampled to **exactly 30 frames per second (FPS)**.

### Spatial Normalization (Dimensionless Ratios)

The dataset explicitly **normalizes 3D spatial coordinates by the subject's leg length**.

Therefore, a value of **0.5 represents exactly 50% of the patient's leg length**.

Example interpretation:

| Metric Value | Meaning |
|---|---|
| Walking Speed = 0.8 | 0.8 leg-lengths per second |
| Step Length = 0.5 | Step equals 50% of leg length |
| Foot Height = 0.1 | Foot lifted 10% of leg length |

### Upper-Body Exception

During feature extraction, **local normalizations for Stooped Posture and Arm Swing were intentionally removed**.

These features operate directly on the **native pre-normalized dataset units**.

---

# 3. Physical Realism Features

These metrics evaluate whether generated motion obeys **physical and biomechanical constraints**, independent of Parkinsonian symptoms.

---

## 3.1 Structural Constancy & Smoothness

### Bone Length Variance

**Computation**

Calculates the 3D Euclidean distance between connected joints (e.g., hip to knee) at every frame.  
The statistical variance of this distance is computed across the sequence and averaged across major bones.

**Interpretation**

Measures whether the skeleton maintains structural rigidity.

- Low values indicate physically consistent bone lengths
- High values indicate stretching or compression artifacts

**Units**

Leg-lengths squared

---

### Jitter (Acceleration Magnitude)

**Computation**

Computes the difference in joint position between consecutive frames (velocity), then the difference between velocities (acceleration).  
The mean magnitude is calculated across all frames.

**Interpretation**

Measures high-frequency motion instability.

- Low jitter indicates smooth motion
- High jitter indicates vibration or animation artifacts

**Units**

Leg-lengths per frame squared

---

### Mean Joint Jerk (rate of change of acceleration)

**Computation**

Calculates the third derivative of joint position (rate of change of acceleration) and averages its magnitude across the sequence.

**Interpretation**

Measures biomechanical motion smoothness.

Human motor control naturally minimizes jerk.

- Low jerk indicates natural movement
- High jerk indicates unnatural motion dynamics

**Units**

Leg-lengths per frame cubed

---

## 3.2 Environment Interaction

### Floor Consistency (Heel Strike Y-Range)

**Computation**

Detects each heel strike event and records the vertical (Y-axis) coordinate of the planted ankle.  
The peak-to-peak range of these floor heights is computed across the entire sequence.

**Interpretation**

Measures whether the character is walking on a consistent ground plane.

Large values indicate:

- Slanted ground
- Floating feet
- Foot-ground penetration

**Units**

Leg-lengths

---

### Mean Foot Skating Velocity

**Computation**

At the frame immediately before each heel strike, the planted ankle is identified.  
Its horizontal velocity (X and Z axes) is computed and multiplied by FPS to convert to seconds.  
Velocities are averaged across all steps.

**Interpretation**

Quantifies the common animation artifact known as **foot skating** or **moonwalking**.

A perfect motion would have **zero skating velocity**.

**Units**

Leg-lengths per second

---

# 4. Clinical Parkinsonian Gait Metrics

These metrics capture **pathological biomechanical patterns associated with Parkinson's Disease**.

---

## 4.1 Temporal & Pace Features

### Cadence

**Computation**

Counts total detected heel strikes and divides by the sequence duration in minutes.

**Interpretation**

Measures walking rhythm in **steps per minute**.

Parkinsonian gait often shows **festinating cadence**, where steps become abnormally rapid.

**Units**

Steps per minute

---

### Walking Speed

**Computation**

Calculates the straight-line displacement of the pelvis from the first to the last detected heel strike and divides by elapsed time.

**Interpretation**

Measures overall forward locomotion.

Lower speeds may indicate:

- Bradykinesia
- Reduced propulsion

**Units**

Leg-lengths per second

---

### Step Time (Mean and Standard Deviation)

**Computation**

Measures the time interval between consecutive heel strikes.

**Interpretation**

- Mean step time describes walking pace
- Standard deviation captures irregular stepping patterns

High variability may indicate:

- Freezing of gait
- Arrhythmic walking

**Units**

Seconds

---

## 4.2 Spatial Kinematics

### Step Length (Mean and Standard Deviation)

**Computation**

At each heel strike, computes the absolute distance between ankles along the **Z-axis (forward direction)**.

**Interpretation**

Short step lengths are a defining characteristic of **Parkinsonian shuffling gait**.

**Units**

Leg-lengths

---

### Step Width (Mean and Standard Deviation)

**Computation**

At each heel strike, computes the absolute distance between ankles along the **X-axis (side-to-side direction)**.

**Interpretation**

A wider base of support indicates **postural instability**.

**Units**

Leg-lengths

---

### Foot Lifting (Clearance)

**Computation**

For each ankle, finds the highest and lowest vertical positions across the sequence and computes their difference.  
Left and right ankle ranges are then averaged.

**Interpretation**

Low foot lifting indicates:

- Flat-footed walking
- Shuffling gait
- Reduced toe clearance

**Units**

Leg-lengths

---

## 4.3 Dynamic Stability & Upper-Body Posture

### Estimated Margin of Stability (eMoS - Min and Std)

**Computation**

The **Extrapolated Center of Mass (XCoM)** is estimated using pelvis horizontal position and velocity:

XCoM = x + v / omega_0

The lateral boundary of the base of support is approximated using the planted ankle position.  
The margin of stability is the distance between the XCoM and this boundary.

**Interpretation**

Measures **dynamic lateral balance**.

- Large positive values indicate stable walking
- Values near zero indicate the subject is near loss of balance

**Units**

Leg-lengths

---

### GaitGen Stooped Posture

**Computation**

Calculates the average vertical distance between the neck and pelvis across all frames.

This feature is intentionally **not normalized by leg length**.

**Interpretation**

Quantifies the severity of the characteristic **forward-hunched posture** in advanced Parkinson's disease.

**Units**

Native spatial units

---

### GaitGen Arm Swing

**Computation**

For each arm, calculates the 3D distance between wrist and shoulder at every frame.  
The peak-to-peak range of this distance is computed for each arm independently, and the **minimum swing range** is selected.

This feature is intentionally **not normalized by leg length**.

**Interpretation**

Penalizes asymmetric arm motion typical of Parkinsonian bradykinesia, where one arm may remain rigid while the other swings normally.

**Units**

Native spatial units

---

### GaitGen Joint Variances (Expected Motion Energy)

**Computation**

For each joint, the temporal mean position is calculated.  
The variance of the joint's position relative to its mean is then computed across all frames.

**Interpretation**

Represents the **distribution of motion energy across the skeleton** and helps identify rigidity or reduced joint activity.

**Units**

Native spatial units squared



# 5. Intentionally Avoided Metrics

Some metrics have been purposefully avoided, either because they do not fit the data (format) as given, or they were deemed superfluous or uninformative. These include:

### Floor penetration
This metric counts the number of times the floor has been visibly penetrated by the foot. As generated motion does not respect physical impossibilities, this is normally a good measure of the realism of the generated data. 

However, from inspecting the data it quickly became apparent that the imagined floor was often not aligned with the z-axis. Furthermore, patients many times seemed to be floating increasingly more above the z-axis as the sequence progressed. 

For these reasons it was deemed an impractical measure for this particular dataset.