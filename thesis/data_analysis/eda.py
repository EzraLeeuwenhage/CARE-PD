import pickle
from pprint import pprint
import numpy as np
from pathlib import Path

""" 
Base SMPL in pickle file format
 """

with open("thesis/data/raw/PD-GaM/PD-GaM.pkl", "rb") as f:
    data = pickle.load(f)

# pprint(data, depth=1)

p1_data = data['001']
# pprint(p1_data, depth=1)

p1_w1_data = p1_data['001-12-104704_wid01_0']
# pprint(p1_w1_data, depth=2)

p1_w1_pose_data = p1_w1_data['pose']
# print(len(p1_w1_pose_data))

p1_w1_translation_data = p1_w1_data['trans']
# print(len(p1_w1_translation_data))

def inspect_smpl_dimensions(data_dict, name="Data Group"):
    print(f"\n--- Dimensions for {name} ---")
    print(f"{'Key':<15} | {'Shape':<20} | {'Type'}")
    print("-" * 50)
    
    for key, value in data_dict.items():
        # Check if it's a numpy array or list to get shape/length
        if isinstance(value, (np.ndarray, list)):
            shape = np.shape(value)
            dtype = type(value).__name__
            print(f"{key:<15} | {str(shape):<20} | {dtype}")
        elif isinstance(value, int):
            print(f"{key:<15} | {str(value):<20} | int")
        else:
            print(f"{key:<15} | {'N/A':<20} | {type(value).__name__}")

# Usage with your specific data
# inspect_smpl_dimensions(p1_w1_data, name="Participant 1, Walk 1")

# Get all walk IDs for the first patient ('003')
walk_ids = data['003'].keys()

# pprint(walk_ids)

# pprint(data['001']['001-12-104704_wid01_0'].keys())

updrs_scores = []

for wid in walk_ids:
    walk_entry = data['003'][wid]
    # Check if 'updrs' is a key in the walk data
    if 'UPDRS_GAIT' in walk_entry:
        updrs_scores.append(walk_entry['UPDRS_GAIT'])

if updrs_scores:
    min_u = min(updrs_scores)
    max_u = max(updrs_scores)
    print(f"\nUPDRS Range for Patient 003: {min_u} to {max_u} (Diff: {max_u - min_u})")
else:
    print("\nUPDRS key not found in walk data. You might need to check p1_w1_data.keys().")


def compare_walk_suffixes(patient_data, base_id):
    """
    Compares two trials (suffix _0 and _1) for a specific walk ID.
    Example base_id: '001-12-104704_wid01'
    """
    key0 = f"{base_id}_0"
    key1 = f"{base_id}_1"

    pprint(patient_data[key0].keys())
    
    if key0 in patient_data and key1 in patient_data:
        print(f"\n--- Comparing {key0} vs {key1} ---")
        print(f"{'Metric':<20} | {'Trial _0':<15} | {'Trial _1':<15}")
        print("-" * 55)
        
        # Compare shapes of 'pose' and 'trans'
        for attr in ['pose', 'trans']:
            shape0 = np.shape(patient_data[key0][attr])
            shape1 = np.shape(patient_data[key1][attr])
            print(f"{attr + ' shape':<20} | {str(shape0):<15} | {str(shape1):<15}")
            
        # Check if UPDRS is different between them
        u0 = patient_data[key0].get('UPDRS_GAIT', 'N/A')
        u1 = patient_data[key1].get('UPDRS_GAIT', 'N/A')
        print(f"{'UPDRS Score':<20} | {u0:<15} | {u1:<15}")
    else:
        print(f"One or both keys ({key0}, {key1}) not found in this patient's data.")

# Example usage for the first walk you accessed
# We strip the '_0' to get the base ID
base_walk_id = '001-12-104704_wid06' 
compare_walk_suffixes(data['001'], base_walk_id)


""" 
H36M format
 """

# new script for inspection .npz file format
npz_path = "thesis/data/raw/PD-GaM/h36m/h36m_3d_world_floorXZZplus_30f_or_longer.npz"

def load_patient_walks(filepath, patient_prefix="003"):
    """
    Loads walking sequences for a specific patient from a flat .npz file.
    Returns a list of dictionaries containing the clip_id, tensor, and frame count.
    """
    print(f"Loading data from: {filepath}...")
    
    try:
        data = np.load(filepath, allow_pickle=True)
        # If the npz contains a pickled dictionary, it usually lives inside 'arr_0'
        if hasattr(data, 'files') and len(data.files) == 1 and data.files[0] == 'arr_0':
            data = data['arr_0'].item()
    except Exception as e:
        print(f"Error loading file: {e}")
        return []
    
    print(f"Total sequences in dataset: {len(data)}")

    patient_walks = []
    
    # Iterate through the flat dictionary and filter by patient prefix
    for clip_id, tensor in data.items():
        # Handle cases where the key might be "003__003-12..." or just "003-12..."
        if clip_id.startswith(f"{patient_prefix}__") or clip_id.startswith(patient_prefix):
            patient_walks.append({
                'clip_id': clip_id,
                'tensor': tensor,
                'frames': tensor.shape[0] if tensor is not None else 0
            })
            
    print(f"Found {len(patient_walks)} walks for patient '{patient_prefix}'.\n")
    return patient_walks

patient_003_data = load_patient_walks(npz_path, patient_prefix="003")

for i in range(len(patient_003_data)):
    clip = patient_003_data[i]
    print(f"  Clip ID: {clip['clip_id']:<30} | Frames: {clip['frames']:<4} | Tensor Shape: {clip['tensor'].shape}")


""" 
6D SMPL npz file format (split in pose and translation data files)
 """
def load_smpl_walks(pose_filepath, trans_filepath, patient_prefix="003"):
    """
    Loads SMPL pose and translation sequences for a specific patient.
    Returns a list of dictionaries containing the clip_id, pose tensor, 
    translation tensor, and frame count.
    """
    print(f"Loading pose data from: {pose_filepath}...")
    print(f"Loading translation data from: {trans_filepath}...")
    
    # Helper to unpack npz arrays safely
    def safe_load(filepath):
        try:
            data = np.load(filepath, allow_pickle=True)
            if hasattr(data, 'files') and len(data.files) == 1 and data.files[0] == 'arr_0':
                return data['arr_0'].item()
            return data
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    pose_data = safe_load(pose_filepath)
    trans_data = safe_load(trans_filepath)

    if pose_data is None or trans_data is None:
        return []
    
    print(f"Total pose sequences: {len(pose_data)}")
    print(f"Total translation sequences: {len(trans_data)}")

    patient_walks = []
    
    # Iterate through the pose dictionary and filter by patient prefix
    for clip_id, pose_tensor in pose_data.items():
        # Handle cases where the key might be "003__..." or "003-..."
        if clip_id.startswith(f"{patient_prefix}__") or clip_id.startswith(patient_prefix):
            
            # Fetch the matching translation data from the second file
            trans_tensor = trans_data.get(clip_id)
            
            # Extract frame counts
            frames = pose_tensor.shape[0] if pose_tensor is not None else 0
            trans_frames = trans_tensor.shape[0] if trans_tensor is not None else 0
            
            # Sanity check: Ensure poses and translations are synchronized
            if frames != trans_frames:
                print(f"  [!] Warning: Frame mismatch for {clip_id} (Pose: {frames}, Trans: {trans_frames})")

            patient_walks.append({
                'clip_id': clip_id,
                'pose': pose_tensor,
                'trans': trans_tensor,
                'frames': frames
            })
            
    print(f"\nFound {len(patient_walks)} walks for patient '{patient_prefix}'.\n")
    return patient_walks

print(f"\n--- Loading 6D SMPL Data for Patient 003 ---")
pose_path = "thesis/data/raw/PD-GaM/6D_SMPL/6D_SMPL_30f_or_longer.npz"
trans_path = "thesis/data/raw/PD-GaM/6D_SMPL/6D_SMPL_30f_or_longer_translations.npz"
patient_003_data = load_smpl_walks(pose_path, trans_path, patient_prefix="003")

if patient_003_data:
    print(f"Data format keys per walk: {list(patient_003_data[0].keys())}")
    print("-" * 90)
    # Formatted table header
    print(f"{'Clip ID':<30} | {'Frames':<8} | {'Pose Shape (Rotations)':<22} | {'Trans Shape (Global)'}")
    print("-" * 90)
    
    # Loop through and print the details neatly
    for i in range(len(patient_003_data)):
        clip = patient_003_data[i]
        
        # Format shapes as strings for clean table alignment
        pose_shape = str(clip['pose'].shape) if clip['pose'] is not None else "None"
        trans_shape = str(clip['trans'].shape) if clip['trans'] is not None else "None"
        
        print(f"  {clip['clip_id']:<28} | {clip['frames']:<8} | {pose_shape:<22} | {trans_shape}")
else:
    print("No sequences found. Please double-check the filepaths and patient prefix.")

def analyze_long_sequences(npz_filepath, raw_pkl_data, frame_threshold=300):
    """
    Finds sequences longer than a specific threshold and cross-references 
    the raw .pkl data to tally them by UPDRS_GAIT severity class.
    """
    print(f"\n--- Analyzing sequences with > {frame_threshold} frames ---")
    
    # Load the flat H36M dictionary
    try:
        npz_data = np.load(npz_filepath, allow_pickle=True)
        if hasattr(npz_data, 'files') and len(npz_data.files) == 1 and npz_data.files[0] == 'arr_0':
            npz_data = npz_data['arr_0'].item()
    except Exception as e:
        print(f"Error loading {npz_filepath}: {e}")
        return

    long_seqs = []
    # Initialize counters for the 4 severity classes (plus an unknown fallback)
    severity_counts = {0: 0, 1: 0, 2: 0, 3: 0, "Unknown": 0}

    for clip_id, tensor in npz_data.items():
        num_frames = tensor.shape[0]
        
        if num_frames > frame_threshold:
            # The clip_id is formatted as "PatientID__WalkID"
            parts = clip_id.split('__')
            severity = "Unknown"
            
            if len(parts) == 2:
                patient_id = parts[0]
                walk_id = parts[1]
                
                # Dig into the raw pickle data to find the severity score
                if patient_id in raw_pkl_data and walk_id in raw_pkl_data[patient_id]:
                    walk_entry = raw_pkl_data[patient_id][walk_id]
                    if 'UPDRS_GAIT' in walk_entry:
                        severity = int(walk_entry['UPDRS_GAIT'])
            
            long_seqs.append((clip_id, num_frames, severity))
            
            # Increment the corresponding severity bucket
            if severity in severity_counts:
                severity_counts[severity] += 1
            else:
                severity_counts["Unknown"] += 1

    # Print Summary Output
    print(f"Total sequences with > {frame_threshold} frames: {len(long_seqs)}")
    print("\nBreakdown by Severity Class (UPDRS_GAIT):")
    for sev in [0, 1, 2, 3, "Unknown"]:
        count = severity_counts[sev]
        if count > 0 or sev != "Unknown":
            print(f"  Class {sev}: {count} sequences")
            
    # Print Detailed Table sorted by length
    print("\nList of Long Sequences:")
    print(f"{'Clip ID':<35} | {'Frames':<8} | {'Severity'}")
    print("-" * 60)
    
    long_seqs.sort(key=lambda x: x[1], reverse=True)
    for cid, frames, sev in long_seqs:
        print(f"{cid:<35} | {frames:<8} | {sev}")

# Execute the analysis using your already-loaded variables
analyze_long_sequences(npz_path, data, frame_threshold=300)