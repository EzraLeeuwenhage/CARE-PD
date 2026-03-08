import pickle
from pprint import pprint
import numpy as np

with open("../assets/datasets/PD-GaM.pkl", "rb") as f:
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
    print(f"UPDRS Range for Patient 001: {min_u} to {max_u} (Diff: {max_u - min_u})")
else:
    print("UPDRS key not found in walk data. You might need to check p1_w1_data.keys().")


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