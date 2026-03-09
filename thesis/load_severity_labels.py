import pickle
import json
from pathlib import Path

def build_metadata_registry(pkl_path, output_json_path="pd_gam_labels.json"):
    """
    Extracts UPDRS_GAIT scores from the heavy .pkl file and saves them 
    into a lightweight JSON registry, using composite keys (patientID__walkID)
    to perfectly match the .npz format.
    """
    print(f"Loading heavy .pkl file from: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
        
    # Dictionary 1: Quick lookup for a specific sequence (O(1) time)
    key_to_severity = {}
    
    # Dictionary 2: Easy access to all sequences of a specific class
    severity_to_keys = {}

    for patient_id, patient_walks in data.items():
        if not isinstance(patient_walks, dict):
            continue
            
        for walk_id, walk_entry in patient_walks.items():
            if isinstance(walk_entry, dict) and 'UPDRS_GAIT' in walk_entry:
                score = int(walk_entry['UPDRS_GAIT'])
                
                # --- The Fix: Create the composite key to match the .npz ---
                composite_key = f"{patient_id}__{walk_id}"
                
                # Update Dictionary 1
                key_to_severity[composite_key] = score
                
                # Update Dictionary 2
                if score not in severity_to_keys:
                    severity_to_keys[score] = []
                severity_to_keys[score].append(composite_key)

    # Combine into one clean registry
    registry = {
        "key_to_severity": key_to_severity,
        "severity_to_keys": severity_to_keys
    }

    with open(output_json_path, "w") as f:
        json.dump(registry, f, indent=4)
        
    print(f"Successfully extracted metadata for {len(key_to_severity)} sequences.")
    print(f"Saved lightweight registry to: {output_json_path}")
    
    # Print a quick summary
    for score, keys in sorted(severity_to_keys.items()):
        print(f"  Severity Class {score}: {len(keys)} sequences")


if __name__ == "__main__":
    # Use pathlib to make paths robust if running from different directories
    SCRIPT_DIR = Path(__file__).parent.resolve()
    pkl_file = SCRIPT_DIR / ".." / "assets" / "datasets" / "PD-GaM.pkl"
    json_out = SCRIPT_DIR / "pd_gam_labels.json"
    
    build_metadata_registry(pkl_file, json_out)