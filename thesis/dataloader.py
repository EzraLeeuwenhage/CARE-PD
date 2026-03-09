import os

import numpy as np
import torch

def load_patient_walks(npz_path, patient_prefix="003"):
    """
    Step 1 & 2: Load the .npz file, filter for a specific patient,
    and format all their walking sequences.
    """
    print(f"Loading data from: {npz_path}")
    data_dict = np.load(npz_path, allow_pickle=True)
    
    # Filter keys that start with the patient prefix (e.g., "003" or "003__")
    # We add the "__" to ensure we don't accidentally grab "0031__"
    search_str = f"{patient_prefix}__"
    patient_keys = [key for key in data_dict.files if key.startswith(search_str)]
    
    if not patient_keys:
        print(f"No walks found for patient prefix: {patient_prefix}")
        return []
        
    print(f"Found {len(patient_keys)} walks for patient {patient_prefix}.")
    
    formatted_walks = []
    
    for key in patient_keys:
        walk_data = data_dict[key]
        
        # Convert to float32 PyTorch tensor
        motion_tensor = torch.tensor(walk_data, dtype=torch.float32)
        
        # Expected shape from CARE-PD preprocessing is usually (T, V, C)
        # We need (C, T, V) for the GNN. 
        # C=3 (xyz), T=frames, V=17 (joints)
        if motion_tensor.shape[-1] == 3: 
            # Permute from (T, V, C) to (C, T, V)
            gnn_input = motion_tensor.permute(2, 0, 1)
        else:
            gnn_input = motion_tensor

        # Add a batch dimension so it becomes (1, C, T, V)
        gnn_input = gnn_input.unsqueeze(0)
        
        formatted_walks.append({
            'clip_id': key,
            'tensor': gnn_input,
            'frames': gnn_input.shape[2] # T dimension
        })
        
    # T_values = [item['frames'] for item in formatted_walks]

    # T_array = np.array(T_values)

    # print("===== T Dimension Statistics =====")
    # print(f"Count:      {len(T_array)}")
    # print(f"Min:        {T_array.min()}")
    # print(f"Max:        {T_array.max()}")
    # print(f"Mean:       {T_array.mean():.2f}")
    # print(f"Median:     {np.median(T_array):.2f}")
    # print(f"Std Dev:    {T_array.std():.2f}")
    # print(f"25th %ile:  {np.percentile(T_array, 25):.2f}")
    # print(f"75th %ile:  {np.percentile(T_array, 75):.2f}")

    return formatted_walks

def get_h36m_edge_index():
    """
    Defines the standard 17-joint Human3.6M skeleton edges.
    This acts as the graph structure for your GNN.
    """
    # H36M 17-joint standard connections:
    # 0:Pelvis, 1:RHip, 2:RKnee, 3:RAnkle, 4:LHip, 5:LKnee, 6:LAnkle, 
    # 7:Spine, 8:Neck, 9:Nose, 10:Head, 11:LShoulder, 12:LElbow, 13:LWrist, 
    # 14:RShoulder, 15:RElbow, 16:RWrist
    
    bones = [
        (0, 1), (1, 2), (2, 3),       # Right leg
        (0, 4), (4, 5), (5, 6),       # Left leg
        (0, 7), (7, 8),               # Spine to Neck
        (8, 9), (9, 10),              # Neck to Head
        (8, 11), (11, 12), (12, 13),  # Left arm
        (8, 14), (14, 15), (15, 16)   # Right arm
    ]
    
    # For PyTorch Geometric or standard GCNs, edges are usually formatted 
    # as a (2, num_edges) tensor. We make them bidirectional.
    src = [edge[0] for edge in bones] + [edge[1] for edge in bones]
    dst = [edge[1] for edge in bones] + [edge[0] for edge in bones]
    
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index

def create_fixed_length_windows(walks_list, window_size=60, step_size=30):
    """
    Takes a list of variable-length walk dictionaries and slices them 
    into fixed-length overlapping chunks.
    """
    print(f"Chunking {len(walks_list)} walks into {window_size}-frame windows...")
    fixed_chunks = []
    
    for walk in walks_list:
        tensor = walk['tensor'] # Shape: (1, C, T, V) -> e.g., (1, 3, T, 17)
        T = walk['frames']
        
        # Safety check: skip if sequence is somehow shorter than window size
        if T < window_size:
            continue
            
        # Slide a window across the Time dimension (dim=2)
        for start_idx in range(0, T - window_size + 1, step_size):
            end_idx = start_idx + window_size
            
            # Slice the time dimension
            chunk = tensor[:, :, start_idx:end_idx, :] # Shape: (1, 3, 60, 17)
            fixed_chunks.append(chunk)
            
    # Stack all individual chunks along the batch dimension (dim=0)
    if len(fixed_chunks) > 0:
        batched_data = torch.cat(fixed_chunks, dim=0) 
    else:
        print("Error: No chunks generated. Check your window_size and data.")
        return None
        
    print(f"Successfully generated {len(fixed_chunks)} fixed-length chunks.")
    print(f"Final Batched Tensor Shape (N, C, T, V): {batched_data.shape}")
    
    return batched_data


if __name__ == "__main__":
    npz_path = "../assets/datasets/h36m/PD-GaM/h36m_3d_world_floorXZZplus_30f_or_longer.npz"
    
    # 1. Get the target data (x_1 for your flow matching)
    patient_003_data = load_patient_walks(npz_path, patient_prefix="003")

    x_1 = create_fixed_length_windows(
        patient_003_data, 
        window_size=60, 
        step_size=20
    )
    # 2. Get the graph connectivity 
    edge_index = get_h36m_edge_index()
    
    print(f"Edge Index Shape: {edge_index.shape}")