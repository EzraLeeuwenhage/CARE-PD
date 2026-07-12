import json
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

CONFIG_PATH = "thesis/configs/dataloader.yaml"

class SMPL6DDataset(Dataset):
    def __init__(self, config_path=CONFIG_PATH):
        super().__init__()
        
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
            
        self.window_size = self.cfg['windowing']['total_window_size']
        self.prefix_length = self.cfg['windowing']['prefix_length']
        self.step_size = self.cfg['windowing']['step_size']
        
        # Extract minimum z travel setting with a fallback default of 0.0
        self.min_z_travel = self.cfg['windowing'].get('min_z_travel', 0.0)

        with np.load(self.cfg['data']['6d_smpl_path'], allow_pickle=True) as npz:
            self.pose_data = {k: np.array(v) for k, v in npz.items()}

        with np.load(self.cfg['data']['smpl_translations_path'], allow_pickle=True) as npz:
            self.trans_data = {k: np.array(v) for k, v in npz.items()}

        # Check for specific patient prefix or load all data
        patient_prefix = self.cfg['data'].get('patient_prefix')
        
        if not patient_prefix or str(patient_prefix).lower() == 'all':
            self.valid_keys = list(self.pose_data.keys())
            print(f"Using full dataset. Found {len(self.valid_keys)} total valid keys.")
        else:
            search_str = f"{patient_prefix}__"
            self.valid_keys = [k for k in self.pose_data.keys() if k.startswith(search_str)]
            print(f"Found {len(self.valid_keys)} valid keys for prefix '{patient_prefix}' in the dataset.")
            
        if not self.valid_keys:
            raise ValueError(f"No keys found for prefix: {patient_prefix}")
        
        # use sliding windows to build index map
        self.window_indices = []
        self.discarded_keys = []
        
        # track statistics on filtered out chunks
        total_chunks_inspected = 0
        
        for key in self.valid_keys:
            num_frames = self.pose_data[key].shape[0]
            
            if num_frames >= self.window_size:
                for start_idx in range(0, num_frames - self.window_size + 1, self.step_size):
                    total_chunks_inspected += 1
                    end_idx = start_idx + self.window_size
                    
                    # Compute Z-distance travelled across this specific sequence chunk
                    start_z = self.trans_data[key][start_idx, 2]
                    end_z = self.trans_data[key][end_idx - 1, 2]
                    z_travel = abs(end_z - start_z)
                    
                    if z_travel >= self.min_z_travel:
                        self.window_indices.append((key, start_idx))
            else:
                self.discarded_keys.append(key)
                    
        print(f"Dataset initialized with {len(self.window_indices)} sequence chunks (Filtered out \
              {total_chunks_inspected - len(self.window_indices)} chunks with < {self.min_z_travel}m Z-travel).")
        print(f"Discarded {len(self.discarded_keys)} keys due to insufficient sequence length: {self.discarded_keys}")

        # load severity labels registry
        with open(self.cfg['data']['severity_labels_path'], "r") as f:
            metadata = json.load(f)
            self.key_to_severity = metadata["key_to_severity"]

    def get_sample_metadata(self, idx):
        """Returns the source sequence key and frame offset for evaluation tracking."""
        key, start_idx = self.window_indices[idx]
        end_idx = start_idx + self.window_size
        start_z = self.trans_data[key][start_idx, 2]
        end_z = self.trans_data[key][end_idx - 1, 2]
        return {
            "sequence_key": key,
            "start_frame": start_idx,
            "end_frame": end_idx,
            "z_travel_meters": abs(end_z - start_z)
        }

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        key, start_idx = self.window_indices[idx]
        end_idx = start_idx + self.window_size
        
        # Sliced to 24 joints here to drop the empty 25th padding joint
        pose_window = torch.tensor(self.pose_data[key][start_idx:end_idx, :24, :], dtype=torch.float32)
        trans_window = torch.tensor(self.trans_data[key][start_idx:end_idx], dtype=torch.float32)
        
        # split as prefix and target sequence        
        prefix = {
            'pose': pose_window[:self.prefix_length],
            'trans': trans_window[:self.prefix_length]
        }
        
        target = {
            'pose': pose_window[self.prefix_length:],
            'trans': trans_window[self.prefix_length:]
        }

        # extract severity score from registry
        base_key = key.split('_down')[0] if '_down' in key else key
        severity_score = self.key_to_severity[base_key]
        
        return prefix, target, torch.tensor(severity_score, dtype=torch.long)


def get_dataloader(config_path=CONFIG_PATH):
    dataset = SMPL6DDataset(config_path)
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    loader = DataLoader(
        dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=cfg['training']['shuffle'],
        num_workers=cfg['training']['num_workers'],
        drop_last=True
    )
    return loader