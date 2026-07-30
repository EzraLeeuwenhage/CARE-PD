import json
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

CONFIG_PATH = "thesis/configs/dataloader.yaml"

class SMPL6DDataset(Dataset):
    def __init__(self, config_path=CONFIG_PATH, mode='train'):
        super().__init__()
        self.mode = mode
        
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
            
        self.window_size = self.cfg['windowing']['total_window_size']
        self.prefix_length = self.cfg['windowing']['prefix_length']
        self.step_size = self.cfg['windowing']['step_size']
        
        # Extract minimum z travel setting with a fallback default of 0.0
        self.min_z_travel = self.cfg['windowing'].get('min_z_travel', 0.0)
        
        # Determine split percentages
        eval_split = self.cfg['training'].get('eval_split', 0.1)
        test_split = self.cfg['training'].get('test_split', 0.2)

        with np.load(self.cfg['data']['6d_smpl_path'], allow_pickle=True) as npz:
            raw_data = {k: np.array(v) for k, v in npz.items()}

        self.pose_data = {}
        self.trans_data = {}

        # Separate pose and translation data on suffix
        for key, tensor in raw_data.items():
            if not key.endswith('_trans'):
                self.pose_data[key] = tensor
                trans_key = f"{key}_trans"
                if trans_key in raw_data:
                    self.trans_data[key] = raw_data[trans_key]
                else:
                    raise KeyError(f"Missing paired translation data for pose sequence: '{key}'")

        # Check for specific patient prefix or load all data
        patient_prefix = self.cfg['data'].get('patient_prefix')
        
        if not patient_prefix or str(patient_prefix).lower() == 'all':
            all_keys = list(self.pose_data.keys())
        else:
            search_str = f"{patient_prefix}__"
            all_keys = [k for k in self.pose_data.keys() if k.startswith(search_str)]
            
        if not all_keys:
            raise ValueError(f"No keys found for prefix: {patient_prefix}")

        # Do stratified split on severity class
        with open(self.cfg['data']['severity_labels_path'], "r") as f:
            metadata = json.load(f)
            self.key_to_severity = metadata["key_to_severity"]

        # Pre-filter sequences so ONLY those that produce >= 1 valid chunk enter the split pool
        valid_pool_keys, discarded_short, discarded_no_travel = self._filter_valid_sequences(all_keys)
        self.discarded_keys = discarded_short

        self.valid_keys, seq_stats = self._get_stratified_keys(
            all_keys=valid_pool_keys, 
            mode=mode, 
            eval_split=eval_split, 
            test_split=test_split
        )

        # use sliding windows to build index map
        self.window_indices = []
        total_chunks_inspected = 0
        chunk_counts = defaultdict(int)
        
        for key in self.valid_keys:
            num_frames = self.pose_data[key].shape[0]
            base_key = key.split('_down')[0] if '_down' in key else key
            sev = self.key_to_severity.get(base_key, 0)

            for start_idx in range(0, num_frames - self.window_size + 1, self.step_size):
                total_chunks_inspected += 1
                end_idx = start_idx + self.window_size

                # Compute Z-distance travelled across this specific sequence chunk
                start_z = self.trans_data[key][start_idx, 2]
                end_z = self.trans_data[key][end_idx - 1, 2]
                z_travel = abs(end_z - start_z)
                
                if z_travel >= self.min_z_travel:
                    self.window_indices.append((key, start_idx))
                    chunk_counts[sev] += 1
                
        self._print_split_summary(
            mode=mode,
            seq_stats=seq_stats,
            chunk_counts=chunk_counts,
            total_inspected=total_chunks_inspected,
            discarded_keys=self.discarded_keys,
            discarded_no_travel=discarded_no_travel
        )

    def _get_stratified_keys(self, all_keys, mode, eval_split, test_split):
        """Deterministically splits sequence keys by clinical severity class."""
        from collections import defaultdict
        class_groups = defaultdict(list)
        
        for k in all_keys:
            base_k = k.split('_down')[0] if '_down' in k else k
            sev = self.key_to_severity.get(base_k, 0)
            class_groups[sev].append(k)

        stratified_keys = []
        seq_stats = {}
        
        for sev, keys_in_class in sorted(class_groups.items()):
            keys_in_class.sort()  # Ensure deterministic order within class
            
            n_cls = len(keys_in_class)
            n_test = max(1, int(n_cls * test_split)) if n_cls >= 1 else 0
            n_eval = max(1, int(n_cls * eval_split)) if n_cls >= 2 else 0
            
            train_end = max(0, n_cls - n_eval - n_test)
            eval_end = n_cls - n_test
            
            if mode == 'train':
                selected = keys_in_class[:train_end]
            elif mode == 'eval':
                selected = keys_in_class[train_end:eval_end]
            elif mode == 'test':
                selected = keys_in_class[eval_end:]
                
            stratified_keys.extend(selected)
            seq_stats[sev] = (len(selected), n_cls)
            
        return stratified_keys, seq_stats

    def _filter_valid_sequences(self, all_keys):
        """Pre-filters sequences to only include those that yield at least 1 valid chunk."""
        valid_seq_keys = []
        discarded_short = []
        discarded_no_travel = []
        
        for key in all_keys:
            num_frames = self.pose_data[key].shape[0]
            if num_frames < self.window_size:
                discarded_short.append(key)
                continue
            
            # Fast check: does at least ONE window satisfy min_z_travel?
            has_valid_chunk = False
            for start_idx in range(0, num_frames - self.window_size + 1, self.step_size):
                end_idx = start_idx + self.window_size
                start_z = self.trans_data[key][start_idx, 2]
                end_z = self.trans_data[key][end_idx - 1, 2]
                
                if abs(end_z - start_z) >= self.min_z_travel:
                    has_valid_chunk = True
                    break
                    
            if has_valid_chunk:
                valid_seq_keys.append(key)
            else:
                discarded_no_travel.append(key)
                
        return valid_seq_keys, discarded_short, discarded_no_travel

    def _print_split_summary(self, mode, seq_stats, chunk_counts, total_inspected, discarded_keys, discarded_no_travel):
        """Prints a clean, organized table of sequence and chunk counts per class."""
        print(f"\n{mode.upper()} SET")
        print(f" {'Severity':<10} | {'Sequences (Split / Total)':<26} | {'Valid Chunks':<12}")
        print("-" * 65)
        
        total_seq_selected = 0
        total_seq_all = 0
        total_chunks = 0
        
        for sev in sorted(seq_stats.keys()):
            sel_seq, all_seq = seq_stats[sev]
            chunks = chunk_counts.get(sev, 0)
            
            total_seq_selected += sel_seq
            total_seq_all += all_seq
            total_chunks += chunks
            
            seq_str = f"{sel_seq} / {all_seq}"
            print(f"  Class {sev:<4} | {seq_str:<26} | {chunks:>10,}")
            
        print("-" * 65)
        total_seq_str = f"{total_seq_selected} / {total_seq_all}"
        print(f" {'TOTAL':<10} | {total_seq_str:<26} | {total_chunks:>10,}")
        
        filtered_out = total_inspected - total_chunks
        if filtered_out > 0:
            print(f"  * Filtered out {filtered_out:,} individual chunks with < {self.min_z_travel}m Z-travel.")
        if discarded_no_travel:
            print(f"  * Excluded {len(discarded_no_travel)} entire sequence(s) from split pool (0 valid chunks >= {self.min_z_travel}m).")
        if discarded_keys:
            print(f"  * Discarded {len(discarded_keys)} short sequence(s) (< {self.window_size} frames): {discarded_keys}")

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


def get_dataloader(config_path=CONFIG_PATH, mode='train'):
    dataset = SMPL6DDataset(config_path, mode=mode)
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    is_train = mode == 'train'
    
    loader = DataLoader(
        dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=cfg['training']['shuffle'] if is_train else False,
        num_workers=cfg['training']['num_workers'],
        drop_last=is_train
    )
    return loader


if __name__ == "__main__":
    from tqdm import tqdm
    from pathlib import Path
    from collections import Counter
    import sys
    import os
    
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from thesis.utils.sixD2smpl import build_smpl_pkl_from_6d_smpl
    
    config_path = CONFIG_PATH
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    output_dir = Path("thesis/data/processed/ground_truth_chunks")
    smpl_output_path = output_dir / "SMPL" / "ground_truth.pkl"
    labels_output_path = output_dir / "h36m" / "gt_labels.json"
    
    print(f"--- Generating Ground Truth Chunk Dataset ---")
    print(f"Config path:   {config_path}")
    print(f"Output SMPL:   {smpl_output_path}")
    print(f"Output Labels: {labels_output_path}")
    
    dataset = SMPL6DDataset(config_path)
    
    # Instantiate a custom DataLoader that guarantees NO shuffling and NO dropped data
    loader = DataLoader(
        dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,   
        drop_last=False, 
        num_workers=cfg['training']['num_workers']
    )
    
    all_gt_pose = []
    all_gt_trans = []
    all_severities = []
    
    for prefix, target, severity in tqdm(loader, desc="Extracting Chunks"):
        # Concatenate prefix and target to restore the full continuous chunk
        gt_pose = torch.cat([prefix['pose'], target['pose']], dim=1)
        gt_trans = torch.cat([prefix['trans'], target['trans']], dim=1)
        
        all_gt_pose.append(gt_pose)
        all_gt_trans.append(gt_trans)
        all_severities.extend(severity.tolist())
        
    full_pose = torch.cat(all_gt_pose, dim=0)
    full_trans = torch.cat(all_gt_trans, dim=0)
    
    print("\nSaving chunked data to SMPL .pkl format...")
    smpl_output_path.parent.mkdir(parents=True, exist_ok=True)
    build_smpl_pkl_from_6d_smpl(
        generated_pose_6d=full_pose, 
        generated_trans=full_trans, 
        output_filepath=str(smpl_output_path), 
        subject_id="GT", 
        walk_prefix="gt"
    )
    
    print("Saving severity labels to JSON...")
    labels_output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_dict = {"key_to_severity": {}}
    for i, sev in enumerate(all_severities):
        labels_dict["key_to_severity"][f"GT__gt_{i:03d}"] = sev
        
    with open(labels_output_path, 'w') as f:
        json.dump(labels_dict, f, indent=4)
    
    print("\nGround Truth Chunk Extraction Complete!")
    print(f"Total 60-frame chunks extracted: {full_pose.shape[0]}")
    
    sev_counts = Counter(all_severities)
    for severity in sorted(sev_counts.keys()):
        print(f"  -> Class '{severity}': {sev_counts[severity]} chunks processed.")