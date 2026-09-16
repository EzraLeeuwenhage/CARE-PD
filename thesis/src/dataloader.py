import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random

class SMPLDataset(Dataset):
    """Base dataset class for Autoregressive Windowed Flow (AR-WG)."""
    def __init__(self, cfg, mode='train'):
        super().__init__()
        self.mode = mode
        self.cfg = cfg
            
        self.window_size = self.cfg['windowing']['total_window_size']
        self.prefix_length = self.cfg['windowing']['prefix_length']
        self.step_size = self.cfg['windowing']['step_size']
        
        # Extract minimum z travel setting with a fallback default of 0.0
        self.min_z_travel = self.cfg['windowing'].get('min_z_travel', 0.0)
        self.filter_z_travel = self.cfg['windowing'].get('filter_z_travel', True)
        
        # Determine split percentages
        eval_split = self.cfg['training'].get('eval_split', 0.1)
        test_split = self.cfg['training'].get('test_split', 0.2)

        # Load the SMPL representation irrespective of 3D or 6D representation
        with np.load(self.cfg['data']['smpl_path'], allow_pickle=True) as npz:
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
            all_keys=valid_pool_keys, mode=mode, eval_split=eval_split, test_split=test_split
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
                
                if not self.filter_z_travel or z_travel >= self.min_z_travel:
                    self.window_indices.append((key, start_idx))
                    chunk_counts[sev] += 1
                
        self._print_split_summary(
            mode=mode, seq_stats=seq_stats, chunk_counts=chunk_counts,
            total_inspected=total_chunks_inspected, discarded_keys=self.discarded_keys,
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
            keys_in_class.sort()
            n_cls = len(keys_in_class)
            n_test = max(1, int(n_cls * test_split)) if n_cls >= 1 else 0
            n_eval = max(1, int(n_cls * eval_split)) if n_cls >= 2 else 0
            
            train_end = max(0, n_cls - n_eval - n_test)
            eval_end = n_cls - n_test
            
            if mode == 'train': selected = keys_in_class[:train_end]
            elif mode == 'eval': selected = keys_in_class[train_end:eval_end]
            elif mode == 'test': selected = keys_in_class[eval_end:]
                
            stratified_keys.extend(selected)
            seq_stats[sev] = (len(selected), n_cls)
        return stratified_keys, seq_stats

    def _filter_valid_sequences(self, all_keys):
        """Pre-filters sequences to only include those that yield at least 1 valid chunk."""
        valid_seq_keys, discarded_short, discarded_no_travel = [], [], []
        for key in all_keys:
            num_frames = self.pose_data[key].shape[0]
            if num_frames < self.window_size:
                discarded_short.append(key)
                continue
            
            # make sure at least one chunk satisfies min_z_travel
            has_valid_chunk = False
            for start_idx in range(0, num_frames - self.window_size + 1, self.step_size):
                end_idx = start_idx + self.window_size
                start_z = self.trans_data[key][start_idx, 2]
                end_z = self.trans_data[key][end_idx - 1, 2]

                if not self.filter_z_travel or abs(end_z - start_z) >= self.min_z_travel:
                    has_valid_chunk = True
                    break
                    
            if has_valid_chunk: valid_seq_keys.append(key)
            else: discarded_no_travel.append(key)
        return valid_seq_keys, discarded_short, discarded_no_travel

    def _print_split_summary(self, mode, seq_stats, chunk_counts, total_inspected, discarded_keys, discarded_no_travel):
        print(f"\n{mode.upper()} SET (Windowed Chunks)")
        print(f" {'Severity':<10} | {'Sequences (Split / Total)':<26} | {'Valid Chunks':<12}")
        print("-" * 65)
        total_seq_selected = total_seq_all = total_chunks = 0
        for sev in sorted(seq_stats.keys()):
            sel_seq, all_seq = seq_stats[sev]
            chunks = chunk_counts.get(sev, 0)
            total_seq_selected += sel_seq
            total_seq_all += all_seq
            total_chunks += chunks
            print(f"  Class {sev:<4} | {f'{sel_seq} / {all_seq}':<26} | {chunks:>10,}")
            
        print("-" * 65)
        print(f" {'TOTAL':<10} | {f'{total_seq_selected} / {total_seq_all}':<26} | {total_chunks:>10,}")
        
        filtered_out = total_inspected - total_chunks
        if filtered_out > 0: print(f"  * Filtered out {filtered_out:,} chunks with < {self.min_z_travel}m Z-travel.")
        if discarded_no_travel: print(f"  * Excluded {len(discarded_no_travel)} entire sequence(s) (0 valid chunks).")
        if discarded_keys: print(f"  * Discarded {len(discarded_keys)} short sequence(s).")

    def __len__(self):
        return len(self.window_indices)

    def get_severity(self, idx):
        key = self.window_indices[idx][0]
        base_key = key.split('_down')[0] if '_down' in key else key
        return self.key_to_severity.get(base_key, 0)

    def __getitem__(self, idx):
        key, start_idx = self.window_indices[idx]
        end_idx = start_idx + self.window_size
        
        pose_window = self.pose_data[key][start_idx:end_idx]
        trans_window = self.trans_data[key][start_idx:end_idx]
        
        # Create M_cond mask (1 for prefix frames, 0 for target frames)
        cond_mask = np.zeros(self.window_size, dtype=bool)
        cond_mask[:self.prefix_length] = True
        
        # AR-WG windows are fixed length and never padded, so M_pad is strictly 0
        pad_mask = np.zeros(self.window_size, dtype=bool)
        
        severity_score = self.get_severity(idx)
        
        # Ensure parity with One-Shot OS-SG dataloader dictionary keys
        return {
            "pose": torch.tensor(pose_window, dtype=torch.float32),
            "trans": torch.tensor(trans_window, dtype=torch.float32),
            "pad_mask": torch.tensor(pad_mask, dtype=torch.bool),
            "cond_mask": torch.tensor(cond_mask, dtype=torch.bool),
            "seq_len": torch.tensor(self.window_size, dtype=torch.long),
            "severity": torch.tensor(severity_score, dtype=torch.long),
            "key": key
        }


class FullSequenceSMPLDataset(Dataset):
    """Dataset class for One-Shot Padded Flow (OS-SG) and AR-WG Evaluation.
    Yields zero-padded full sequences and respective boolean masks.
    Extracts multiple windows for sequences longer than max_len during training.
    """
    def __init__(self, cfg, mode='train'):
        super().__init__()
        self.mode = mode
        self.cfg = cfg
        self.max_len = cfg['windowing'].get('max_sequence_len', 200)
        self.prefix_length = cfg['windowing']['prefix_length']
        self.min_z_travel = cfg['windowing'].get('min_z_travel', 0.0)
        self.filter_z_travel = cfg['windowing'].get('filter_z_travel', True)
        
        # Stride for extracting multiple windows from long sequences.
        # Defaults to max_len (non-overlapping). Set to max_len // 2 for 50% overlap.
        self.stride = cfg['windowing'].get('full_seq_stride', self.max_len)

        eval_split = self.cfg['training'].get('eval_split', 0.1)
        test_split = self.cfg['training'].get('test_split', 0.2)

        with np.load(self.cfg['data']['smpl_path'], allow_pickle=True) as npz:
            raw_data = {k: np.array(v) for k, v in npz.items()}

        self.pose_data = {}
        self.trans_data = {}

        for key, tensor in raw_data.items():
            if not key.endswith('_trans'):
                self.pose_data[key] = tensor
                trans_key = f"{key}_trans"
                if trans_key in raw_data:
                    self.trans_data[key] = raw_data[trans_key]
                else:
                    raise KeyError(f"Missing paired translation data for pose sequence: '{key}'")

        patient_prefix = self.cfg['data'].get('patient_prefix')
        
        if not patient_prefix or str(patient_prefix).lower() == 'all':
            all_keys = list(self.pose_data.keys())
        else:
            search_str = f"{patient_prefix}__"
            all_keys = [k for k in self.pose_data.keys() if k.startswith(search_str)]
            
        if not all_keys:
            raise ValueError(f"No keys found for prefix: {patient_prefix}")

        with open(self.cfg['data']['severity_labels_path'], "r") as f:
            metadata = json.load(f)
            self.key_to_severity = metadata["key_to_severity"]

        valid_pool_keys, discarded_short, discarded_no_travel = self._filter_valid_sequences(all_keys)
        self.discarded_keys = discarded_short

        self.valid_keys, seq_stats = self._get_stratified_keys(
            all_keys=valid_pool_keys, mode=mode, eval_split=eval_split, test_split=test_split
        )

        # Build window index map
        self.window_indices = []
        chunk_counts = defaultdict(int)

        for key in self.valid_keys:
            num_frames = self.pose_data[key].shape[0]
            base_key = key.split('_down')[0] if '_down' in key else key
            sev = self.key_to_severity.get(base_key, 0)

            if self.mode == 'train' and num_frames > self.max_len:
                # Sliced window extraction across long sequences
                starts = list(range(0, num_frames - self.max_len + 1, self.stride))
                # Ensure the final frames are included if not aligned with stride
                last_start = num_frames - self.max_len
                if starts[-1] != last_start:
                    starts.append(last_start)

                for start_idx in starts:
                    self.window_indices.append((key, start_idx))
                    chunk_counts[sev] += 1
            else:
                # Single window: start at 0 (val/test, or training sequences <= max_len)
                self.window_indices.append((key, 0))
                chunk_counts[sev] += 1

        # Sort sequences by length for efficient batching during eval/test
        if self.mode != 'train':
            self.window_indices.sort(key=lambda item: self.pose_data[item[0]].shape[0])

        self._print_split_summary(
            mode=mode, seq_stats=seq_stats, chunk_counts=chunk_counts,
            total_inspected=len(all_keys), discarded_keys=self.discarded_keys,
            discarded_no_travel=discarded_no_travel
        )

    def _get_stratified_keys(self, all_keys, mode, eval_split, test_split):
        class_groups = defaultdict(list)
        for k in all_keys:
            base_k = k.split('_down')[0] if '_down' in k else k
            sev = self.key_to_severity.get(base_k, 0)
            class_groups[sev].append(k)

        stratified_keys = []
        seq_stats = {}
        for sev, keys_in_class in sorted(class_groups.items()):
            keys_in_class.sort()
            n_cls = len(keys_in_class)
            n_test = max(1, int(n_cls * test_split)) if n_cls >= 1 else 0
            n_eval = max(1, int(n_cls * eval_split)) if n_cls >= 2 else 0
            
            train_end = max(0, n_cls - n_eval - n_test)
            eval_end = n_cls - n_test
            
            if mode == 'train': selected = keys_in_class[:train_end]
            elif mode == 'eval': selected = keys_in_class[train_end:eval_end]
            elif mode == 'test': selected = keys_in_class[eval_end:]
                
            stratified_keys.extend(selected)
            seq_stats[sev] = (len(selected), n_cls)
        return stratified_keys, seq_stats

    def _filter_valid_sequences(self, all_keys):
        valid_seq_keys, discarded_short, discarded_no_travel = [], [], []
        for key in all_keys:
            num_frames = self.pose_data[key].shape[0]
            if num_frames <= self.prefix_length:
                discarded_short.append(key)
                continue
            
            # Require minimum z travel over the ENTIRE sequence
            start_z = self.trans_data[key][0, 2]
            end_z = self.trans_data[key][-1, 2]
            if not self.filter_z_travel or abs(end_z - start_z) >= self.min_z_travel:
                valid_seq_keys.append(key)
            else:
                discarded_no_travel.append(key)
        return valid_seq_keys, discarded_short, discarded_no_travel

    def _print_split_summary(self, mode, seq_stats, chunk_counts, total_inspected, discarded_keys, discarded_no_travel):
        print(f"\n{mode.upper()} SET (Full Sequences / Windows)")
        print(f" {'Severity':<10} | {'Sequences (Split / Total)':<26} | {'Samples/Epoch':<14}")
        print("-" * 58)
        total_seq_selected = total_seq_all = total_samples = 0
        for sev in sorted(seq_stats.keys()):
            sel_seq, all_seq = seq_stats[sev]
            samples = chunk_counts.get(sev, 0)
            total_seq_selected += sel_seq
            total_seq_all += all_seq
            total_samples += samples
            print(f"  Class {sev:<4} | {f'{sel_seq} / {all_seq}':<26} | {samples:>12,}")
            
        print("-" * 58)
        print(f" {'TOTAL':<10} | {f'{total_seq_selected} / {total_seq_all}':<26} | {total_samples:>12,}")
        
        if discarded_no_travel: print(f"  * Excluded {len(discarded_no_travel)} sequence(s) from pool (Total Z-travel < {self.min_z_travel}m).")
        if discarded_keys: print(f"  * Discarded {len(discarded_keys)} short sequence(s) (<= prefix length).")

    def __len__(self):
        return len(self.window_indices)

    def get_severity(self, idx):
        key = self.window_indices[idx][0]
        base_key = key.split('_down')[0] if '_down' in key else key
        return self.key_to_severity.get(base_key, 0)

    def __getitem__(self, idx):
        key, start_idx = self.window_indices[idx]
        raw_pose = self.pose_data[key]
        raw_trans = self.trans_data[key]

        pose = raw_pose[start_idx:start_idx + self.max_len]
        trans = raw_trans[start_idx:start_idx + self.max_len]
        T = pose.shape[0]
        pad_len = self.max_len - T

        if pad_len > 0:
            pose_pad = np.pad(pose, ((0, pad_len), (0, 0), (0, 0)), mode='constant')
            trans_pad = np.pad(trans, ((0, pad_len), (0, 0)), mode='constant')
            pad_mask = np.concatenate([np.zeros(T, dtype=bool), np.ones(pad_len, dtype=bool)])
        else:
            pose_pad = pose
            trans_pad = trans
            pad_mask = np.zeros(self.max_len, dtype=bool)

        cond_mask = np.zeros(self.max_len, dtype=bool)
        cond_frames = min(T, self.prefix_length)
        cond_mask[:cond_frames] = True

        severity = self.get_severity(idx)

        return {
            "pose": torch.tensor(pose_pad, dtype=torch.float32),
            "trans": torch.tensor(trans_pad, dtype=torch.float32),
            "pad_mask": torch.tensor(pad_mask, dtype=torch.bool),
            "cond_mask": torch.tensor(cond_mask, dtype=torch.bool),
            "seq_len": torch.tensor(T, dtype=torch.long),
            "severity": torch.tensor(severity, dtype=torch.long),
            "key": key
        }


class OverfitWrapper(Dataset):
    """Wraps any dataset to artificially repeat a single randomly selected sequence of a specific class."""
    def __init__(self, dataset, cfg):
        super().__init__()
        self.dataset = dataset
        target_sev = cfg['training'].get('overfit_severity_class', 0)
        
        valid_indices = [i for i in range(len(dataset)) if dataset.get_severity(i) == target_sev]
        if not valid_indices: 
            raise ValueError(f"No valid sequences found for severity class {target_sev}")
            
        seed = cfg['training'].get('overfit_seed', 42)
        rng = random.Random(seed)
        self.single_idx = rng.choice(valid_indices)
        self.dummy_epoch_size = cfg['training']['batch_size'] * 10
        
        if hasattr(dataset, 'window_indices'):
            info = dataset.window_indices[self.single_idx]
            print(f"\n[OVERFIT MODE] Locked to chunk -> {info[0]} (Start: {info[1]}) | Class: {target_sev} | Seed: {seed}")
        else:
            info = dataset.valid_keys[self.single_idx]
            print(f"\n[OVERFIT MODE] Locked to sequence -> {info} | Class: {target_sev} | Seed: {seed}")

    def __len__(self):
        return self.dummy_epoch_size

    def __getitem__(self, idx):
        return self.dataset[self.single_idx]


def get_dataloader(cfg, mode='train'):
    """Builds appropriate dataloader and routes depending on generative paradigm.
    """
    gen_mode = cfg['model'].get('generation_mode', 'ar_rollout')
    is_train = mode == 'train'
    is_overfit = cfg['training'].get('overfit_severity_class', -1) >= 0
    
    # OS-SG uses full sequence padding for all splits.
    # AR-WG needs windowed chunks for training, but full sequences for validation/testing
    if gen_mode == 'one_shot' or not is_train:
        dataset = FullSequenceSMPLDataset(cfg, mode=mode)
    else:
        dataset = SMPLDataset(cfg, mode=mode)
        
    if is_overfit:
        dataset = OverfitWrapper(dataset, cfg)
    
    return DataLoader(
        dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=cfg['training']['shuffle'] if is_train and not is_overfit else False,
        num_workers=cfg['training'].get('num_workers', 4),
        drop_last=is_train and not is_overfit
    )