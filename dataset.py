import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class SCADataset(Dataset):
    def __init__(self, data_path, split='train', input_length=700, desync=0):
        self.data_path = data_path
        self.split = split
        self.input_length = input_length
        self.desync = desync

        # Open HDF5 file
        try:
            self.h5_file = h5py.File(data_path, 'r')
        except OSError:
            raise FileNotFoundError(f"Could not open HDF5 file at {data_path}")

        # --- 1. Robust Group Detection (AES_PT vs ASCAD) ---
        self.group = None
        
        # List of possible group paths
        if split == 'train':
            candidates = [
                'Profiling_traces',           # ASCAD
                'D1/Unprotected/Profiling',   # AES_PT
                'D1/AES_unprotected/Profiling', 
                'Profiling'
            ]
        else:
            candidates = [
                'Attack_traces',              # ASCAD
                'D1/Unprotected/Attack',      # AES_PT
                'D1/AES_unprotected/Attack',
                'Attack'
            ]

        for c in candidates:
            if c in self.h5_file:
                self.group = self.h5_file[c]
                break
        
        # Fallback: Check root
        if self.group is None:
            if 'traces' in self.h5_file or 'Traces' in self.h5_file:
                self.group = self.h5_file
            else:
                raise ValueError(f"Could not find valid group for split '{split}'. Keys: {list(self.h5_file.keys())}")

        # --- 2. Robust Dataset Loading ---
        def get_ds(names):
            for name in names:
                if name in self.group: return self.group[name]
                for k in self.group.keys():
                    if k.lower() == name.lower(): return self.group[k]
            return None

        self.traces = get_ds(['traces', 'trace'])
        self.labels = get_ds(['labels', 'label', 'Y', 'output'])
        metadata_ds = get_ds(['metadata', 'meta', 'metadata_set'])

        if self.traces is None or self.labels is None:
            raise ValueError(f"Could not find traces/labels in {self.group.name}")

        self.full_length = self.traces.shape[1]

        # --- 3. Pre-load Metadata (For Rank Calculation) ---
        self.plaintexts = None
        self.keys = None
        
        # Determine Target Byte (AES_PT=0, ASCAD=2)
        is_aes_pt = 'AES_PT' in data_path or 'D1' in self.group.name
        target_byte = 0 if is_aes_pt else 2

        if metadata_ds is not None:
            dtype_names = metadata_ds.dtype.names if metadata_ds.dtype.names else []
            pt_field = next((n for n in dtype_names if n.lower() in ['plaintext', 'pt', 'p', 'input']), None)
            key_field = next((n for n in dtype_names if n.lower() in ['key', 'k', 'key_input']), None)
            
            if pt_field and key_field:
                raw_pt = metadata_ds[pt_field]
                raw_key = metadata_ds[key_field]
                
                # Extract target byte if array, otherwise take as is
                if len(raw_pt.shape) > 1 and raw_pt.shape[1] >= 16:
                    self.plaintexts = raw_pt[:, target_byte]
                else:
                    self.plaintexts = raw_pt

                if len(raw_key.shape) > 1 and raw_key.shape[1] >= 16:
                    self.keys = raw_key[:, target_byte]
                else:
                    self.keys = raw_key

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        # 1. Load Trace
        trace_numpy = self.traces[idx]
        
        # Crop
        if self.input_length < len(trace_numpy):
            trace_numpy = trace_numpy[:self.input_length]
        
        # 2. Augmentation
        if self.split == 'train':
            if self.desync > 0:
                shift = np.random.randint(-self.desync, self.desync + 1)
                if shift > 0:
                    trace_numpy = np.pad(trace_numpy, (shift, 0), 'constant')[:-shift]
                elif shift < 0:
                    trace_numpy = np.pad(trace_numpy, (0, -shift), 'constant')[-shift:]
            
            noise = np.random.normal(0, 0.1, trace_numpy.shape)
            trace_numpy = trace_numpy + noise

        # 3. Normalize
        mean = np.mean(trace_numpy)
        std = np.std(trace_numpy)
        trace_numpy = (trace_numpy - mean) / (std + 1e-12)

        # 4. Return Tensors
        # CHANGED: Removed .unsqueeze(0). 
        # Output shape is now [Length], which DataLoader batches to [Batch, Length].
        # Your model.py then does .unsqueeze(1) to make it [Batch, 1, Length], which is correct.
        trace_tensor = torch.from_numpy(trace_numpy).float()
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return trace_tensor, label_tensor

    def close(self):
        if hasattr(self, 'h5_file') and self.h5_file:
            self.h5_file.close()

# Alias
ASCADDataset = SCADataset