import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class ASCADDataset(Dataset):
    def __init__(self, data_path, split='train', input_length=700, desync=0):
        """
        Args:
            data_path (str): Path to the ASCAD.h5 file.
            split (str): 'train' for Profiling_traces, 'test' for Attack_traces.
            input_length (int): Number of time samples to slice from the trace.
            desync (int): Maximum desynchronization shift (data augmentation).
        """
        self.data_path = data_path
        self.split = split
        self.input_length = input_length
        self.desync = desync

        # Open HDF5 file
        try:
            self.h5_file = h5py.File(data_path, 'r')
        except OSError:
            raise FileNotFoundError(f"Could not open HDF5 file at {data_path}")

        # Select the correct group based on standard ASCAD structure
        if split == 'train':
            self.group_name = 'Profiling_traces'
        elif split == 'test':
            self.group_name = 'Attack_traces'
        else:
            raise ValueError("Split must be 'train' or 'test'")

        if self.group_name not in self.h5_file:
            raise ValueError(f"Group {self.group_name} not found in {data_path}")
            
        self.group = self.h5_file[self.group_name]
        
        # Verify structure
        if 'traces' not in self.group or 'labels' not in self.group:
             raise ValueError(f"Group {self.group_name} must contain 'traces' and 'labels'")

        # Cache dataset properties
        self.n_samples = self.group['traces'].shape[0]
        self.full_length = self.group['traces'].shape[1]
        
        # Load Labels into memory (efficient as they are small integers)
        self.labels = np.array(self.group['labels'], dtype=np.int64)
        
        # Load Metadata (Plaintext and Key) for evaluation
        # ASCAD usually targets Byte 2 (Index 2). 
        # We load specific bytes here so evaluation scripts can use them directly.
        if 'metadata' in self.group:
            meta = self.group['metadata']
            # Check if metadata is structured or standard array
            # ASCAD metadata is usually a structured array with fields 'plaintext' and 'key'
            if 'plaintext' in meta.dtype.names and 'key' in meta.dtype.names:
                self.plaintexts = np.array(meta['plaintext'][:, 2], dtype=np.uint8) # Extract Byte 2
                self.keys = np.array(meta['key'][:, 2], dtype=np.uint8)             # Extract Byte 2
            else:
                print("Warning: Metadata format not recognized (missing 'plaintext' or 'key' fields).")
                self.plaintexts = None
                self.keys = None
        else:
            self.plaintexts = None
            self.keys = None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 1. Read Trace
        # We access the disk here. For high performance on small systems, 
        # one might load everything to RAM, but ASCAD traces are large.
        trace = self.group['traces'][idx]
        trace = np.array(trace, dtype=np.float32)
        
        # 2. Data Augmentation (Desynchronization)
        # Randomly shift the window if we are training and desync is enabled
        if self.desync > 0 and self.split == 'train':
            shift = np.random.randint(-self.desync, self.desync + 1)
            if shift != 0:
                # We roll the array to simulate timing jitter
                # Note: Real desync might lose data at edges, but rolling is standard 
                # if the signal of interest is centered.
                trace = np.roll(trace, shift)

        # 3. Cropping / Slicing to input_length
        # If the requested length is smaller than the full trace, we slice.
        if self.input_length < self.full_length:
            trace = trace[:self.input_length]
        elif self.input_length > self.full_length:
             # Pad with zeros if requested length is larger
             padding = self.input_length - self.full_length
             trace = np.pad(trace, (0, padding), 'constant')

        # 4. Convert to Tensor
        trace_tensor = torch.from_numpy(trace)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return trace_tensor, label_tensor

    def close(self):
        """Close the HDF5 file handle."""
        if self.h5_file:
            self.h5_file.close()

    def __del__(self):
        self.close()