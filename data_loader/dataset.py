import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np

class OnHWDataset(Dataset):
    """
    Dataset loader for 13-channel IMU time-series handwriting data (e.g. OnHW-words500).
    Processes variable-length sequence traces without static global padding.
    """
    def __init__(self, annotation_file, data_dir, char_to_idx=None):
        self.data_dir = data_dir
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
        self.samples = self.metadata.get("annotations", [])
        
        # Build character vocabulary if not provided
        if char_to_idx is None:
            vocab = set()
            for item in self.samples:
                vocab.update(list(item["label"]))
            vocab = sorted(list(vocab))
            # 0 is reserved for CTC blank, 1 for PAD, 2 for SOS, 3 for EOS
            self.char_to_idx = {
                "<blank>": 0,
                "<pad>": 1,
                "<sos>": 2,
                "<eos>": 3,
                **{c: i + 4 for i, c in enumerate(vocab)}
            }
        else:
            self.char_to_idx = char_to_idx
            
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta = self.samples[idx]
        file_path = os.path.join(self.data_dir, meta["filename"])
        
        # Load 13-channel IMU CSV/numpy array [Time, Channels]
        # In OnHW: channels contain accelerometers, gyroscope, force, etc.
        imu_trace = np.loadtxt(file_path, delimiter=';')
        
        # Normalize per-channel: (x - mean) / (std + 1e-6)
        mean = np.mean(imu_trace, axis=0, keepdims=True)
        std = np.std(imu_trace, axis=0, keepdims=True)
        imu_trace = (imu_trace - mean) / (std + 1e-6)
        
        # Convert to Tensor [Channels, Time]
        signal = torch.tensor(imu_trace.T, dtype=torch.float32)
        
        # Convert text label to integer tokens
        text = meta["label"]
        tokens = [self.char_to_idx[c] for c in text if c in self.char_to_idx]
        label_tensor = torch.tensor(tokens, dtype=torch.long)
        
        return {
            "signal": signal,
            "label": label_tensor,
            "raw_text": text,
            "seq_len": signal.shape[1],
            "label_len": len(tokens),
            "writer_id": meta.get("id_writer", -1)
        }

def dynamic_collate_fn(batch):
    """
    Batch-wise dynamic rectangularization:
    Pads signals only to the maximum length in the current batch (T_max)
    and constructs attention key-padding masks.
    """
    batch_size = len(batch)
    
    # 1. Determine batch maximum signal time length and label length
    max_t = max(item["seq_len"] for item in batch)
    max_label_len = max(item["label_len"] for item in batch)
    in_channels = batch[0]["signal"].shape[0]
    
    # 2. Allocate padded signal tensor: [Batch, Channels, T_max]
    padded_signals = torch.zeros(batch_size, in_channels, max_t, dtype=torch.float32)
    # Mask: True for padded positions, False for valid time positions
    signal_padding_mask = torch.ones(batch_size, max_t, dtype=torch.bool)
    
    # 3. Allocate padded labels: [Batch, L_max]
    padded_labels = torch.zeros(batch_size, max_label_len, dtype=torch.long)
    
    seq_lens = []
    label_lens = []
    raw_texts = []
    writer_ids = []
    
    for i, item in enumerate(batch):
        sig = item["signal"]
        t = item["seq_len"]
        padded_signals[i, :, :t] = sig
        signal_padding_mask[i, :t] = False  # Valid positions are unmasked
        
        lbl = item["label"]
        l = item["label_len"]
        padded_labels[i, :l] = lbl
        
        seq_lens.append(t)
        label_lens.append(l)
        raw_texts.append(item["raw_text"])
        writer_ids.append(item["writer_id"])
        
    return {
        "signals": padded_signals,                       # [B, C, T_max]
        "signal_padding_mask": signal_padding_mask,     # [B, T_max]
        "labels": padded_labels,                         # [B, L_max]
        "seq_lens": torch.tensor(seq_lens, dtype=torch.long),
        "label_lens": torch.tensor(label_lens, dtype=torch.long),
        "raw_texts": raw_texts,
        "writer_ids": torch.tensor(writer_ids, dtype=torch.long)
    }
