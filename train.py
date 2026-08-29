import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from data_loader.dataset import OnHWDataset, dynamic_collate_fn
from models.hybrid_model import Hybrid3DHandwritingRecognizer
from utils.metrics import compute_cer, compute_wer

def decode_ctc_predictions(logits, idx_to_char, blank_idx=0):
    preds = torch.argmax(logits, dim=-1).cpu().numpy()
    decoded_texts = []
    for seq in preds:
        collapsed = []
        prev = None
        for token in seq:
            if token != prev:
                if token != blank_idx and token in idx_to_char:
                    collapsed.append(idx_to_char[token])
                prev = token
        decoded_texts.append("".join(collapsed))
    return decoded_texts

def train_one_epoch(model, dataloader, optimizer, criterion_ctc, device):
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        signals = batch["signals"].to(device)  # [B, C, T]
        labels = batch["labels"].to(device)    # [B, L]
        label_lens = batch["label_lens"]       # [B]
        
        optimizer.zero_grad()
        outputs = model(signals)
        ctc_logits = outputs["ctc_logits"]      # [B, T', Vocab]
        
        # CTC expects [T', B, Vocab] with log_softmax
        log_probs = ctc_logits.permute(1, 0, 2).log_softmax(2)
        
        t_prime = ctc_logits.shape[1]
        input_lens = torch.full((signals.size(0),), t_prime, dtype=torch.long)
        
        loss = criterion_ctc(log_probs, labels, input_lens, label_lens)
        
        if not torch.isnan(loss) and not torch.isinf(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def evaluate(model, dataloader, idx_to_char, device):
    model.eval()
    all_preds = []
    all_gts = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            signals = batch["signals"].to(device)
            raw_texts = batch["raw_texts"]
            
            outputs = model(signals)
            ctc_logits = outputs["ctc_logits"]
            
            decoded_batch = decode_ctc_predictions(ctc_logits, idx_to_char)
            all_preds.extend(decoded_batch)
            all_gts.extend(raw_texts)
            
    cer = compute_cer(all_preds, all_gts)
    wer = compute_wer(all_preds, all_gts)
    return cer, wer, all_preds, all_gts

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")
    
    train_json = "./data/OnHW/train.json"
    val_json = "./data/OnHW/val.json"
    data_dir = "./data/OnHW/data"
    
    if not os.path.exists(train_json):
        print("[!] Dataset path not found. Please run create_sample_data.py first.")
        return
        
    train_dataset = OnHWDataset(train_json, data_dir)
    val_dataset = OnHWDataset(val_json, data_dir, char_to_idx=train_dataset.char_to_idx)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8, 
        shuffle=True, 
        collate_fn=dynamic_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=8, 
        shuffle=False, 
        collate_fn=dynamic_collate_fn
    )
    
    vocab_size = len(train_dataset.char_to_idx)
    model = Hybrid3DHandwritingRecognizer(
        in_channels=13, 
        d_model=256, 
        nhead=8, 
        num_layers=4, 
        vocab_size=vocab_size
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion_ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    
    epochs = 10
    best_cer = float("inf")
    
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_ctc, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        cer, wer, preds, gts = evaluate(model, val_loader, train_dataset.idx_to_char, device)
        print(f"Validation Results -> CER: {cer:.2f}% | WER: {wer:.2f}%")
        
        if cer < best_cer:
            best_cer = cer
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_hybrid_model.pth")
            print(f"[+] Saved Best Model Checkpoint (CER: {best_cer:.2f}%)")

if __name__ == "__main__":
    main()