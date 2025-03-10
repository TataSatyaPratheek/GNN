#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LSTM model for next activity prediction in process mining
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import time
import gc
import os
import matplotlib.pyplot as plt

class NextActivityLSTM(nn.Module):
    """
    LSTM model for next activity prediction
    """
    def __init__(self, num_cls, emb_dim=64, hidden_dim=64, num_layers=1, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(num_cls+1, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_cls)

    def forward(self, x, seq_len):
        # Sort sequences by length for more efficient packing
        seq_len_sorted, perm_idx = seq_len.sort(0, descending=True)
        x_sorted = x[perm_idx]
        
        # Apply embedding
        x_emb = self.emb(x_sorted)
        
        # Pack sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            x_emb, seq_len_sorted.cpu(), batch_first=True, enforce_sorted=True
        )
        
        # Process with LSTM
        out_packed, (h_n, c_n) = self.lstm(packed)
        
        # Get last hidden state
        last_hidden = h_n[-1]
        
        # Reorder to original order
        _, unperm_idx = perm_idx.sort(0)
        last_hidden = last_hidden[unperm_idx]
        
        # Apply batch normalization and dropout for more stable training
        last_hidden = self.batch_norm(last_hidden)
        last_hidden = self.dropout(last_hidden)
        
        # Final prediction
        logits = self.fc(last_hidden)
        return logits

def prepare_sequence_data(df, max_len=None, val_ratio=0.2):
    """
    Prepare sequence data for LSTM training with improved efficiency
    """
    print("\n==== Preparing Sequence Data ====")
    start_time = time.time()
    
    # More efficient grouping
    prefix_samples = []
    case_groups = df.groupby("case_id")
    
    # Add progress bar
    progress_bar = tqdm(
        case_groups, 
        desc="Processing cases",
        bar_format="{l_bar}{bar:30}{r_bar}",
        ncols=100
    )
    
    for cid, cdata in progress_bar:
        cdata = cdata.sort_values("timestamp")
        tasks_list = cdata["task_id"].tolist()
        
        # More efficient handling of sequences
        for i in range(1, len(tasks_list)):
            prefix = tasks_list[:i]
            label = tasks_list[i]
            prefix_samples.append((prefix, label))
    
    print(f"Generated {len(prefix_samples)} sequence samples in {time.time() - start_time:.2f}s")
    
    # Deterministic shuffling for reproducibility
    start_time = time.time()
    np.random.seed(42)
    
    # Use numpy for faster shuffling of large lists
    indices = np.arange(len(prefix_samples))
    np.random.shuffle(indices)
    prefix_samples = [prefix_samples[i] for i in indices]
    
    # Split into train/test
    split_idx = int((1 - val_ratio) * len(prefix_samples))
    train_seq = prefix_samples[:split_idx]
    test_seq = prefix_samples[split_idx:]
    
    print(f"Split data into {len(train_seq)} training and {len(test_seq)} validation samples in {time.time() - start_time:.2f}s")
    
    # Free memory
    del indices
    gc.collect()
    
    return train_seq, test_seq

def make_padded_dataset(sample_list, num_cls):
    """
    Convert sequence data to padded tensor format with improved efficiency
    """
    start_time = time.time()
    
    # Find max length once
    max_len = max(len(s[0]) for s in sample_list)
    
    # Pre-allocate arrays for speed
    n_samples = len(sample_list)
    X_padded = np.zeros((n_samples, max_len), dtype=np.int64)
    X_lens = np.zeros(n_samples, dtype=np.int64)
    Y_labels = np.zeros(n_samples, dtype=np.int64)
    
    # Process in batches with progress bar
    batch_size = 1000
    num_batches = (n_samples + batch_size - 1) // batch_size
    
    progress_bar = tqdm(
        range(num_batches),
        desc="Creating padded dataset",
        bar_format="{l_bar}{bar:30}{r_bar}",
        ncols=100
    )
    
    for b in progress_bar:
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, n_samples)
        
        for i, idx in enumerate(range(start_idx, end_idx)):
            pfx, nxt = sample_list[idx]
            seqlen = len(pfx)
            X_lens[idx] = seqlen
            
            # Shift for pad=0
            for j, tid in enumerate(pfx):
                X_padded[idx, j] = tid + 1
                
            Y_labels[idx] = nxt
    
    # Convert to tensors
    X_padded_tensor = torch.tensor(X_padded, dtype=torch.long)
    X_lens_tensor = torch.tensor(X_lens, dtype=torch.long)
    Y_labels_tensor = torch.tensor(Y_labels, dtype=torch.long)
    
    print(f"Created padded dataset in {time.time() - start_time:.2f}s")
    
    return (
        X_padded_tensor,
        X_lens_tensor,
        Y_labels_tensor,
        max_len
    )

def train_lstm_model(model, X_train_pad, X_train_len, y_train, 
                    device, batch_size=64, epochs=5, 
                    model_path="lstm_next_activity.pth", viz_dir=None):
    """
    Train the LSTM model with enhanced progress tracking
    """
    print("\n==== Training LSTM Model ====")
    
    # Switch to mixed precision training if available
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    use_amp = scaler is not None
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    dataset_size = X_train_pad.size(0)
    train_losses = []
    patience = 5
    patience_counter = 0
    best_loss = float('inf')
    
    for ep in range(1, epochs+1):
        model.train()
        start_time = time.time()
        indices = np.random.permutation(dataset_size)
        total_loss = 0.0
        
        # Create progress bar for training
        progress_bar = tqdm(
            range(0, dataset_size, batch_size),
            desc=f"Epoch {ep}/{epochs} [Train]",
            bar_format="{l_bar}{bar:30}{r_bar}",
            ncols=100
        )
        
        for start in progress_bar:
            end = min(start+batch_size, dataset_size)
            idx = indices[start:end]
            
            bx = X_train_pad[idx].to(device)
            blen = X_train_len[idx].to(device)
            by = y_train[idx].to(device)
            
            optimizer.zero_grad()
            
            # Use mixed precision if available
            if use_amp:
                with torch.cuda.amp.autocast():
                    out = model(bx, blen)
                    lval = loss_fn(out, by)
                
                scaler.scale(lval).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(bx, blen)
                lval = loss_fn(out, by)
                lval.backward()
                optimizer.step()
            
            total_loss += lval.item()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{lval.item():.4f}"})
            
        avg_loss = total_loss/((dataset_size + batch_size - 1)//batch_size)
        train_losses.append(avg_loss)
        
        # Calculate time taken
        epoch_time = time.time() - start_time
        
        # Print epoch summary with color
        print(f"\033[1m[LSTM Epoch {ep}/{epochs}]\033[0m Loss=\033[92m{avg_loss:.4f}\033[0m, Time=\033[96m{epoch_time:.2f}s\033[0m")
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)
            print(f"  \033[1m\033[94mSaved best model\033[0m (loss=\033[92m{best_loss:.4f}\033[0m)")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter}/{patience} epochs.")
            
            if patience_counter >= patience:
                print(f"\033[93mEarly stopping triggered after {ep} epochs\033[0m")
                break
    
    # Plot loss curve if matplotlib is available
    try:
        # Create figure
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('LSTM Training Loss')
        
        # Save to visualization directory if provided
        if viz_dir:
            loss_curve_path = os.path.join(viz_dir, 'lstm_training_loss.png')
            plt.savefig(loss_curve_path)
            print(f"Loss curve saved to {loss_curve_path}")
        else:
            plt.savefig('lstm_training_loss.png')
            print(f"Loss curve saved to lstm_training_loss.png")
        
        plt.close()
    except Exception as e:
        print(f"Error saving loss curve: {e}")
    
    return model

def evaluate_lstm_model(model, X_test_pad, X_test_len, y_test, batch_size, device):
    """
    Evaluate LSTM model and return predictions and probabilities with progress bar
    """
    model.eval()
    test_size = X_test_pad.size(0)
    all_preds = []
    all_probs = []
    all_targets = []
    
    # Create progress bar for evaluation
    progress_bar = tqdm(
        range(0, test_size, batch_size),
        desc="Evaluating LSTM model",
        bar_format="{l_bar}{bar:30}{r_bar}",
        ncols=100
    )
    
    with torch.no_grad():
        for start in progress_bar:
            end = min(start+batch_size, test_size)
            bx = X_test_pad[start:end].to(device)
            blen = X_test_len[start:end].to(device)
            by = y_test[start:end]
            
            out = model(bx, blen)
            logits = out.cpu().numpy()
            
            # Stable softmax computation
            logits_exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = logits_exp / np.sum(logits_exp, axis=1, keepdims=True)
            preds = np.argmax(logits, axis=1)
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_targets.extend(by.numpy())
    
    # Calculate accuracy for feedback
    correct = sum(1 for true, pred in zip(all_targets, all_preds) if true == pred)
    total = len(all_targets)
    acc = correct / total if total > 0 else 0
    print(f"\033[1mLSTM Evaluation\033[0m: Accuracy = \033[92m{acc:.4f}\033[0m ({correct}/{total})")
    
    return np.array(all_preds), np.array(all_probs), np.array(all_targets)