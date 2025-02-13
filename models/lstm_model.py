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

class NextActivityLSTM(nn.Module):
    """
    LSTM model for next activity prediction
    """
    def __init__(self, num_cls, emb_dim=64, hidden_dim=64, num_layers=1):
        super().__init__()
        self.emb = nn.Embedding(num_cls+1, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_cls)

    def forward(self, x, seq_len):
        seq_len_sorted, perm_idx = seq_len.sort(0, descending=True)
        x_sorted = x[perm_idx]
        x_emb = self.emb(x_sorted)
        packed = nn.utils.rnn.pack_padded_sequence(
            x_emb, seq_len_sorted.cpu(), batch_first=True, enforce_sorted=True
        )
        out_packed, (h_n, c_n) = self.lstm(packed)
        last_hidden = h_n[-1]
        _, unperm_idx = perm_idx.sort(0)
        last_hidden = last_hidden[unperm_idx]
        logits = self.fc(last_hidden)
        return logits

def prepare_sequence_data(df, max_len=None):
    """
    Prepare sequence data for LSTM training
    """
    prefix_samples = []
    for cid, cdata in df.groupby("case_id"):
        cdata = cdata.sort_values("timestamp")
        tasks_list = cdata["task_id"].tolist()
        for i in range(1, len(tasks_list)):
            prefix = tasks_list[:i]
            label = tasks_list[i]
            prefix_samples.append((prefix, label))
    
    random.shuffle(prefix_samples)
    split_idx = int(0.8*len(prefix_samples))
    train_seq = prefix_samples[:split_idx]
    test_seq = prefix_samples[split_idx:]
    
    return train_seq, test_seq

def make_padded_dataset(sample_list, num_cls):
    """
    Convert sequence data to padded tensor format
    """
    max_len = max(len(s[0]) for s in sample_list)
    X_padded, X_lens, Y_labels = [], [], []
    
    for (pfx, nxt) in sample_list:
        seqlen = len(pfx)
        X_lens.append(seqlen)
        seq = [(tid+1) for tid in pfx]  # shift for pad=0
        pad_len = max_len - seqlen
        seq += [0]*pad_len
        X_padded.append(seq)
        Y_labels.append(nxt)
    
    return (
        torch.tensor(X_padded, dtype=torch.long),
        torch.tensor(X_lens, dtype=torch.long),
        torch.tensor(Y_labels, dtype=torch.long),
        max_len
    )

def train_lstm_model(model, X_train_pad, X_train_len, y_train, 
                    device, batch_size=64, epochs=5, 
                    model_path="lstm_next_activity.pth"):
    """
    Train the LSTM model
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dataset_size = X_train_pad.size(0)
    
    for ep in range(1, epochs+1):
        model.train()
        indices = np.random.permutation(dataset_size)
        total_loss = 0.0
        
        for start in range(0, dataset_size, batch_size):
            end = min(start+batch_size, dataset_size)
            idx = indices[start:end]
            
            bx = X_train_pad[idx].to(device)
            blen = X_train_len[idx].to(device)
            by = y_train[idx].to(device)
            
            optimizer.zero_grad()
            out = model(bx, blen)
            lval = loss_fn(out, by)
            lval.backward()
            optimizer.step()
            total_loss += lval.item()
            
        avg_loss = total_loss/((dataset_size + batch_size - 1)//batch_size)
        print(f"[LSTM Ep {ep}/{epochs}] Loss={avg_loss:.4f}")
    
    torch.save(model.state_dict(), model_path)
    return model

def evaluate_lstm_model(model, X_test_pad, X_test_len, batch_size, device):
    """
    Evaluate LSTM model and return predictions and probabilities
    """
    model.eval()
    test_size = X_test_pad.size(0)
    logits_list = []
    
    with torch.no_grad():
        for start in range(0, test_size, batch_size):
            end = min(start+batch_size, test_size)
            bx = X_test_pad[start:end].to(device)
            blen = X_test_len[start:end].to(device)
            out = model(bx, blen)
            logits_list.append(out.cpu().numpy())
    
    logits_arr = np.concatenate(logits_list, axis=0)
    
    # Stable softmax
    logits_exp = np.exp(logits_arr - np.max(logits_arr, axis=1, keepdims=True))
    probs = logits_exp / np.sum(logits_exp, axis=1, keepdims=True)
    preds = np.argmax(logits_arr, axis=1)
    
    return preds, probs 