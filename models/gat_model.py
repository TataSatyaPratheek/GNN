#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Graph Attention Network (GAT) model for process mining
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
import time
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt

class NextTaskGAT(nn.Module):
    """
    Graph Attention Network for next task prediction
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, heads=4, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, concat=True))
        
        # Add residual connections to combat over-smoothing
        self.residuals = nn.ModuleList()
        for _ in range(num_layers-1):
            self.convs.append(GATConv(hidden_dim*heads, hidden_dim, heads=heads, concat=True))
            self.residuals.append(nn.Linear(hidden_dim*heads, hidden_dim*heads))
            
        # Use batch normalization for more stable training
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim*heads) for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(hidden_dim*heads, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            # Apply convolution
            new_x = conv(x, edge_index)
            
            # Apply batch normalization
            new_x = self.batch_norms[i](new_x)
            
            # Apply activation
            new_x = torch.nn.functional.elu(new_x)
            
            # Apply dropout
            new_x = torch.nn.functional.dropout(new_x, p=self.dropout, training=self.training)
            
            # Apply residual connection if not the first layer
            if i > 0:
                x = new_x + self.residuals[i-1](x)
            else:
                x = new_x
        
        # Global pooling
        x = global_mean_pool(x, batch)
        return self.fc(x)

def train_gat_model(model, train_loader, val_loader, criterion, optimizer, 
                   device, num_epochs=20, model_path="best_gnn_model.pth", viz_dir=None):
    """
    Train the GAT model with enhanced progress tracking
    """
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience = 5
    patience_counter = 0
    
    # Use mixed precision for faster training if available
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    use_amp = scaler is not None
    
    print("\n==== Training GAT Model ====")
    for epoch in range(1, num_epochs+1):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        
        # Create a tqdm progress bar for training
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch}/{num_epochs} [Train]",
            bar_format="{l_bar}{bar:30}{r_bar}",
            ncols=100
        )
        
        for batch_data in progress_bar:
            # Move data to device
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            
            # Use automatic mixed precision if available
            if use_amp:
                with torch.cuda.amp.autocast():
                    out = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                    graph_labels = compute_graph_label(batch_data.y, batch_data.batch).to(device, dtype=torch.long)
                    loss = criterion(out, graph_labels)
                
                # Scale gradients
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                graph_labels = compute_graph_label(batch_data.y, batch_data.batch).to(device, dtype=torch.long)
                loss = criterion(out, graph_labels)
                
                # Standard backprop
                loss.backward()
                optimizer.step()
                
            total_loss += loss.item()
            
            # Update progress bar with current loss
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        
        # Create a tqdm progress bar for validation
        val_progress = tqdm(
            val_loader, 
            desc=f"Epoch {epoch}/{num_epochs} [Valid]",
            bar_format="{l_bar}{bar:30}{r_bar}",
            ncols=100
        )
        
        with torch.no_grad():
            for batch_data in val_progress:
                batch_data = batch_data.to(device)
                out = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                glabels = compute_graph_label(batch_data.y, batch_data.batch).to(device, dtype=torch.long)
                val_loss += criterion(out, glabels).item()
                
                # Update validation progress bar
                if val_progress.n > 0:
                    val_progress.set_postfix({"val_loss": f"{val_loss/val_progress.n:.4f}"})
                else:
                    val_progress.set_postfix({"val_loss": "N/A"})
        
        avg_val_loss = val_loss/len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Calculate time taken
        epoch_time = time.time() - start_time
        
        # Print epoch summary with color
        print(f"\033[1m[Epoch {epoch}/{num_epochs}]\033[0m train_loss=\033[92m{avg_train_loss:.4f}\033[0m, val_loss=\033[93m{avg_val_loss:.4f}\033[0m, time=\033[96m{epoch_time:.2f}s\033[0m")
        
        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"  \033[1m\033[94mSaved best model\033[0m (val_loss=\033[92m{best_val_loss:.4f}\033[0m)")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter}/{patience} epochs.")
            
            if patience_counter >= patience:
                print(f"\033[93mEarly stopping triggered after {epoch} epochs\033[0m")
                break
    
    # Plot loss curves if matplotlib is available
    try:
        # Create figure
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # Save to visualization directory if provided
        if viz_dir:
            loss_curve_path = os.path.join(viz_dir, 'gat_training_loss.png')
            plt.savefig(loss_curve_path)
            print(f"Loss curve saved to {loss_curve_path}")
        else:
            plt.savefig('gat_training_loss.png')
            print(f"Loss curve saved to gat_training_loss.png")
        
        plt.close()
    except Exception as e:
        print(f"Error saving loss curve: {e}")
        
    return model

def compute_graph_label(y, batch):
    """
    Compute graph-level labels (MPS/CUDA-compatible)
    """
    # More efficient implementation that works on any device
    unique_batches = batch.unique()
    labels_out = []
    
    for bidx in unique_batches:
        mask = (batch == bidx)
        # Move data to CPU only if needed
        if y.device.type != 'cpu' and not hasattr(y, 'is_cuda'):
            yvals = y[mask]
        else:
            yvals = y[mask].detach().cpu()
            
        vals, counts = torch.unique(yvals, return_counts=True)
        lbl = vals[torch.argmax(counts)]
        labels_out.append(lbl)
        
    return torch.tensor(labels_out, device=y.device)

def evaluate_gat_model(model, val_loader, device):
    """
    Evaluate GAT model and return predictions and probabilities with progress bar
    """
    model.eval()
    y_true_all, y_pred_all, y_prob_all = [], [], []
    
    # Add progress bar for evaluation
    eval_progress = tqdm(
        val_loader, 
        desc="Evaluating GAT model",
        bar_format="{l_bar}{bar:30}{r_bar}",
        ncols=100
    )
    
    with torch.no_grad():
        for batch_data in eval_progress:
            batch_data = batch_data.to(device)
            logits = model(batch_data.x, batch_data.edge_index, batch_data.batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            glabels = compute_graph_label(batch_data.y, batch_data.batch)
            
            for i in range(logits.size(0)):
                y_pred_all.append(int(torch.argmax(logits[i]).cpu()))
                y_prob_all.append(probs[i])
                y_true_all.append(int(glabels[i]))
    
    # Calculate accuracy for feedback
    correct = sum(1 for true, pred in zip(y_true_all, y_pred_all) if true == pred)
    total = len(y_true_all)
    acc = correct / total if total > 0 else 0
    print(f"\033[1mGAT Evaluation\033[0m: Accuracy = \033[92m{acc:.4f}\033[0m ({correct}/{total})")
    
    return (
        torch.tensor(y_true_all),
        torch.tensor(y_pred_all),
        torch.tensor(y_prob_all)
    )