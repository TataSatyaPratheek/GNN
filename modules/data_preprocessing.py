#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data preprocessing module for process mining
Handles data loading, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Normalizer, RobustScaler
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import time
import gc

def load_and_preprocess_data(data_path, required_cols=None):
    """Load and preprocess the event log data with progress feedback"""
    print("\n==== Loading and Preprocessing Data ====")
    start_time = time.time()
    
    if required_cols is None:
        required_cols = ["case_id", "task_name", "timestamp", "resource", "amount"]
    
    print(f"Loading data from {data_path}...")
    try:
        # Use more efficient CSV reading with chunking for large files
        chunksize = 100000
        chunks = []
        
        # Count total lines for progress bar
        with open(data_path, 'r') as f:
            total_lines = sum(1 for _ in f)
        
        # Read in chunks with progress bar
        with tqdm(total=total_lines, desc="Reading CSV", ncols=100) as pbar:
            for chunk in pd.read_csv(data_path, chunksize=chunksize):
                chunks.append(chunk)
                pbar.update(len(chunk))
                
        df = pd.concat(chunks, ignore_index=True)
    except Exception as e:
        print(f"\033[91mError reading CSV in chunks: {e}\033[0m")
        print("Falling back to standard pandas read_csv...")
        df = pd.read_csv(data_path)
    
    print(f"Data loaded: {len(df)} rows, {df.shape[1]} columns")
    
    # Rename columns for consistency
    df.rename(columns={
        "case:id": "case_id",
        "concept:name": "task_name",
        "time:timestamp": "timestamp",
        "org:resource": "resource",
        "case:Amount": "amount"
    }, inplace=True, errors="ignore")

    # Validate required columns
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}. Found columns: {df.columns.tolist()}")

    # Process timestamps
    print("Processing timestamps...")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    before_na = len(df)
    df.dropna(subset=["timestamp"], inplace=True)
    after_na = len(df)
    if before_na > after_na:
        print(f"\033[93mWarning: Dropped {before_na - after_na} rows with invalid timestamps\033[0m")
    
    # Sort by case ID and timestamp
    print("Sorting data...")
    df.sort_values(["case_id","timestamp"], inplace=True)
    
    # Report basic statistics
    case_count = df["case_id"].nunique()
    task_count = df["task_name"].nunique() 
    resource_count = df["resource"].nunique()
    
    # Calculate time period
    min_date = df["timestamp"].min().strftime('%Y-%m-%d')
    max_date = df["timestamp"].max().strftime('%Y-%m-%d')
    
    print(f"\033[1mData Summary\033[0m:")
    print(f"  Time period: \033[96m{min_date}\033[0m to \033[96m{max_date}\033[0m")
    print(f"  Cases: \033[96m{case_count:,}\033[0m")
    print(f"  Activities: \033[96m{task_count}\033[0m")
    print(f"  Resources: \033[96m{resource_count}\033[0m")
    print(f"  Events: \033[96m{len(df):,}\033[0m")
    print(f"Preprocessing completed in \033[96m{time.time() - start_time:.2f}s\033[0m")

    return df

def create_feature_representation(df, use_norm_features=True):
    """Create scaled or normalized feature representation with enhanced progress tracking"""
    print("\n==== Creating Feature Representation ====")
    start_time = time.time()
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Time features with progress feedback
    print("Extracting temporal features...")
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour_of_day"] = df["timestamp"].dt.hour

    # Encode tasks and resources
    print("Encoding categorical features...")
    le_task = LabelEncoder()
    le_resource = LabelEncoder()
    
    df["task_id"] = le_task.fit_transform(df["task_name"])
    df["resource_id"] = le_resource.fit_transform(df["resource"])

    # Add next task with progress feedback
    print("Computing next tasks...")
    df["next_task"] = df.groupby("case_id")["task_id"].shift(-1)
    before_na = len(df)
    df.dropna(subset=["next_task"], inplace=True)
    after_na = len(df)
    
    if before_na > after_na:
        print(f"Dropped {before_na - after_na} rows with no next task (end events)")
    
    df["next_task"] = df["next_task"].astype(int)

    # Feature scaling with progress
    print("Scaling features...")
    feature_cols = ["task_id", "resource_id", "amount", "day_of_week", "hour_of_day"]
    raw_features = df[feature_cols].values

    # Diagnose data issues
    nans = np.isnan(raw_features).sum()
    if nans > 0:
        print(f"\033[93mWarning: Found {nans} NaN values in features. Replacing with zeros.\033[0m")
        raw_features = np.nan_to_num(raw_features, nan=0.0)
    
    infs = np.isinf(raw_features).sum()
    if infs > 0:
        print(f"\033[93mWarning: Found {infs} infinite values in features. Replacing with large values.\033[0m")
        raw_features = np.nan_to_num(raw_features, posinf=1e6, neginf=-1e6)
    
    # Analyze feature characteristics
    feature_means = np.mean(raw_features, axis=0)
    feature_stds = np.std(raw_features, axis=0)
    feature_mins = np.min(raw_features, axis=0)
    feature_maxs = np.max(raw_features, axis=0)
    
    print("Feature statistics:")
    for i, col in enumerate(feature_cols):
        print(f"  {col}: mean={feature_means[i]:.4f}, std={feature_stds[i]:.4f}, min={feature_mins[i]:.4f}, max={feature_maxs[i]:.4f}")
    
    # Adaptive normalization based on data characteristics
    if use_norm_features:
        print("Using L2 normalization...")
        normalizer = Normalizer(norm='l2')
        combined_features = normalizer.fit_transform(raw_features)
    else:
        # Check if we should use robust scaling
        if np.any(feature_maxs / np.maximum(feature_mins, 1e-10) > 100):
            print("Data has extreme values - using RobustScaler...")
            scaler = RobustScaler()
        else:
            print("Using MinMaxScaler...")
            scaler = MinMaxScaler()
        combined_features = scaler.fit_transform(raw_features)

    # Add features back to dataframe
    df["feat_task_id"] = combined_features[:,0]
    df["feat_resource_id"] = combined_features[:,1]
    df["feat_amount"] = combined_features[:,2]
    df["feat_day_of_week"] = combined_features[:,3]
    df["feat_hour_of_day"] = combined_features[:,4]

    print(f"Feature representation created in \033[96m{time.time() - start_time:.2f}s\033[0m")
    return df, le_task, le_resource

def build_graph_data(df):
    """Convert preprocessed data into graph format for GNN with progress tracking"""
    print("\n==== Building Graph Data ====")
    start_time = time.time()
    
    graphs = []
    case_groups = df.groupby("case_id")
    num_cases = len(case_groups)
    
    # Create progress bar
    progress_bar = tqdm(
        case_groups, 
        desc="Building graphs",
        bar_format="{l_bar}{bar:30}{r_bar}",
        ncols=100,
        total=num_cases
    )
    
    # Track statistics
    edge_counts = []
    node_counts = []
    
    for cid, cdata in progress_bar:
        cdata = cdata.sort_values("timestamp")

        # Create node features
        x_data = torch.tensor(cdata[[
            "feat_task_id","feat_resource_id","feat_amount",
            "feat_day_of_week","feat_hour_of_day"
        ]].values, dtype=torch.float)

        # Create edges between sequential activities (more efficient)
        n_nodes = len(cdata)
        node_counts.append(n_nodes)
        
        if n_nodes > 1:
            src = list(range(n_nodes-1))
            tgt = list(range(1,n_nodes))
            edge_index = torch.tensor([src+tgt, tgt+src], dtype=torch.long)
            edge_counts.append(2 * (n_nodes - 1))  # Bidirectional edges
        else:
            edge_index = torch.empty((2,0), dtype=torch.long)
            edge_counts.append(0)
            
        y_data = torch.tensor(cdata["next_task"].values, dtype=torch.long)
        data_obj = Data(x=x_data, edge_index=edge_index, y=y_data)
        graphs.append(data_obj)
        
        # Periodically collect garbage to prevent memory buildup
        if len(graphs) % 1000 == 0:
            gc.collect()

    # Report statistics
    avg_nodes = np.mean(node_counts)
    avg_edges = np.mean(edge_counts)
    max_nodes = np.max(node_counts)
    
    print(f"\033[1mGraph Statistics\033[0m:")
    print(f"  Total graphs: \033[96m{len(graphs):,}\033[0m")
    print(f"  Avg nodes per graph: \033[96m{avg_nodes:.2f}\033[0m")
    print(f"  Avg edges per graph: \033[96m{avg_edges:.2f}\033[0m")
    print(f"  Max nodes in a graph: \033[96m{max_nodes}\033[0m")
    print(f"Graphs built in \033[96m{time.time() - start_time:.2f}s\033[0m")

    return graphs

def compute_class_weights(df, num_classes):
    """Compute balanced class weights for training with improved efficiency"""
    from sklearn.utils.class_weight import compute_class_weight
    
    print("\nComputing class weights...")
    start_time = time.time()
    
    # Extract labels more efficiently
    train_labels = df["next_task"].values
    
    # Count class frequencies
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    
    # Report class distribution
    total = len(train_labels)
    print(f"Class distribution ({len(unique_labels)} classes):")
    for label, count in zip(unique_labels[:5], counts[:5]):
        print(f"  Class {label}: {count:,} samples ({count/total*100:.2f}%)")
    if len(unique_labels) > 5:
        print(f"  ... and {len(unique_labels)-5} more classes")
    
    # Compute weights
    class_weights = np.ones(num_classes, dtype=np.float32)
    present = np.unique(train_labels)
    cw = compute_class_weight("balanced", classes=present, y=train_labels)
    
    for i, cval in enumerate(present):
        class_weights[cval] = cw[i]
    
    # Report weight range
    min_weight = np.min(class_weights[class_weights > 0])
    max_weight = np.max(class_weights)
    print(f"Class weight range: {min_weight:.4f} - {max_weight:.4f}")
    print(f"Class weights computed in \033[96m{time.time() - start_time:.2f}s\033[0m")
    
    return torch.tensor(class_weights, dtype=torch.float32)