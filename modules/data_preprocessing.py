#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data preprocessing module for process mining
Handles data loading, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Normalizer
import torch
from torch_geometric.data import Data

def load_and_preprocess_data(data_path, required_cols=None):
    """Load and preprocess the event log data"""
    if required_cols is None:
        required_cols = ["case_id", "task_name", "timestamp", "resource", "amount"]
        
    df = pd.read_csv(data_path)
    df.rename(columns={
        "case:id": "case_id",
        "concept:name": "task_name",
        "time:timestamp": "timestamp",
        "org:resource": "resource",
        "case:Amount": "amount"
    }, inplace=True, errors="ignore")

    # Validate required columns
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing '{c}' in CSV. Found cols: {df.columns.tolist()}")

    # Process timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)
    df.sort_values(["case_id","timestamp"], inplace=True)

    return df

def create_feature_representation(df, use_norm_features=True):
    """Create scaled or normalized feature representation"""
    # Time features
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour_of_day"] = df["timestamp"].dt.hour

    # Encode tasks and resources
    le_task = LabelEncoder()
    le_resource = LabelEncoder()
    
    df["task_id"] = le_task.fit_transform(df["task_name"])
    df["resource_id"] = le_resource.fit_transform(df["resource"])

    # Next task
    df["next_task"] = df.groupby("case_id")["task_id"].shift(-1)
    df.dropna(subset=["next_task"], inplace=True)
    df["next_task"] = df["next_task"].astype(int)

    # Feature scaling
    feature_cols = ["task_id", "resource_id", "amount", "day_of_week", "hour_of_day"]
    raw_features = df[feature_cols].values

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(raw_features)

    normalizer = Normalizer(norm='l2')
    features_normed = normalizer.fit_transform(raw_features)

    # Choose feature representation
    combined_features = features_normed if use_norm_features else features_scaled

    # Add features back to dataframe
    df["feat_task_id"] = combined_features[:,0]
    df["feat_resource_id"] = combined_features[:,1]
    df["feat_amount"] = combined_features[:,2]
    df["feat_day_of_week"] = combined_features[:,3]
    df["feat_hour_of_day"] = combined_features[:,4]

    return df, le_task, le_resource

def build_graph_data(df):
    """Convert preprocessed data into graph format for GNN"""
    graphs = []
    for cid, cdata in df.groupby("case_id"):
        cdata.sort_values("timestamp", inplace=True)

        x_data = torch.tensor(cdata[[
            "feat_task_id","feat_resource_id","feat_amount",
            "feat_day_of_week","feat_hour_of_day"
        ]].values, dtype=torch.float)

        n_nodes = len(cdata)
        if n_nodes > 1:
            src = list(range(n_nodes-1))
            tgt = list(range(1,n_nodes))
            edge_index = torch.tensor([src+tgt, tgt+src], dtype=torch.long)
        else:
            edge_index = torch.empty((2,0), dtype=torch.long)
            
        y_data = torch.tensor(cdata["next_task"].values, dtype=torch.long)
        data_obj = Data(x=x_data, edge_index=edge_index, y=y_data)
        graphs.append(data_obj)

    return graphs

def compute_class_weights(df, num_classes):
    """Compute balanced class weights for training"""
    from sklearn.utils.class_weight import compute_class_weight
    train_labels = df["next_task"].values
    class_weights = np.ones(num_classes, dtype=np.float32)
    present = np.unique(train_labels)
    cw = compute_class_weight("balanced", classes=present, y=train_labels)
    for i, cval in enumerate(present):
        class_weights[cval] = cw[i]
    return torch.tensor(class_weights, dtype=torch.float32) 