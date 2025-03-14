#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization module for process mining analysis with enhanced visuals
and better error handling for dependencies
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
import numpy as np
from sklearn.manifold import TSNE
import time
import os
import warnings

# Make UMAP optional to avoid dependency issues
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("\033[93mWarning: UMAP not available. Dimensionality reduction will use t-SNE only.\033[0m")

# Set better style defaults for matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 150
sns.set_palette("viridis")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png"):
    """Plot enhanced confusion matrix with improved visuals"""
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
    
    print("\n==== Creating Confusion Matrix ====")
    start_time = time.time()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize for better interpretation
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Handle potential NaN values
    cm_norm = np.nan_to_num(cm_norm)
    
    # Calculate plot size based on class count
    n_classes = len(class_names)
    fig_size = max(8, min(20, n_classes / 2))
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(fig_size * 2, fig_size))
    
    # Plot absolute confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names if n_classes <= 20 else [],
                yticklabels=class_names if n_classes <= 20 else [],
                ax=axes[0], cbar=False)
    axes[0].set_title("Confusion Matrix (Absolute Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    
    # Plot normalized confusion matrix
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names if n_classes <= 20 else [],
                yticklabels=class_names if n_classes <= 20 else [],
                ax=axes[1], cbar=True)
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    
    # If many classes, rotate labels
    if n_classes > 10:
        for ax in axes:
            plt.setp(ax.get_xticklabels(), rotation=90)
            plt.setp(ax.get_yticklabels(), rotation=0)
    
    # Add overall metrics
    plt.figtext(0.5, 0.01, f"Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f}", 
                ha="center", fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    plt.suptitle("Model Performance Analysis", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # Save with high quality
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {save_path} in {time.time() - start_time:.2f}s")
    return accuracy, f1

# Update the t-SNE code in visualization/process_viz.py:

def plot_embeddings(embeddings, labels=None, method="tsne", save_path=None):
    """Plot task embeddings using t-SNE or UMAP with enhanced visuals"""
    print(f"\n==== Creating Embeddings Visualization (using {method}) ====")
    start_time = time.time()
    
    # Handle UMAP not being available
    if method.lower() == "umap" and not UMAP_AVAILABLE:
        print("\033[93mWarning: UMAP not available. Falling back to t-SNE.\033[0m")
        method = "tsne"
    
    # Calculate embeddings
    if method.lower() == "tsne":
        # Ensure perplexity is less than number of samples
        safe_perplexity = min(30, max(5, embeddings.shape[0] - 1))
        print(f"Running t-SNE with perplexity {safe_perplexity}...")
        coords = TSNE(n_components=2, perplexity=safe_perplexity, 
                     random_state=42, n_jobs=-1).fit_transform(embeddings)
        title = "Task Embeddings - t-SNE"
    else:  # umap
        print("Running UMAP...")
        try:
            # Ensure n_neighbors is less than number of samples
            n_neighbors = min(30, max(2, embeddings.shape[0] // 2))
            coords = umap.UMAP(n_components=2, random_state=42,
                              n_neighbors=n_neighbors,
                              min_dist=0.1).fit_transform(embeddings)
            title = "Task Embeddings - UMAP"
        except Exception as e:
            print(f"\033[91mError running UMAP: {e}\033[0m")
            print("Falling back to t-SNE...")
            # Safe perplexity for t-SNE
            safe_perplexity = min(30, max(5, embeddings.shape[0] - 1))
            coords = TSNE(n_components=2, perplexity=safe_perplexity, 
                         random_state=42, n_jobs=-1).fit_transform(embeddings)
            title = "Task Embeddings - t-SNE (UMAP failed)"
                
    # Create plot with better visuals
    plt.figure(figsize=(10, 8))
    
    # If labels are provided, use them for coloring
    if labels is not None:
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='viridis', 
                          alpha=0.8, s=100, edgecolors='w', linewidths=0.5)
        plt.colorbar(scatter, label="Task ID" if isinstance(labels[0], (int, float)) else "Cluster")
    else:
        plt.scatter(coords[:, 0], coords[:, 1], alpha=0.8, s=100, 
                    edgecolors='w', linewidths=0.5)
    
    # Add annotations if not too many points
    if embeddings.shape[0] <= 30 and labels is not None:
        for i, (x, y) in enumerate(coords):
            plt.annotate(str(labels[i]), (x, y), fontsize=9, 
                        ha='center', va='center', color='black', 
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.title(title, fontsize=14)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add timestamp and method info
    plt.figtext(0.02, 0.02, f"Generated: {time.strftime('%Y-%m-%d %H:%M')}", fontsize=8)
    plt.figtext(0.98, 0.02, f"Method: {method.upper()}", fontsize=8, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Embeddings visualization saved to {save_path} in {time.time() - start_time:.2f}s")
    
    plt.close()
    return coords

def plot_cycle_time_distribution(durations, save_path="cycle_time_distribution.png"):
    """Plot enhanced cycle time distribution with better visuals and percentiles"""
    print("\n==== Creating Cycle Time Distribution ====")
    start_time = time.time()
    
    # Calculate statistics
    mean_duration = np.mean(durations)
    median_duration = np.median(durations)
    p90 = np.percentile(durations, 90)
    p95 = np.percentile(durations, 95)
    
    # Create figure with improved aesthetics
    plt.figure(figsize=(12, 7))
    
    # Plot histogram with KDE
    sns.histplot(durations, bins=min(50, len(durations) // 20), 
                kde=True, color="royalblue", edgecolor="white", alpha=0.7)
    
    # Add percentile lines
    plt.axvline(mean_duration, color="red", linestyle="-", 
               linewidth=2, label=f"Mean: {mean_duration:.1f}h")
    plt.axvline(median_duration, color="green", linestyle="--", 
               linewidth=2, label=f"Median: {median_duration:.1f}h")
    plt.axvline(p90, color="purple", linestyle="-.", 
               linewidth=2, label=f"90th Percentile: {p90:.1f}h")
    plt.axvline(p95, color="orange", linestyle="-.", 
               linewidth=2, label=f"95th Percentile: {p95:.1f}h")
    
    # Enhanced styling
    plt.title("Process Cycle Time Distribution", fontsize=16)
    plt.xlabel("Duration (hours)", fontsize=14)
    plt.ylabel("Number of Cases", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='upper right')
    
    # Add summary statistics as text
    stats_text = f"Total Cases: {len(durations)}\n"
    stats_text += f"Mean: {mean_duration:.2f}h\n"
    stats_text += f"Median: {median_duration:.2f}h\n"
    stats_text += f"Min: {np.min(durations):.2f}h\n"
    stats_text += f"Max: {np.max(durations):.2f}h\n"
    stats_text += f"Std Dev: {np.std(durations):.2f}h"
    
    plt.annotate(stats_text, xy=(0.02, 0.97), xycoords='axes fraction',
                fontsize=11, ha='left', va='top',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"Cycle time distribution saved to {save_path} in {time.time() - start_time:.2f}s")
    return {"mean": mean_duration, "median": median_duration, "p90": p90, "p95": p95}

def plot_process_flow(bottleneck_stats, le_task, top_bottlenecks, 
                     save_path="process_flow_bottlenecks.png"):
    """Plot enhanced process flow with bottlenecks highlighted using improved layout"""
    print("\n==== Creating Process Flow Visualization ====")
    start_time = time.time()
    
    # Create graph
    G_flow = nx.DiGraph()
    
    # Add edges with attributes
    for i, row in bottleneck_stats.iterrows():
        src = int(row["task_id"])
        dst = int(row["next_task_id"])
        G_flow.add_edge(src, dst, 
                        freq=int(row["count"]), 
                        mean_hours=row["mean_hours"],
                        weight=float(row["count"]))  # Use count for edge weight in layout
    
    # Identify bottleneck edges
    btop_edges = set()
    for _, row in top_bottlenecks.iterrows():
        btop_edges.add((int(row["task_id"]), int(row["next_task_id"])))
    
    # Calculate edge colors and widths based on whether they are bottlenecks
    edge_cols, edge_wids, edge_alphas = [], [], []
    for (u, v) in G_flow.edges():
        if (u, v) in btop_edges:
            edge_cols.append("red")
            edge_wids.append(3.0)
            edge_alphas.append(1.0)
        else:
            edge_cols.append("gray")
            edge_wids.append(1.0)
            edge_alphas.append(0.6)
    
    # Calculate node sizes based on their importance in the graph
    node_sizes = {}
    for node in G_flow.nodes():
        # Size based on sum of in and out degrees
        node_sizes[node] = 300 + 100 * (G_flow.in_degree(node) + G_flow.out_degree(node))
    
    # Choose better layout based on graph size
    n_nodes = len(G_flow.nodes())
    if n_nodes <= 20:
        # For smaller graphs, use a more structured layout
        pos = nx.kamada_kawai_layout(G_flow)
    else:
        # For larger graphs, use a force-directed layout with adjustments
        pos = nx.spring_layout(G_flow, k=0.3, iterations=50, seed=42)
    
    # Create a larger figure for better visibility
    plt.figure(figsize=(14, 12))
    
    # Draw nodes with varying sizes
    nx.draw_networkx_nodes(G_flow, pos, 
                          node_size=[node_sizes[n] for n in G_flow.nodes()],
                          node_color="lightblue", 
                          edgecolors="black",
                          alpha=0.8)
    
    # Draw edges with proper styling
    for i, (u, v) in enumerate(G_flow.edges()):
        nx.draw_networkx_edges(G_flow, pos, 
                               edgelist=[(u, v)],
                               width=edge_wids[i],
                               alpha=edge_alphas[i],
                               edge_color=edge_cols[i],
                               arrows=True,
                               arrowsize=20,
                               connectionstyle="arc3,rad=0.1")
    
    # Draw labels with improved visibility
    labels_dict = {}
    for n in G_flow.nodes():
        try:
            # Handle potential encoding issues
            label = le_task.inverse_transform([int(n)])[0]
            # Truncate too long labels
            if len(label) > 20:
                label = label[:17] + "..."
            labels_dict[n] = label
        except:
            labels_dict[n] = f"Task {n}"
    
    nx.draw_networkx_labels(G_flow, pos, labels_dict, 
                           font_size=10, font_weight='bold',
                           bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2", alpha=0.7))
    
    # Add edge labels for bottlenecks only
    edge_lbl = {}
    for (u, v) in btop_edges:
        if (u, v) in G_flow.edges():
            edge_lbl[(u, v)] = f"{G_flow[u][v]['mean_hours']:.1f}h\n({G_flow[u][v]['freq']} cases)"
    
    nx.draw_networkx_edge_labels(G_flow, pos, edge_labels=edge_lbl, 
                                font_color="darkred", font_size=9, font_weight='bold',
                                bbox=dict(facecolor="white", alpha=0.7, edgecolor="red", boxstyle="round"))
    
    # Add title and legend
    plt.title("Process Flow with Critical Bottlenecks Highlighted", fontsize=16)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=3, label='Bottleneck Transition'),
        Line2D([0], [0], color='gray', lw=1, label='Normal Transition')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add summary statistics
    plt.figtext(0.02, 0.02, 
                f"Total transitions: {len(G_flow.edges())}\n"
                f"Bottlenecks highlighted: {len(btop_edges)}\n"
                f"Activities: {len(G_flow.nodes())}",
                fontsize=10, 
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="round,pad=0.5"))
    
    plt.axis('off')  # Hide axis
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Process flow visualization saved to {save_path} in {time.time() - start_time:.2f}s")
    
    # Also create a simpler version for large graphs if needed
    if n_nodes > 30:
        create_simplified_flow(G_flow, le_task, btop_edges, 
                              save_path.replace('.png', '_simplified.png'))
    
    return G_flow

def create_simplified_flow(G, le_task, bottlenecks, save_path):
    """Create a simplified version of process flow for large graphs"""
    # Find main components of the graph - focus only on important nodes
    important_nodes = set()
    
    # Include nodes in bottlenecks
    for u, v in bottlenecks:
        important_nodes.add(u)
        important_nodes.add(v)
    
    # Add nodes with high centrality
    centrality = nx.betweenness_centrality(G)
    top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    for node, _ in top_central:
        important_nodes.add(node)
    
    # Add high-degree nodes
    degrees = dict(G.degree())
    top_degree = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    for node, _ in top_degree:
        important_nodes.add(node)
    
    # Create subgraph
    important_nodes = list(important_nodes)
    H = G.subgraph(important_nodes)
    
    # Draw simplified graph
    plt.figure(figsize=(12, 10))
    pos = nx.kamada_kawai_layout(H)
    
    # Draw edges with colors based on bottlenecks
    for u, v in H.edges():
        if (u, v) in bottlenecks:
            nx.draw_networkx_edges(H, pos, edgelist=[(u, v)], 
                                  width=3.0, edge_color="red", arrows=True)
        else:
            nx.draw_networkx_edges(H, pos, edgelist=[(u, v)], 
                                  width=1.0, edge_color="gray", arrows=True, alpha=0.6)
    
    # Draw nodes
    nx.draw_networkx_nodes(H, pos, node_color="lightblue", 
                          node_size=800, edgecolors="black", alpha=0.8)
    
    # Draw labels
    labels_dict = {}
    for n in H.nodes():
        try:
            label = le_task.inverse_transform([int(n)])[0]
            if len(label) > 20:
                label = label[:17] + "..."
            labels_dict[n] = label
        except:
            labels_dict[n] = f"Task {n}"
    
    nx.draw_networkx_labels(H, pos, labels_dict, font_size=10, font_weight='bold',
                          bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.2"))
    
    plt.title("Simplified Process Flow (Key Activities Only)", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"Simplified process flow saved to {save_path}")

def plot_transition_heatmap(transitions, le_task, save_path="transition_probability_heatmap.png"):
    """Plot enhanced transition probability heatmap with improved readability"""
    print("\n==== Creating Transition Heatmap ====")
    start_time = time.time()
    
    # Group and calculate transition matrix
    trans_count = transitions.groupby(["task_id", "next_task_id"]).size().unstack(fill_value=0)
    
    # Calculate probability matrix
    row_sums = trans_count.sum(axis=1)
    prob_matrix = trans_count.div(row_sums, axis=0)
    
    # Handle case of empty rows (divide by zero)
    prob_matrix = prob_matrix.fillna(0)
    
    # Determine how to handle large matrices
    n_tasks = prob_matrix.shape[0]
    
    if n_tasks > 20:
        # For large matrices, focus on the most important tasks
        # Calculate total transitions for each task (in + out)
        importance = row_sums + trans_count.sum(axis=0)
        top_tasks = importance.nlargest(20).index
        prob_matrix = prob_matrix.loc[top_tasks, top_tasks]
        print(f"Matrix too large ({n_tasks}x{n_tasks}), focusing on top 20 tasks only")
    
    # Get task names
    try:
        xticklabels = [le_task.inverse_transform([int(c)])[0] for c in prob_matrix.columns]
        yticklabels = [le_task.inverse_transform([int(r)])[0] for r in prob_matrix.index]
        
        # Truncate long labels
        xticklabels = [l[:20] + '...' if len(l) > 20 else l for l in xticklabels]
        yticklabels = [l[:20] + '...' if len(l) > 20 else l for l in yticklabels]
    except:
        # Fallback if transformation fails
        xticklabels = [f"Task {c}" for c in prob_matrix.columns]
        yticklabels = [f"Task {r}" for r in prob_matrix.index]
    
    # Create enhanced heatmap
    plt.figure(figsize=(14, 12))
    
    # Use a visually more appealing colormap
    ax = sns.heatmap(prob_matrix, cmap="viridis", annot=True, fmt=".2f",
                    xticklabels=xticklabels,
                    yticklabels=yticklabels,
                    linewidths=0.5, linecolor='whitesmoke')
    
    # Improve label readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    # Add title and labels
    plt.title("Transition Probability Heatmap", fontsize=16)
    plt.xlabel("Next Activity", fontsize=14)
    plt.ylabel("Current Activity", fontsize=14)
    
    # Add timestamp
    plt.figtext(0.02, 0.02, f"Generated: {time.strftime('%Y-%m-%d %H:%M')}", fontsize=8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Transition heatmap saved to {save_path} in {time.time() - start_time:.2f}s")
    
    return prob_matrix

def create_sankey_diagram(transitions, le_task, save_path="process_flow_sankey.html"):
    """Create enhanced Sankey diagram of process flow with better styling"""
    print("\n==== Creating Sankey Diagram ====")
    start_time = time.time()
    
    # Extract case-level information from transitions
    df = transitions.copy()
    
    # Get start and end activities for each case
    start_counts = df.groupby("case_id").first()["task_id"].value_counts().to_dict()
    end_counts = df.groupby("case_id").last()["task_id"].value_counts().to_dict()
    
    # Calculate transition counts
    trans_count = df.groupby(["task_id", "next_task_id"]).size().reset_index(name='count')
    
    # Create node list with better naming
    unique_tasks = sorted(set(df["task_id"].unique()).union(set(df["next_task_id"].dropna().unique())))
    
    try:
        # Map task IDs to readable names
        task_names = {task_id: le_task.inverse_transform([int(task_id)])[0] for task_id in unique_tasks}
        
        # Truncate long names for better display
        task_names = {k: (v[:30] + '...' if len(v) > 30 else v) for k, v in task_names.items()}
    except:
        # Fallback if transformation fails
        task_names = {task_id: f"Task {task_id}" for task_id in unique_tasks}
    
    # Create nodes list with "Start" and "End" pseudo-nodes
    nodes = ["Start"] + [task_names[task_id] for task_id in unique_tasks] + ["End"]
    
    # Create node mapping for indices
    node_idx = {n: i for i, n in enumerate(nodes)}
    task_to_idx = {task_id: node_idx[task_names[task_id]] for task_id in unique_tasks}
    
    # Prepare Sankey data
    sources, targets, values, labels = [], [], [], []
    
    # Add start transitions
    for act_id, count in start_counts.items():
        sources.append(node_idx["Start"])
        targets.append(task_to_idx[act_id])
        values.append(int(count))
        labels.append("Process Start")
    
    # Add end transitions
    for act_id, count in end_counts.items():
        sources.append(task_to_idx[act_id])
        targets.append(node_idx["End"])
        values.append(int(count))
        labels.append("Process End")
    
    # Add internal transitions
    for _, row in trans_count.iterrows():
        src_id, tgt_id, count = row["task_id"], row["next_task_id"], row["count"]
        if pd.notna(tgt_id):  # Skip NaN targets
            sources.append(task_to_idx[src_id])
            targets.append(task_to_idx[tgt_id])
            values.append(int(count))
            labels.append(f"{int(count)} transitions")
    
    # Calculate node colors based on their role in the process
    node_colors = []
    for i, node in enumerate(nodes):
        if node == "Start":
            node_colors.append("rgba(0, 128, 0, 0.8)")  # Green for start
        elif node == "End":
            node_colors.append("rgba(128, 0, 0, 0.8)")  # Red for end
        else:
            # Calculate if this is mainly a source, target, or mixed node
            out_sum = sum(values[j] for j, s in enumerate(sources) if s == i)
            in_sum = sum(values[j] for j, t in enumerate(targets) if t == i)
            
            if out_sum > in_sum * 2:
                node_colors.append("rgba(31, 119, 180, 0.8)")  # Blue for source nodes
            elif in_sum > out_sum * 2:
                node_colors.append("rgba(255, 127, 14, 0.8)")  # Orange for target nodes
            else:
                node_colors.append("rgba(44, 160, 44, 0.8)")  # Green for mixed nodes
    
    # Create enhanced Sankey diagram
    sankey_fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=labels,
            hoverinfo="all",
            hovertemplate="From: %{source.label}<br>To: %{target.label}<br>Value: %{value}<br>%{label}<extra></extra>"
        )
    )])
    
    # Improve layout with better styling
    sankey_fig.update_layout(
        title_text="Process Flow Sankey Diagram",
        font=dict(size=12, family="Arial"),
        autosize=True,
        width=1200,
        height=800,
        margin=dict(l=25, r=25, b=25, t=50),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # Save as HTML with higher quality
    sankey_fig.write_html(
        save_path, 
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True}
    )
    
    print(f"Sankey diagram saved to {save_path} in {time.time() - start_time:.2f}s")
    
    return sankey_fig