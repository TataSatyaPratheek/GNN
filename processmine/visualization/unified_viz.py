#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified visualization API for process mining analysis
"""

import os
import time
import logging
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from processmine.utils.dependency import (
    check_dependency, requires_dependency, with_fallback, 
    HAVE_PLOTLY, HAVE_NETWORKX
)

logger = logging.getLogger(__name__)

# Set better default styles
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 150
sns.set_palette("viridis")

class ProcessVisualizer:
    """
    Unified process mining visualization 
    """
    
    def __init__(self, output_dir: Optional[str] = None, style: str = "default"):
        """
        Initialize process visualizer
        
        Args:
            output_dir: Directory to save visualizations
            style: Visualization style ('default', 'dark', 'light', 'colorblind')
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        self.set_style(style)
        
        # Check for optional dependencies
        self.have_plotly = check_dependency("plotly")
        self.have_networkx = check_dependency("networkx")
        self.have_umap = check_dependency("umap")
        
        if not self.have_plotly:
            logger.warning("Plotly not found. Some interactive visualizations will not be available.")
        
        if not self.have_networkx:
            logger.warning("NetworkX not found. Some graph visualizations will not be available.")
        
        # Track rendered visualizations
        self.visualizations = []
    
    def set_style(self, style: str) -> None:
        """
        Set visualization style
        
        Args:
            style: Style name ('default', 'dark', 'light', 'colorblind')
        """
        self.style = style
        
        if style == "dark":
            plt.style.use('dark_background')
            self.color_palette = "plasma"
            self.edge_color = "white"
            self.node_color = "lightblue"
            self.background_color = "#333333"
        elif style == "light":
            plt.style.use('seaborn-v0_8-bright')
            self.color_palette = "viridis"
            self.edge_color = "gray"
            self.node_color = "skyblue"
            self.background_color = "#ffffff"
        elif style == "colorblind":
            plt.style.use('seaborn-v0_8-colorblind')
            self.color_palette = "colorblind"
            self.edge_color = "gray"
            self.node_color = "skyblue"
            self.background_color = "#ffffff"
        else:  # default
            plt.style.use('seaborn-v0_8-whitegrid')
            self.color_palette = "viridis"
            self.edge_color = "gray"
            self.node_color = "skyblue"
            self.background_color = "#ffffff"
        
        # Update seaborn palette
        sns.set_palette(self.color_palette)
    
    def save_figure(self, fig: plt.Figure, filename: str) -> str:
        """
        Save figure to output directory
        
        Args:
            fig: Matplotlib figure
            filename: Filename for the visualization
            
        Returns:
            Path to saved figure
        """
        if self.output_dir:
            # Ensure directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Add file extension if not present
            if not filename.endswith((".png", ".jpg", ".svg", ".pdf")):
                filename += ".png"
            
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, bbox_inches='tight')
            plt.close(fig)
            
            # Track visualization
            self.visualizations.append({
                'type': 'figure',
                'name': filename,
                'path': filepath,
                'timestamp': time.time()
            })
            
            return filepath
        else:
            # No output directory, just display
            plt.show()
            return ""
    
    def cycle_time_distribution(self, durations: np.ndarray, filename: str = "cycle_time_distribution.png") -> plt.Figure:
        """
        Plot cycle time distribution with statistics
        
        Args:
            durations: Array of case durations in hours
            filename: Filename for the visualization
            
        Returns:
            Matplotlib figure
        """
        logger.info("Creating cycle time distribution visualization")
        
        # Calculate statistics
        mean_duration = np.mean(durations)
        median_duration = np.median(durations)
        p90 = np.percentile(durations, 90)
        p95 = np.percentile(durations, 95)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot histogram with KDE
        sns.histplot(durations, bins=min(50, len(durations) // 20), 
                    kde=True, color="royalblue", ax=ax)
        
        # Add percentile lines
        ax.axvline(mean_duration, color="red", linestyle="-", 
                  linewidth=2, label=f"Mean: {mean_duration:.1f}h")
        ax.axvline(median_duration, color="green", linestyle="--", 
                  linewidth=2, label=f"Median: {median_duration:.1f}h")
        ax.axvline(p90, color="purple", linestyle="-.", 
                  linewidth=2, label=f"90th Percentile: {p90:.1f}h")
        ax.axvline(p95, color="orange", linestyle="-.", 
                  linewidth=2, label=f"95th Percentile: {p95:.1f}h")
        
        # Enhanced styling
        ax.set_title("Process Cycle Time Distribution", fontsize=16)
        ax.set_xlabel("Duration (hours)", fontsize=14)
        ax.set_ylabel("Number of Cases", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=12, loc='upper right')
        
        # Add summary statistics as text
        stats_text = f"Total Cases: {len(durations)}\n"
        stats_text += f"Mean: {mean_duration:.2f}h\n"
        stats_text += f"Median: {median_duration:.2f}h\n"
        stats_text += f"Min: {np.min(durations):.2f}h\n"
        stats_text += f"Max: {np.max(durations):.2f}h\n"
        stats_text += f"Std Dev: {np.std(durations):.2f}h"
        
        ax.annotate(stats_text, xy=(0.02, 0.97), xycoords='axes fraction',
                   fontsize=11, ha='left', va='top',
                   bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure if output directory provided
        self.save_figure(fig, filename)
        
        return fig
    
    def bottleneck_analysis(self, bottleneck_stats: pd.DataFrame, 
                           significant_bottlenecks: pd.DataFrame,
                           task_encoder: Any,
                           filename: str = "bottleneck_analysis.png") -> plt.Figure:
        """
        Visualize bottleneck analysis results
        
        Args:
            bottleneck_stats: DataFrame with bottleneck statistics
            significant_bottlenecks: DataFrame with significant bottlenecks
            task_encoder: Task label encoder
            filename: Filename for the visualization
            
        Returns:
            Matplotlib figure
        """
        logger.info("Creating bottleneck analysis visualization")
        
        # Top N bottlenecks to display
        top_n = min(10, len(significant_bottlenecks))
        
        if top_n == 0:
            logger.warning("No significant bottlenecks found")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No significant bottlenecks found", 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            self.save_figure(fig, filename)
            return fig
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
        
        # Top bottlenecks bar chart
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Get top bottlenecks
        top_bottlenecks = significant_bottlenecks.head(top_n)
        
        # Create labels
        labels = []
        for _, row in top_bottlenecks.iterrows():
            src_id, dst_id = int(row["task_id"]), int(row["next_task_id"])
            try:
                src_name = task_encoder.inverse_transform([src_id])[0]
                dst_name = task_encoder.inverse_transform([dst_id])[0]
                
                # Truncate long names
                if len(src_name) > 15:
                    src_name = src_name[:12] + "..."
                if len(dst_name) > 15:
                    dst_name = dst_name[:12] + "..."
                    
                labels.append(f"{src_name} → {dst_name}")
            except:
                labels.append(f"Task {src_id} → Task {dst_id}")
        
        # Plot bar chart
        bars = ax1.barh(labels, top_bottlenecks["mean_hours"], color="royalblue")
        
        # Add count annotations
        for i, bar in enumerate(bars):
            count = int(top_bottlenecks.iloc[i]["count"])
            ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
                    f"n={count}", va='center')
        
        ax1.set_title("Top Process Bottlenecks", fontsize=16)
        ax1.set_xlabel("Average Wait Time (hours)", fontsize=12)
        ax1.set_ylabel("Transition", fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.6, axis='x')
        
        # Wait time histogram
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Get all wait times from top bottlenecks
        wait_times = []
        top_transitions = set()
        for _, row in top_bottlenecks.iterrows():
            src_id, dst_id = int(row["task_id"]), int(row["next_task_id"])
            top_transitions.add((src_id, dst_id))
        
        # Make a visualization of wait time distribution
        ax2.set_title("Wait Time Distribution for Bottlenecks", fontsize=16)
        ax2.set_xlabel("Wait Time (hours)", fontsize=12)
        ax2.set_ylabel("Frequency", fontsize=12)
        
        # Placeholder for real data
        x = np.linspace(0, 20, 1000)
        for i, (src_id, dst_id) in enumerate(list(top_transitions)[:5]):  # Only show top 5
            # Use a log-normal distribution as placeholder
            mean = significant_bottlenecks[
                (significant_bottlenecks["task_id"] == src_id) & 
                (significant_bottlenecks["next_task_id"] == dst_id)
            ]["mean_hours"].values[0]
            
            std = significant_bottlenecks[
                (significant_bottlenecks["task_id"] == src_id) & 
                (significant_bottlenecks["next_task_id"] == dst_id)
            ]["std"].values[0] / 3600.0  # Convert to hours
            
            # Handle missing or invalid std
            if np.isnan(std) or std == 0:
                std = mean * 0.5  # Use reasonable default
            
            # Parameters for lognormal
            sigma = np.sqrt(np.log(1 + (std/mean)**2))
            mu = np.log(mean) - sigma**2/2
            
            # Generate distribution
            y = np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi))
            
            # Scale distribution by frequency
            count = significant_bottlenecks[
                (significant_bottlenecks["task_id"] == src_id) & 
                (significant_bottlenecks["next_task_id"] == dst_id)
            ]["count"].values[0]
            
            y = y * count / np.max(y)
            
            # Plot
            try:
                src_name = task_encoder.inverse_transform([src_id])[0]
                dst_name = task_encoder.inverse_transform([dst_id])[0]
                
                # Truncate long names
                if len(src_name) > 10:
                    src_name = src_name[:7] + "..."
                if len(dst_name) > 10:
                    dst_name = dst_name[:7] + "..."
                    
                label = f"{src_name} → {dst_name}"
            except:
                label = f"Task {src_id} → Task {dst_id}"
                
            ax2.plot(x, y, label=label)
        
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        # Bottleneck impact table
        ax3 = fig.add_subplot(gs[1, 0:])
        ax3.axis('tight')
        ax3.axis('off')
        
        # Prepare table data
        table_data = []
        headers = ["Rank", "Source", "Target", "Avg Wait (h)", "Count", "Impact Score"]
        
        for i, (_, row) in enumerate(top_bottlenecks.iterrows()):
            src_id, dst_id = int(row["task_id"]), int(row["next_task_id"])
            try:
                src_name = task_encoder.inverse_transform([src_id])[0]
                dst_name = task_encoder.inverse_transform([dst_id])[0]
            except:
                src_name = f"Task {src_id}"
                dst_name = f"Task {dst_id}"
            
            table_data.append([
                i+1,
                src_name,
                dst_name,
                f"{row['mean_hours']:.2f}",
                int(row['count']),
                f"{row['bottleneck_score']:.2f}" if 'bottleneck_score' in row else "N/A"
            ])
        
        table = ax3.table(
            cellText=table_data,
            colLabels=headers,
            loc='center',
            cellLoc='center',
            colWidths=[0.05, 0.3, 0.3, 0.1, 0.1, 0.1]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Add title to table
        ax3.set_title("Bottleneck Details", fontsize=14, pad=20)
        
        plt.tight_layout()
        
        # Save figure if output directory provided
        self.save_figure(fig, filename)
        
        return fig
    
    @requires_dependency("networkx", "NetworkX is required for process flow visualization")
    def process_flow(self, bottleneck_stats: pd.DataFrame, 
                     task_encoder: Any,
                     significant_bottlenecks: Optional[pd.DataFrame] = None,
                     filename: str = "process_flow.png") -> plt.Figure:
        """
        Visualize process flow with bottlenecks
        
        Args:
            bottleneck_stats: DataFrame with bottleneck statistics
            task_encoder: Task label encoder
            significant_bottlenecks: DataFrame with significant bottlenecks
            filename: Filename for the visualization
            
        Returns:
            Matplotlib figure
        """
        logger.info("Creating process flow visualization")
        
        import networkx as nx
        
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
        if significant_bottlenecks is not None:
            for _, row in significant_bottlenecks.iterrows():
                btop_edges.add((int(row["task_id"]), int(row["next_task_id"])))
        
        # Calculate edge colors and widths based on whether they are bottlenecks
        edge_cols, edge_wids, edge_alphas = [], [], []
        for (u, v) in G_flow.edges():
            if (u, v) in btop_edges:
                edge_cols.append("red")
                edge_wids.append(3.0)
                edge_alphas.append(1.0)
            else:
                edge_cols.append(self.edge_color)
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
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Draw nodes with varying sizes
        nx.draw_networkx_nodes(G_flow, pos, 
                              node_size=[node_sizes[n] for n in G_flow.nodes()],
                              node_color=self.node_color, 
                              edgecolors="black",
                              alpha=0.8,
                              ax=ax)
        
        # Draw edges with proper styling
        for i, (u, v) in enumerate(G_flow.edges()):
            nx.draw_networkx_edges(G_flow, pos, 
                                  edgelist=[(u, v)],
                                  width=edge_wids[i],
                                  alpha=edge_alphas[i],
                                  edge_color=edge_cols[i],
                                  arrows=True,
                                  arrowsize=20,
                                  connectionstyle="arc3,rad=0.1",
                                  ax=ax)
        
        # Draw labels with improved visibility
        labels_dict = {}
        for n in G_flow.nodes():
            try:
                # Handle potential encoding issues
                label = task_encoder.inverse_transform([int(n)])[0]
                # Truncate too long labels
                if len(label) > 20:
                    label = label[:17] + "..."
                labels_dict[n] = label
            except:
                labels_dict[n] = f"Task {n}"
        
        nx.draw_networkx_labels(G_flow, pos, labels_dict, 
                               font_size=10, font_weight='bold',
                               bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2", alpha=0.7),
                               ax=ax)
        
        # Add edge labels for bottlenecks only
        edge_lbl = {}
        for (u, v) in btop_edges:
            if (u, v) in G_flow.edges():
                edge_lbl[(u, v)] = f"{G_flow[u][v]['mean_hours']:.1f}h\n({G_flow[u][v]['freq']} cases)"
        
        nx.draw_networkx_edge_labels(G_flow, pos, edge_labels=edge_lbl, 
                                    font_color="darkred", font_size=9, font_weight='bold',
                                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="red", boxstyle="round"),
                                    ax=ax)
        
        # Add title
        ax.set_title("Process Flow with Critical Bottlenecks Highlighted", fontsize=16)
        
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', lw=3, label='Bottleneck Transition'),
            Line2D([0], [0], color=self.edge_color, lw=1, label='Normal Transition')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Add summary statistics
        ax.text(0.02, 0.02, 
                f"Total transitions: {len(G_flow.edges())}\n"
                f"Bottlenecks highlighted: {len(btop_edges)}\n"
                f"Activities: {len(G_flow.nodes())}",
                fontsize=10, 
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="round,pad=0.5"),
                transform=ax.transAxes)
        
        ax.axis('off')  # Hide axis
        plt.tight_layout()
        
        # Save figure if output directory provided
        self.save_figure(fig, filename)
        
        return fig
    
    def transition_heatmap(self, transitions: pd.DataFrame, 
                          task_encoder: Any,
                          filename: str = "transition_heatmap.png") -> plt.Figure:
        """
        Create transition probability heatmap
        
        Args:
            transitions: DataFrame with transitions data
            task_encoder: Task label encoder
            filename: Filename for the visualization
            
        Returns:
            Matplotlib figure
        """
        logger.info("Creating transition probability heatmap")
        
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
            logger.info(f"Matrix too large ({n_tasks}x{n_tasks}), focusing on top 20 tasks only")
        
        # Get task names
        try:
            xticklabels = [task_encoder.inverse_transform([int(c)])[0] for c in prob_matrix.columns]
            yticklabels = [task_encoder.inverse_transform([int(r)])[0] for r in prob_matrix.index]
            
            # Truncate long labels
            xticklabels = [l[:20] + '...' if len(l) > 20 else l for l in xticklabels]
            yticklabels = [l[:20] + '...' if len(l) > 20 else l for l in yticklabels]
        except:
            # Fallback if transformation fails
            xticklabels = [f"Task {c}" for c in prob_matrix.columns]
            yticklabels = [f"Task {r}" for r in prob_matrix.index]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Use a visually more appealing colormap
        heatmap = sns.heatmap(prob_matrix, cmap=self.color_palette, annot=True, fmt=".2f",
                             xticklabels=xticklabels,
                             yticklabels=yticklabels,
                             linewidths=0.5, linecolor='whitesmoke',
                             ax=ax)
        
        # Improve label readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        # Add title and labels
        ax.set_title("Transition Probability Heatmap", fontsize=16)
        ax.set_xlabel("Next Activity", fontsize=14)
        ax.set_ylabel("Current Activity", fontsize=14)
        
        # Add timestamp
        ax.text(0.02, 0.02, f"Generated: {time.strftime('%Y-%m-%d %H:%M')}", fontsize=8,
               transform=ax.transAxes)
        
        plt.tight_layout()
        
        # Save figure if output directory provided
        self.save_figure(fig, filename)
        
        return fig
    
    @requires_dependency("plotly", "Plotly is required for creating Sankey diagrams")
    def sankey_diagram(self, transitions: pd.DataFrame, 
                       task_encoder: Any,
                       filename: str = "sankey_diagram.html") -> Any:
        """
        Create Sankey diagram of process flow
        
        Args:
            transitions: DataFrame with transitions data
            task_encoder: Task label encoder
            filename: Filename for the visualization
            
        Returns:
            Plotly figure
        """
        logger.info("Creating Sankey diagram")
        
        import plotly.graph_objects as go
        
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
            task_names = {task_id: task_encoder.inverse_transform([int(task_id)])[0] for task_id in unique_tasks}
            
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
        
        # Create Sankey diagram
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
            paper_bgcolor=self.background_color,
            plot_bgcolor=self.background_color
        )
        
        # Save if output directory provided
        if self.output_dir:
            # Ensure directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            filepath = os.path.join(self.output_dir, filename)
            sankey_fig.write_html(
                filepath, 
                include_plotlyjs="cdn",
                full_html=True,
                config={"responsive": True}
            )
            
            # Track visualization
            self.visualizations.append({
                'type': 'plotly',
                'name': filename,
                'path': filepath,
                'timestamp': time.time()
            })
            
            logger.info(f"Saved Sankey diagram to {filepath}")
        
        return sankey_fig
    
    def resource_workload(self, resource_stats: pd.DataFrame, 
                         resource_encoder: Optional[Any] = None,
                         filename: str = "resource_workload.png") -> plt.Figure:
        """
        Visualize resource workload
        
        Args:
            resource_stats: DataFrame with resource workload statistics
            resource_encoder: Optional resource label encoder
            filename: Filename for the visualization
            
        Returns:
            Matplotlib figure
        """
        logger.info("Creating resource workload visualization")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        
        # Workload distribution
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Get top resources by activity count
        top_n = min(15, len(resource_stats))
        top_resources = resource_stats.head(top_n)
        
        # Create resource labels
        if resource_encoder is not None:
            try:
                resource_labels = [
                    resource_encoder.inverse_transform([int(r)])[0] 
                    for r in top_resources.index
                ]
                # Truncate long names
                resource_labels = [
                    l[:15] + "..." if len(l) > 15 else l 
                    for l in resource_labels
                ]
            except:
                resource_labels = [f"Resource {r}" for r in top_resources.index]
        else:
            resource_labels = [f"Resource {r}" for r in top_resources.index]
        
        # Plot workload
        bars = ax1.barh(resource_labels, top_resources["workload_percentage"], color="royalblue")
        
        # Add count annotations
        for i, bar in enumerate(bars):
            count = int(top_resources.iloc[i]["activity_count"])
            ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                    f"n={count}", va='center')
        
        ax1.set_title("Resource Workload Distribution", fontsize=16)
        ax1.set_xlabel("Workload (%)", fontsize=12)
        ax1.set_ylabel("Resource", fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.6, axis='x')
        
        # Specialization plot
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Scatter plot of activity count vs specialization
        scatter = ax2.scatter(
            top_resources["unique_activities"], 
            top_resources["specialization"],
            s=top_resources["activity_count"] / top_resources["activity_count"].max() * 200,
            alpha=0.7,
            c=top_resources["workload_percentage"],
            cmap="viridis"
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Workload (%)')
        
        # Add labels for top resources
        for i, txt in enumerate(resource_labels):
            ax2.annotate(
                txt, 
                (top_resources.iloc[i]["unique_activities"], 
                 top_resources.iloc[i]["specialization"]),
                fontsize=8,
                alpha=0.8
            )
        
        ax2.set_title("Resource Specialization vs Activity Variety", fontsize=16)
        ax2.set_xlabel("Number of Unique Activities", fontsize=12)
        ax2.set_ylabel("Specialization Score", fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        # Lorenz curve (workload inequality)
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Calculate Lorenz curve
        sorted_counts = np.sort(resource_stats["activity_count"].values)
        cum_counts = np.cumsum(sorted_counts)
        cum_pcts = cum_counts / cum_counts[-1]
        
        # Population percentiles
        population_pcts = np.arange(1, len(cum_pcts) + 1) / len(cum_pcts)
        
        # Plot Lorenz curve
        ax3.plot(population_pcts, cum_pcts, 'b-', linewidth=2, label='Actual Distribution')
        
        # Add equality line
        ax3.plot([0, 1], [0, 1], 'k--', label='Perfect Equality')
        
        # Add reference points
        ax3.plot([0.2, 0.2], [0, cum_pcts[int(0.2 * len(cum_pcts))]], 'r:', alpha=0.5)
        ax3.plot([0.5, 0.5], [0, cum_pcts[int(0.5 * len(cum_pcts))]], 'r:', alpha=0.5)
        ax3.plot([0.8, 0.8], [0, cum_pcts[int(0.8 * len(cum_pcts))]], 'r:', alpha=0.5)
        
        # Calculate Gini coefficient
        gini = 1 - 2 * np.trapz(cum_pcts, population_pcts)
        
        ax3.set_title(f"Workload Inequality (Gini: {gini:.3f})", fontsize=16)
        ax3.set_xlabel("Cumulative % of Resources", fontsize=12)
        ax3.set_ylabel("Cumulative % of Activities", fontsize=12)
        ax3.grid(True, linestyle='--', alpha=0.6)
        ax3.legend()
        
        # Resource case count vs activity count
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Scatter plot
        scatter = ax4.scatter(
            resource_stats["case_count"],
            resource_stats["activity_count"],
            s=resource_stats["unique_activities"] * 10,
            alpha=0.7,
            c=resource_stats["specialization"],
            cmap="coolwarm"
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Specialization')
        
        # Add labels for top resources by activity count
        for i, txt in enumerate(resource_labels[:5]):  # Just top 5
            ax4.annotate(
                txt, 
                (top_resources.iloc[i]["case_count"], 
                 top_resources.iloc[i]["activity_count"]),
                fontsize=8,
                alpha=0.8
            )
        
        ax4.set_title("Resource Case Count vs Activity Count", fontsize=16)
        ax4.set_xlabel("Number of Cases", fontsize=12)
        ax4.set_ylabel("Number of Activities", fontsize=12)
        ax4.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        
        # Save figure if output directory provided
        self.save_figure(fig, filename)
        
        return fig
    
    def resource_allocation_heatmap(self, df: pd.DataFrame,
                                   task_encoder: Any,
                                   resource_encoder: Any,
                                   filename: str = "resource_allocation.png") -> plt.Figure:
        """
        Create resource allocation heatmap
        
        Args:
            df: Process data dataframe
            task_encoder: Task label encoder
            resource_encoder: Resource label encoder
            filename: Filename for the visualization
            
        Returns:
            Matplotlib figure
        """
        logger.info("Creating resource allocation heatmap")
        
        # Create pivot table of task-resource allocations
        allocation = pd.crosstab(df["task_id"], df["resource_id"])
        
        # For large matrices, focus on the most important tasks and resources
        if allocation.shape[0] > 20 or allocation.shape[1] > 20:
            # Limit to top 20 tasks and resources
            task_totals = allocation.sum(axis=1)
            resource_totals = allocation.sum(axis=0)
            
            top_tasks = task_totals.nlargest(20).index
            top_resources = resource_totals.nlargest(20).index
            
            allocation = allocation.loc[top_tasks, top_resources]
            
            logger.info(f"Matrix too large, focusing on top 20 tasks and resources only")
        
        # Get task and resource names
        try:
            task_labels = [task_encoder.inverse_transform([int(t)])[0] for t in allocation.index]
            resource_labels = [resource_encoder.inverse_transform([int(r)])[0] for r in allocation.columns]
            
            # Truncate long names
            task_labels = [t[:20] + '...' if len(t) > 20 else t for t in task_labels]
            resource_labels = [r[:20] + '...' if len(r) > 20 else r for r in resource_labels]
        except:
            task_labels = [f"Task {t}" for t in allocation.index]
            resource_labels = [f"Resource {r}" for r in allocation.columns]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create normalized version (by task)
        allocation_norm = allocation.div(allocation.sum(axis=1), axis=0) * 100
        
        # Use a visually more appealing colormap
        heatmap = sns.heatmap(allocation_norm, cmap=self.color_palette, annot=True, fmt=".1f",
                             xticklabels=resource_labels,
                             yticklabels=task_labels,
                             linewidths=0.5, linecolor='whitesmoke',
                             ax=ax)
        
        # Improve label readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        # Add title and labels
        ax.set_title("Resource Allocation by Task (%)", fontsize=16)
        ax.set_xlabel("Resource", fontsize=14)
        ax.set_ylabel("Task", fontsize=14)
        
        plt.tight_layout()
        
        # Save figure if output directory provided
        self.save_figure(fig, filename)
        
        return fig
    
    def create_dashboard(self, 
                        cycle_times: Optional[np.ndarray] = None,
                        bottleneck_stats: Optional[pd.DataFrame] = None,
                        significant_bottlenecks: Optional[pd.DataFrame] = None,
                        transition_matrix: Optional[pd.DataFrame] = None,
                        resource_stats: Optional[pd.DataFrame] = None,
                        task_encoder: Optional[Any] = None,
                        resource_encoder: Optional[Any] = None,
                        filename: str = "dashboard.html") -> None:
        """
        Create a comprehensive dashboard with multiple visualizations
        
        Args:
            cycle_times: Optional array of case durations in hours
            bottleneck_stats: Optional DataFrame with bottleneck statistics
            significant_bottlenecks: Optional DataFrame with significant bottlenecks
            transition_matrix: Optional DataFrame with transition matrix
            resource_stats: Optional DataFrame with resource workload statistics
            task_encoder: Optional task label encoder
            resource_encoder: Optional resource label encoder
            filename: Filename for the dashboard
        """
        if not HAVE_PLOTLY:
            logger.error("Plotly is required for creating dashboards. Please install it with: pip install plotly")
            return None
        
        logger.info("Creating process mining dashboard")
        
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Track which visualizations we can create
        can_show_cycle_times = cycle_times is not None and len(cycle_times) > 0
        can_show_bottlenecks = (bottleneck_stats is not None and 
                               significant_bottlenecks is not None and 
                               task_encoder is not None)
        can_show_transitions = transition_matrix is not None and task_encoder is not None
        can_show_resources = resource_stats is not None
        
        # Count number of available visualizations
        num_visualizations = sum([
            can_show_cycle_times,
            can_show_bottlenecks,
            can_show_transitions,
            can_show_resources
        ])
        
        if num_visualizations == 0:
            logger.error("No data provided for dashboard visualization")
            return None
        
        # Create appropriate subplot grid
        if num_visualizations <= 2:
            rows, cols = 1, num_visualizations
        else:
            rows, cols = 2, 2
        
        # Create subplot titles
        subplot_titles = []
        if can_show_cycle_times:
            subplot_titles.append("Cycle Time Distribution")
        if can_show_bottlenecks:
            subplot_titles.append("Top Process Bottlenecks")
        if can_show_transitions:
            subplot_titles.append("Activity Transitions")
        if can_show_resources:
            subplot_titles.append("Resource Workload")
        
        # Create subplots
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
        
        # Add visualizations
        current_idx = 0
        
        # Cycle times
        if can_show_cycle_times:
            row, col = current_idx // cols + 1, current_idx % cols + 1
            
            # Calculate metrics
            mean_duration = np.mean(cycle_times)
            median_duration = np.median(cycle_times)
            p90 = np.percentile(cycle_times, 90)
            
            # Create histogram
            fig.add_trace(
                go.Histogram(
                    x=cycle_times,
                    name="Cycle Time",
                    marker_color='royalblue',
                    opacity=0.7,
                    nbinsx=30
                ),
                row=row, col=col
            )
            
            # Add lines for statistics
            fig.add_vline(x=mean_duration, line_dash="solid", line_color="red",
                         annotation_text=f"Mean: {mean_duration:.1f}h", row=row, col=col)
            fig.add_vline(x=median_duration, line_dash="dash", line_color="green",
                         annotation_text=f"Median: {median_duration:.1f}h", row=row, col=col)
            fig.add_vline(x=p90, line_dash="dot", line_color="purple",
                         annotation_text=f"P90: {p90:.1f}h", row=row, col=col)
            
            # Update layout
            fig.update_xaxes(title_text="Duration (hours)", row=row, col=col)
            fig.update_yaxes(title_text="Number of Cases", row=row, col=col)
            
            current_idx += 1
        
        # Bottlenecks
        if can_show_bottlenecks:
            row, col = current_idx // cols + 1, current_idx % cols + 1
            
            # Get top bottlenecks
            top_n = min(10, len(significant_bottlenecks))
            top_bottlenecks = significant_bottlenecks.head(top_n)
            
            # Create labels
            labels = []
            for _, row_data in top_bottlenecks.iterrows():
                src_id, dst_id = int(row_data["task_id"]), int(row_data["next_task_id"])
                try:
                    src_name = task_encoder.inverse_transform([src_id])[0]
                    dst_name = task_encoder.inverse_transform([dst_id])[0]
                    
                    # Truncate long names
                    if len(src_name) > 15:
                        src_name = src_name[:12] + "..."
                    if len(dst_name) > 15:
                        dst_name = dst_name[:12] + "..."
                        
                    labels.append(f"{src_name} → {dst_name}")
                except:
                    labels.append(f"Task {src_id} → Task {dst_id}")
            
            # Create bar chart
            fig.add_trace(
                go.Bar(
                    y=labels,
                    x=top_bottlenecks["mean_hours"],
                    orientation='h',
                    marker_color='royalblue',
                    text=top_bottlenecks["count"].apply(lambda x: f"n={int(x)}"),
                    textposition='outside'
                ),
                row=row, col=col
            )
            
            # Update layout
            fig.update_xaxes(title_text="Average Wait Time (hours)", row=row, col=col)
            fig.update_yaxes(title_text="Transition", row=row, col=col)
            
            current_idx += 1
        
        # Transitions
        if can_show_transitions:
            row, col = current_idx // cols + 1, current_idx % cols + 1
            
            # Calculate transition metrics
            if transition_matrix is None and bottleneck_stats is not None:
                # Create transition matrix from bottleneck stats
                task_ids = pd.concat([
                    bottleneck_stats["task_id"], 
                    bottleneck_stats["next_task_id"]
                ]).unique()
                
                transitions = bottleneck_stats.copy()
                
                # Calculate the percentage of transitions from each source task
                transitions["pct"] = transitions.groupby("task_id")["count"].transform(
                    lambda x: x / x.sum() * 100
                )
                
                # Convert to heatmap format
                max_cols = min(15, len(task_ids))
                
                # Get top tasks by total transitions
                task_totals = transitions.groupby("task_id")["count"].sum()
                top_tasks = task_totals.nlargest(max_cols).index
                
                # Filter transitions
                filtered_trans = transitions[
                    transitions["task_id"].isin(top_tasks) & 
                    transitions["next_task_id"].isin(top_tasks)
                ]
                
                # Create matrix
                transition_matrix = filtered_trans.pivot_table(
                    values="pct", 
                    index="task_id", 
                    columns="next_task_id", 
                    fill_value=0
                )
            
            if transition_matrix is not None:
                # Get task names
                try:
                    task_labels = {
                        t: task_encoder.inverse_transform([int(t)])[0][:15] 
                        for t in transition_matrix.index
                    }
                except:
                    task_labels = {t: f"Task {t}" for t in transition_matrix.index}
                
                # Create heatmap
                z_data = transition_matrix.values
                
                fig.add_trace(
                    go.Heatmap(
                        z=z_data,
                        x=[task_labels.get(t, f"Task {t}") for t in transition_matrix.columns],
                        y=[task_labels.get(t, f"Task {t}") for t in transition_matrix.index],
                        colorscale="Viridis",
                        colorbar=dict(title="Percentage (%)"),
                        hovertemplate="From: %{y}<br>To: %{x}<br>Pct: %{z:.1f}%<extra></extra>"
                    ),
                    row=row, col=col
                )
                
                # Update layout
                fig.update_xaxes(title_text="Next Activity", row=row, col=col)
                fig.update_yaxes(title_text="Current Activity", row=row, col=col)
            
            current_idx += 1
        
        # Resource workload
        if can_show_resources:
            row, col = current_idx // cols + 1, current_idx % cols + 1
            
            # Get top resources
            top_n = min(15, len(resource_stats))
            top_resources = resource_stats.head(top_n)
            
            # Create resource labels
            if resource_encoder is not None:
                try:
                    resource_labels = [
                        resource_encoder.inverse_transform([int(r)])[0][:15]
                        for r in top_resources.index
                    ]
                except:
                    resource_labels = [f"Resource {r}" for r in top_resources.index]
            else:
                resource_labels = [f"Resource {r}" for r in top_resources.index]
            
            # Create bar chart
            fig.add_trace(
                go.Bar(
                    y=resource_labels,
                    x=top_resources["workload_percentage"],
                    orientation='h',
                    marker_color='royalblue',
                    text=top_resources["activity_count"].apply(lambda x: f"n={int(x)}"),
                    textposition='outside'
                ),
                row=row, col=col
            )
            
            # Update layout
            fig.update_xaxes(title_text="Workload (%)", row=row, col=col)
            fig.update_yaxes(title_text="Resource", row=row, col=col)
            
            current_idx += 1
        
        # Update overall layout
        fig.update_layout(
            title_text="Process Mining Analysis Dashboard",
            height=300 * rows + 100,
            width=600 * cols,
            showlegend=False
        )
        
        # Save if output directory provided
        if self.output_dir:
            # Ensure directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            filepath = os.path.join(self.output_dir, filename)
            fig.write_html(
                filepath, 
                include_plotlyjs="cdn",
                full_html=True,
                config={"responsive": True}
            )
            
            # Track visualization
            self.visualizations.append({
                'type': 'dashboard',
                'name': filename,
                'path': filepath,
                'timestamp': time.time()
            })
            
            logger.info(f"Saved dashboard to {filepath}")
        
        return fig