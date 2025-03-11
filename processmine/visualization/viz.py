import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Optional, Union, Tuple, Any

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAVE_PLOTLY = True
except ImportError:
    HAVE_PLOTLY = False
    logger.warning("Plotly not installed. Interactive visualizations unavailable.")

try:
    import networkx as nx
    HAVE_NETWORKX = True
except ImportError:
    HAVE_NETWORKX = False
    logger.warning("NetworkX not installed. Process flow visualizations will be limited.")


class ProcessVisualizer:
    """Unified visualization class for process mining"""
    
    def __init__(self, output_dir=None, style='default', force_static=False):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save visualizations
            style: Visualization style ('default', 'dark', 'light')
            force_static: Whether to force static (matplotlib) visualizations
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        self.style = style
        self.force_static = force_static
        
        # Set style based on preference
        self._set_style(style)
        
        # Use interactive visualizations if available
        self.use_interactive = HAVE_PLOTLY and not force_static
    
    def _set_style(self, style):
        """Set visualization style"""
        if style == 'dark':
            plt.style.use('dark_background')
            self.colors = {
                'primary': '#3498db',
                'secondary': '#e74c3c',
                'background': '#121212',
                'accent': '#f39c12'
            }
        elif style == 'light':
            plt.style.use('seaborn-v0_8-pastel')
            self.colors = {
                'primary': '#3498db',
                'secondary': '#e74c3c',
                'background': '#ffffff',
                'accent': '#f39c12'
            }
        else:  # default
            plt.style.use('seaborn-v0_8-whitegrid')
            self.colors = {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'background': '#f8f9fa',
                'accent': '#2ca02c'
            }
    
    def _save_fig(self, fig, filename):
        """Save figure to output directory"""
        if self.output_dir:
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Add extension if not present
            if not any(filename.endswith(ext) for ext in ['.png', '.jpg', '.svg', '.pdf']):
                filename += '.png'
            
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, bbox_inches='tight', dpi=300)
            plt.close(fig)
            
            logger.info(f"Saved visualization to {filepath}")
            
            return filepath
        else:
            plt.show()
            return None
    
    def cycle_time_distribution(self, durations, filename="cycle_time_distribution.png"):
        """
        Create cycle time distribution visualization
        
        Args:
            durations: Array of case durations in hours
            filename: Output filename
            
        Returns:
            Path to saved figure or None
        """
        if self.use_interactive and not filename.endswith('.png'):
            return self._cycle_time_interactive(durations, filename)
        
        # Calculate statistics
        mean_duration = np.mean(durations)
        median_duration = np.median(durations)
        p90 = np.percentile(durations, 90)
        p95 = np.percentile(durations, 95)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram with KDE
        sns.histplot(durations, kde=True, color=self.colors['primary'], ax=ax)
        
        # Add percentile lines
        ax.axvline(mean_duration, color=self.colors['secondary'], linestyle='-', 
                  label=f"Mean: {mean_duration:.1f}h")
        ax.axvline(median_duration, color=self.colors['accent'], linestyle='--', 
                  label=f"Median: {median_duration:.1f}h")
        ax.axvline(p95, color='purple', linestyle='-.', 
                  label=f"95th Percentile: {p95:.1f}h")
        
        # Add styling
        ax.set_title("Process Cycle Time Distribution", fontsize=14)
        ax.set_xlabel("Duration (hours)")
        ax.set_ylabel("Number of Cases")
        ax.legend()
        
        # Add text box with statistics
        stats_text = (f"Total cases: {len(durations)}\n"
                     f"Mean: {mean_duration:.2f}h\n"
                     f"Median: {median_duration:.2f}h\n"
                     f"P90: {p90:.2f}h\n"
                     f"P95: {p95:.2f}h")
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
        
        # Save figure
        return self._save_fig(fig, filename)
    
    def _cycle_time_interactive(self, durations, filename="cycle_time_distribution.html"):
        """Create interactive cycle time visualization with Plotly"""
        if not HAVE_PLOTLY:
            logger.warning("Plotly not available. Falling back to static visualization.")
            return self.cycle_time_distribution(durations, filename.replace('.html', '.png'))
        
        # Calculate statistics
        mean_duration = np.mean(durations)
        median_duration = np.median(durations)
        p90 = np.percentile(durations, 90)
        p95 = np.percentile(durations, 95)
        
        # Create figure
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=durations,
            nbinsx=30,
            marker_color=self.colors['primary'],
            opacity=0.7,
            name="Duration"
        ))
        
        # Add lines for statistics
        fig.add_vline(x=mean_duration, line_dash="solid", line_color=self.colors['secondary'],
                     annotation_text=f"Mean: {mean_duration:.1f}h")
        fig.add_vline(x=median_duration, line_dash="dash", line_color=self.colors['accent'],
                     annotation_text=f"Median: {median_duration:.1f}h")
        fig.add_vline(x=p95, line_dash="dot", line_color="purple",
                     annotation_text=f"P95: {p95:.1f}h")
        
        # Update layout
        fig.update_layout(
            title="Process Cycle Time Distribution",
            xaxis_title="Duration (hours)",
            yaxis_title="Number of Cases",
            template="plotly_white" if self.style != 'dark' else "plotly_dark"
        )
        
        # Save figure
        if self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            fig.write_html(filepath)
            logger.info(f"Saved interactive visualization to {filepath}")
            return filepath
        
        return None
    
    def bottleneck_analysis(self, bottleneck_stats, significant_bottlenecks, task_encoder,
                          filename="bottleneck_analysis.png"):
        """
        Create bottleneck analysis visualization
        
        Args:
            bottleneck_stats: DataFrame with bottleneck statistics
            significant_bottlenecks: DataFrame with significant bottlenecks
            task_encoder: Task label encoder
            filename: Output filename
            
        Returns:
            Path to saved figure or None
        """
        # Get top bottlenecks
        top_n = min(10, len(significant_bottlenecks))
        if top_n == 0:
            logger.warning("No significant bottlenecks found")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No significant bottlenecks found", 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return self._save_fig(fig, filename)
        
        # Get top bottlenecks
        top_bottlenecks = significant_bottlenecks.head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
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
        
        # Plot horizontal bar chart
        bars = ax.barh(labels, top_bottlenecks["mean_hours"].values, color=self.colors['primary'])
        
        # Add count annotations
        for i, bar in enumerate(bars):
            count = int(top_bottlenecks.iloc[i]["count"])
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
                   f"n={count}", va='center')
        
        # Add styling
        ax.set_title("Top Process Bottlenecks", fontsize=14)
        ax.set_xlabel("Average Wait Time (hours)")
        ax.set_ylabel("Transition")
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Save figure
        return self._save_fig(fig, filename)
    
    def process_flow(self, bottleneck_stats, task_encoder, significant_bottlenecks=None,
                    filename="process_flow.png"):
        """
        Create process flow visualization
        
        Args:
            bottleneck_stats: DataFrame with bottleneck statistics
            task_encoder: Task label encoder
            significant_bottlenecks: DataFrame with significant bottlenecks
            filename: Output filename
            
        Returns:
            Path to saved figure or None
        """
        if not HAVE_NETWORKX:
            logger.error("NetworkX is required for process flow visualization")
            return None
        
        # Create graph
        G = nx.DiGraph()
        
        # Add edges with attributes
        for _, row in bottleneck_stats.iterrows():
            src = int(row["task_id"])
            dst = int(row["next_task_id"])
            G.add_edge(src, dst, 
                      freq=int(row["count"]), 
                      mean_hours=row["mean_hours"],
                      weight=float(row["count"]))
        
        # Identify bottleneck edges
        bottleneck_edges = set()
        if significant_bottlenecks is not None:
            for _, row in significant_bottlenecks.iterrows():
                bottleneck_edges.add((int(row["task_id"]), int(row["next_task_id"])))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Choose layout based on graph size
        if len(G.nodes()) <= 20:
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
        
        # Draw nodes
        node_sizes = [300 + 50 * (G.in_degree(n) + G.out_degree(n)) for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=self.colors['primary'],
                             alpha=0.8, ax=ax)
        
        # Draw edges
        edge_colors = []
        edge_widths = []
        
        for u, v in G.edges():
            if (u, v) in bottleneck_edges:
                edge_colors.append(self.colors['secondary'])
                edge_widths.append(2.5)
            else:
                edge_colors.append(self.colors['primary'])
                edge_widths.append(1.0)
        
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, 
                            arrows=True, arrowsize=15, ax=ax)
        
        # Draw labels
        try:
            labels = {n: task_encoder.inverse_transform([int(n)])[0] for n in G.nodes()}
            # Truncate long labels
            labels = {k: (v[:12] + "..." if len(v) > 12 else v) for k, v in labels.items()}
        except:
            labels = {n: f"Task {n}" for n in G.nodes()}
        
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, ax=ax)
        
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=self.colors['secondary'], lw=2.5, label='Bottleneck'),
            Line2D([0], [0], color=self.colors['primary'], lw=1, label='Normal Flow')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Add title
        ax.set_title("Process Flow with Bottlenecks", fontsize=14)
        ax.axis('off')
        
        # Save figure
        return self._save_fig(fig, filename)
    
    def transition_heatmap(self, transitions, task_encoder, filename="transition_heatmap.png"):
        """
        Create transition probability heatmap
        
        Args:
            transitions: DataFrame with transition data
            task_encoder: Task label encoder
            filename: Output filename
            
        Returns:
            Path to saved figure or None
        """
        # Group and calculate transition matrix
        trans_count = transitions.groupby(["task_id", "next_task_id"]).size().unstack(fill_value=0)
        
        # Calculate probability matrix
        prob_matrix = trans_count.div(trans_count.sum(axis=1), axis=0)
        
        # Handle empty rows
        prob_matrix = prob_matrix.fillna(0)
        
        # Focus on most important tasks if matrix is large
        if prob_matrix.shape[0] > 15:
            importance = trans_count.sum(axis=1) + trans_count.sum(axis=0)
            top_tasks = importance.nlargest(15).index
            prob_matrix = prob_matrix.loc[top_tasks, top_tasks]
        
        # Get task names
        try:
            xlabels = [task_encoder.inverse_transform([int(c)])[0] for c in prob_matrix.columns]
            ylabels = [task_encoder.inverse_transform([int(r)])[0] for r in prob_matrix.index]
            
            # Truncate long labels
            xlabels = [l[:15] + "..." if len(l) > 15 else l for l in xlabels]
            ylabels = [l[:15] + "..." if len(l) > 15 else l for l in ylabels]
        except:
            xlabels = [f"Task {c}" for c in prob_matrix.columns]
            ylabels = [f"Task {r}" for r in prob_matrix.index]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(prob_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                   xticklabels=xlabels, yticklabels=ylabels, ax=ax)
        
        # Add styling
        ax.set_title("Transition Probability Heatmap", fontsize=14)
        ax.set_xlabel("Next Activity")
        ax.set_ylabel("Current Activity")
        
        # Rotate x-axis labels for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        # Save figure
        return self._save_fig(fig, filename)
    
    def sankey_diagram(self, transitions, task_encoder, filename="sankey_diagram.html"):
        """
        Create Sankey diagram of process flow
        
        Args:
            transitions: DataFrame with transitions data
            task_encoder: Task label encoder
            filename: Output filename
            
        Returns:
            Path to saved figure or None
        """
        if not HAVE_PLOTLY:
            logger.error("Plotly is required for Sankey diagrams")
            return None
        
        # Calculate transition counts
        trans_count = transitions.groupby(["task_id", "next_task_id"]).size().reset_index(name='count')
        
        # Get unique tasks
        unique_tasks = set(transitions["task_id"].unique())
        unique_tasks.update(transitions["next_task_id"].dropna().unique())
        
        # Create node labels
        try:
            nodes = [task_encoder.inverse_transform([int(t)])[0] for t in unique_tasks]
            # Truncate long names
            nodes = [n[:25] + "..." if len(n) > 25 else n for n in nodes]
        except:
            nodes = [f"Task {t}" for t in unique_tasks]
        
        # Create node mapping
        node_idx = {t: i for i, t in enumerate(unique_tasks)}
        
        # Create links
        sources = []
        targets = []
        values = []
        
        for _, row in trans_count.iterrows():
            src, tgt, count = row["task_id"], row["next_task_id"], row["count"]
            if pd.notna(tgt):  # Skip NaN targets
                sources.append(node_idx[src])
                targets.append(node_idx[tgt])
                values.append(int(count))
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values
            )
        )])
        
        # Update layout
        fig.update_layout(
            title_text="Process Flow Sankey Diagram",
            font_size=10
        )
        
        # Save figure
        if self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            fig.write_html(filepath)
            logger.info(f"Saved Sankey diagram to {filepath}")
            return filepath
        
        return None
    
    def resource_workload(self, resource_stats, filename="resource_workload.png"):
        """
        Create resource workload visualization
        
        Args:
            resource_stats: DataFrame with resource workload statistics
            filename: Output filename
            
        Returns:
            Path to saved figure or None
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Top resources by workload
        top_n = min(15, len(resource_stats))
        top_resources = resource_stats.head(top_n)
        
        # Create bar chart
        bars = ax.barh([f"Resource {r}" for r in top_resources.index], 
                      top_resources["workload_percentage"].values,
                      color=self.colors['primary'])
        
        # Add count annotations
        for i, bar in enumerate(bars):
            count = int(top_resources.iloc[i]["activity_count"])
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                   f"n={count}", va='center')
        
        # Add styling
        ax.set_title("Resource Workload Distribution", fontsize=14)
        ax.set_xlabel("Workload (%)")
        ax.set_ylabel("Resource")
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add Gini coefficient if available
        if hasattr(resource_stats, 'attrs') and 'gini_coefficient' in resource_stats.attrs:
            gini = resource_stats.attrs['gini_coefficient']
            ax.text(0.02, 0.98, f"Gini coefficient: {gini:.3f}",
                   transform=ax.transAxes, fontsize=10, va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Save figure
        return self._save_fig(fig, filename)
    
    def create_dashboard(self, df=None, cycle_times=None, bottleneck_stats=None, 
                      significant_bottlenecks=None, task_encoder=None,
                      filename="dashboard.html"):
        """
        Create comprehensive dashboard with multiple visualizations
        
        Args:
            df: Process data dataframe
            cycle_times: Array of case durations in hours
            bottleneck_stats: DataFrame with bottleneck statistics
            significant_bottlenecks: DataFrame with significant bottlenecks
            task_encoder: Task label encoder
            filename: Output filename
            
        Returns:
            Path to saved dashboard or None
        """
        if not HAVE_PLOTLY:
            logger.error("Plotly is required for dashboards")
            return None
        
        # Create dashboard figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Cycle Time Distribution", "Top Bottlenecks", 
                           "Activity Distribution", "Resource Utilization"]
        )
        
        # 1. Cycle time distribution
        if cycle_times is not None:
            # Calculate statistics
            mean_duration = np.mean(cycle_times)
            median_duration = np.median(cycle_times)
            p90 = np.percentile(cycle_times, 90)
            
            # Create histogram
            fig.add_trace(
                go.Histogram(
                    x=cycle_times,
                    nbinsx=30,
                    marker_color=self.colors['primary'],
                    name="Duration"
                ),
                row=1, col=1
            )
            
            # Add statistic lines
            fig.add_vline(x=mean_duration, line_dash="solid", line_color=self.colors['secondary'],
                         row=1, col=1)
            fig.add_vline(x=median_duration, line_dash="dash", line_color=self.colors['accent'],
                         row=1, col=1)
        
        # 2. Top bottlenecks
        if bottleneck_stats is not None and significant_bottlenecks is not None and task_encoder is not None:
            # Get top bottlenecks
            top_n = min(10, len(significant_bottlenecks))
            if top_n > 0:
                top_bottlenecks = significant_bottlenecks.head(top_n)
                
                # Create labels
                labels = []
                for _, row in top_bottlenecks.iterrows():
                    src_id, dst_id = int(row["task_id"]), int(row["next_task_id"])
                    try:
                        src_name = task_encoder.inverse_transform([src_id])[0]
                        dst_name = task_encoder.inverse_transform([dst_id])[0]
                        
                        # Truncate long names
                        if len(src_name) > 10:
                            src_name = src_name[:7] + "..."
                        if len(dst_name) > 10:
                            dst_name = dst_name[:7] + "..."
                            
                        labels.append(f"{src_name} → {dst_name}")
                    except:
                        labels.append(f"Task {src_id} → Task {dst_id}")
                
                # Create bar chart
                fig.add_trace(
                    go.Bar(
                        y=labels[::-1],  # Reverse for better display
                        x=top_bottlenecks["mean_hours"].values[::-1],
                        orientation='h',
                        marker_color=self.colors['primary'],
                        name="Wait Time"
                    ),
                    row=1, col=2
                )
        
        # 3. Activity distribution
        if df is not None:
            # Get activity counts
            activity_counts = df["task_id"].value_counts().nlargest(10)
            
            # Get activity names
            try:
                labels = [task_encoder.inverse_transform([int(t)])[0] for t in activity_counts.index]
                # Truncate long names
                labels = [l[:15] + "..." if len(l) > 15 else l for l in labels]
            except:
                labels = [f"Task {t}" for t in activity_counts.index]
            
            # Create bar chart
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=activity_counts.values,
                    marker_color=self.colors['primary'],
                    name="Activity Count"
                ),
                row=2, col=1
            )
        
        # 4. Resource utilization
        if df is not None:
            # Get resource counts
            resource_counts = df["resource_id"].value_counts().nlargest(10)
            
            # Create bar chart
            fig.add_trace(
                go.Bar(
                    x=[f"Resource {r}" for r in resource_counts.index],
                    y=resource_counts.values,
                    marker_color=self.colors['primary'],
                    name="Resource Count"
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Process Mining Dashboard",
            height=800,
            showlegend=False
        )
        
        # Save dashboard
        if self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            fig.write_html(filepath)
            logger.info(f"Saved dashboard to {filepath}")
            return filepath
        
        return None