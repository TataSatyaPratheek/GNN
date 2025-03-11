#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process analysis pipeline for ProcessMine CLI
"""

import os
import time
import logging
from termcolor import colored
import pandas as pd
import numpy as np

from processmine.core.experiment import save_metrics

logger = logging.getLogger(__name__)

def run_process_analysis(args, df, task_encoder, run_dir, analysis_dir=None, viz_dir=None):
    """
    Run process mining analysis
    
    Args:
        args: Command line arguments
        df: Process data dataframe
        task_encoder: Task label encoder
        run_dir: Base results directory
        analysis_dir: Directory for analysis results
        viz_dir: Directory for visualizations
        
    Returns:
        Dictionary with analysis results
    """
    start_time = time.time()
    
    # Create directories if not provided
    if analysis_dir is None:
        analysis_dir = os.path.join(run_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
    
    if viz_dir is None:
        viz_dir = os.path.join(run_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
    
    # Try to use vectorized versions of the analysis functions
    try:
        from processmine.process_mining.vectorized_analysis import (
            analyze_bottlenecks_vectorized as analyze_bottlenecks,
            analyze_cycle_times_vectorized as analyze_cycle_times,
            identify_process_variants,
            analyze_resource_workload
        )
    except ImportError:
        from processmine.process_mining.analysis import (
            analyze_bottlenecks,
            analyze_cycle_times
        )
        identify_process_variants = None
        analyze_resource_workload = None
    
    # Import other analysis functions
    from processmine.process_mining.analysis import (
        analyze_rare_transitions,
        perform_conformance_checking,
        analyze_transition_patterns,
        spectral_cluster_graph,
        build_task_adjacency
    )
    
    # Try to use unified visualization API
    try:
        from processmine.visualization.unified_viz import ProcessVisualizer
        visualizer = ProcessVisualizer(output_dir=viz_dir)
    except ImportError:
        visualizer = None
        # Import old visualization functions
        from processmine.visualization.process_viz import (
            plot_cycle_time_distribution,
            plot_process_flow,
            plot_transition_heatmap,
            create_sankey_diagram
        )
    
    results = {}
    
    try:
        # Analyze bottlenecks
        print(colored("🔍 Analyzing process bottlenecks...", "cyan"))
        bottleneck_stats, significant_bottlenecks = analyze_bottlenecks(df)
        
        # Save bottleneck data
        bottleneck_stats.to_csv(os.path.join(analysis_dir, "bottleneck_stats.csv"), index=False)
        significant_bottlenecks.to_csv(os.path.join(analysis_dir, "significant_bottlenecks.csv"), index=False)
        
        # Store in results
        results['bottlenecks'] = {
            'total': len(bottleneck_stats),
            'significant': len(significant_bottlenecks)
        }
        
        # Analyze cycle times
        print(colored("🔍 Analyzing cycle times...", "cyan"))
        case_merged, long_cases, cut95 = analyze_cycle_times(df)
        
        # Save cycle time data
        case_merged.to_csv(os.path.join(analysis_dir, "case_cycle_times.csv"), index=False)
        long_cases.to_csv(os.path.join(analysis_dir, "long_running_cases.csv"), index=False)
        
        # Store in results
        results['cycle_times'] = {
            'median': float(np.median(case_merged["duration_h"])),
            'mean': float(np.mean(case_merged["duration_h"])),
            'p95': float(cut95),
            'long_cases': len(long_cases)
        }
        
        # Visualize cycle times
        if visualizer:
            visualizer.cycle_time_distribution(
                case_merged["duration_h"].values,
                filename="cycle_time_distribution.png"
            )
        else:
            plot_cycle_time_distribution(
                case_merged["duration_h"].values,
                os.path.join(viz_dir, "cycle_time_distribution.png")
            )
        
        # Analyze process variants if available
        if identify_process_variants:
            print(colored("🔍 Identifying process variants...", "cyan"))
            variant_stats, variant_sequences = identify_process_variants(df)
            variant_stats.to_csv(os.path.join(analysis_dir, "process_variants.csv"), index=False)
            
            # Store in results
            results['variants'] = {
                'count': len(variant_stats),
                'top_variant_percentage': float(variant_stats['percentage'].max())
            }
        
        # Analyze resource workload if available
        if analyze_resource_workload:
            print(colored("🔍 Analyzing resource workload...", "cyan"))
            resource_stats = analyze_resource_workload(df)
            resource_stats.to_csv(os.path.join(analysis_dir, "resource_workload.csv"))
            
            # Store in results
            results['resources'] = {
                'count': len(resource_stats),
                'gini_coefficient': float(resource_stats.attrs.get('gini_coefficient', 0))
            }
            
            # Visualize resource workload
            if visualizer:
                visualizer.resource_workload(resource_stats, filename="resource_workload.png")
        
        # Analyze rare transitions
        print(colored("🔍 Identifying rare transitions...", "cyan"))
        rare_trans = analyze_rare_transitions(bottleneck_stats)
        rare_trans.to_csv(os.path.join(analysis_dir, "rare_transitions.csv"), index=False)
        
        # Perform conformance checking
        print(colored("🔍 Performing conformance checking...", "cyan"))
        try:
            replayed, n_deviant = perform_conformance_checking(df)
            conformance_metrics = {
                "total_traces": len(replayed),
                "conforming_traces": len(replayed) - n_deviant,
                "deviant_traces": n_deviant,
                "conformance": float((len(replayed) - n_deviant) / len(replayed)) if replayed else 0
            }
            save_metrics(conformance_metrics, run_dir, "conformance_metrics.json")
            
            # Store in results
            results['conformance'] = conformance_metrics
            
        except Exception as e:
            print(colored(f"⚠️ Conformance checking failed: {e}", "yellow"))
            results['conformance'] = {'error': str(e)}
            replayed, n_deviant = [], 0
        
        # Visualize process flow with bottlenecks
        print(colored("📊 Creating process flow visualization...", "cyan"))
        if visualizer:
            visualizer.process_flow(
                bottleneck_stats,
                task_encoder,
                significant_bottlenecks.head(10),
                filename="process_flow_bottlenecks.png"
            )
        else:
            plot_process_flow(
                bottleneck_stats, task_encoder, significant_bottlenecks.head(10),
                os.path.join(viz_dir, "process_flow_bottlenecks.png")
            )
        
        # Get transition patterns and create visualizations
        print(colored("📊 Analyzing transition patterns...", "cyan"))
        transitions, trans_count, prob_matrix = analyze_transition_patterns(df, viz_dir=viz_dir)
        
        # Save transition data
        transitions.to_csv(os.path.join(analysis_dir, "transitions.csv"), index=False)
        
        # Visualize transitions
        if visualizer:
            visualizer.transition_heatmap(
                transitions,
                task_encoder,
                filename="transition_probability_heatmap.png"
            )
            
            # Create Sankey diagram
            visualizer.sankey_diagram(
                transitions,
                task_encoder,
                filename="process_flow_sankey.html"
            )
        else:
            # Plot transition heatmap
            plot_transition_heatmap(
                transitions, task_encoder,
                os.path.join(viz_dir, "transition_probability_heatmap.png")
            )
            
            # Create Sankey diagram
            create_sankey_diagram(
                transitions, task_encoder,
                os.path.join(viz_dir, "process_flow_sankey.html")
            )
        
        # Store transition data in results
        results['transitions'] = {
            'count': len(transitions),
            'unique': len(trans_count)
        }
        
        # Create dashboard
        if visualizer:
            visualizer.create_dashboard(
                cycle_times=case_merged["duration_h"].values,
                bottleneck_stats=bottleneck_stats,
                significant_bottlenecks=significant_bottlenecks,
                transition_matrix=prob_matrix,
                resource_stats=resource_stats if analyze_resource_workload else None,
                task_encoder=task_encoder,
                filename="process_dashboard.html"
            )
        
        # Overall process summary
        process_summary = {
            "num_cases": df["case_id"].nunique(),
            "num_events": len(df),
            "num_activities": len(df["task_id"].unique()),
            "num_resources": len(df["resource_id"].unique()),
            "num_long_cases": len(long_cases),
            "cycle_time_95th_percentile": float(cut95),
            "num_significant_bottlenecks": len(significant_bottlenecks),
            "num_rare_transitions": len(rare_trans),
            "num_deviant_traces": n_deviant if 'n_deviant' in locals() else 0,
            "total_traces": len(replayed) if 'replayed' in locals() and replayed else df["case_id"].nunique(),
            "analysis_time": time.time() - start_time
        }
        save_metrics(process_summary, run_dir, "process_analysis.json")
        
        # Store in results
        results['summary'] = process_summary
        
    except Exception as e:
        logger.error(f"Error in process mining analysis: {e}", exc_info=True)
        results['error'] = str(e)
    
    return results