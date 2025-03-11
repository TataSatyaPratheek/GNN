#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Report generation for ProcessMine CLI
"""

import os
import time
import logging
from datetime import datetime
from termcolor import colored
import json

from processmine.core.experiment import save_metrics

logger = logging.getLogger(__name__)

def generate_report(args, data, model, metrics, analysis_results, run_dir, total_start_time):
    """
    Generate final summary report
    
    Args:
        args: Command line arguments
        data: Dictionary with processed data
        model: Trained model
        metrics: Model metrics
        analysis_results: Process analysis results
        run_dir: Results directory
        total_start_time: Start time of processing
        
    Returns:
        Path to generated report
    """
    try:
        df = data['df']
        total_duration = time.time() - total_start_time
        
        # Create summary
        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration_seconds": total_duration,
            "total_duration_formatted": f"{total_duration//3600:.0f}h {(total_duration%3600)//60:.0f}m {total_duration%60:.0f}s",
            "dataset": {
                "filename": os.path.basename(args.data_path),
                "cases": df["case_id"].nunique(),
                "events": len(df),
                "activities": len(df["task_id"].unique()),
                "resources": len(df["resource_id"].unique())
            },
            "models": {
                args.model_type: {
                    "accuracy": float(metrics['accuracy']) if metrics and 'accuracy' in metrics else 0
                }
            },
            "process_analysis": {
                "bottlenecks": analysis_results.get('bottlenecks', {}).get('significant', 0),
                "median_cycle_time": analysis_results.get('cycle_times', {}).get('median', 0),
                "p95_cycle_time": analysis_results.get('cycle_times', {}).get('p95', 0)
            }
        }
        
        # Save summary as JSON
        save_metrics(summary, run_dir, "execution_summary.json")
        
        # Generate summary report in markdown format
        report_path = os.path.join(run_dir, "execution_summary.md")
        
        with open(report_path, 'w') as f:
            f.write(f"# Process Mining Execution Summary\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write(f"**Time:** {datetime.now().strftime('%H:%M:%S')}\n")
            f.write(f"**Duration:** {summary['total_duration_formatted']}\n\n")
            
            f.write(f"## Dataset Information\n\n")
            f.write(f"- **Filename:** {summary['dataset']['filename']}\n")
            f.write(f"- **Cases:** {summary['dataset']['cases']:,}\n")
            f.write(f"- **Events:** {summary['dataset']['events']:,}\n")
            f.write(f"- **Activities:** {summary['dataset']['activities']}\n")
            f.write(f"- **Resources:** {summary['dataset']['resources']}\n\n")
            
            f.write(f"## Model Performance\n\n")
            if metrics and 'accuracy' in metrics:
                f.write(f"- **{args.model_type.replace('_', ' ').title()} Accuracy:** {metrics['accuracy']:.4f}\n")
                if 'f1_weighted' in metrics:
                    f.write(f"- **F1 Score (weighted):** {metrics['f1_weighted']:.4f}\n")
                if 'mcc' in metrics:
                    f.write(f"- **Matthews Correlation Coefficient:** {metrics['mcc']:.4f}\n")
            
            f.write(f"\n## Process Analysis\n\n")
            bottlenecks = analysis_results.get('bottlenecks', {}).get('significant', 0)
            median_cycle = analysis_results.get('cycle_times', {}).get('median', 0)
            p95_cycle = analysis_results.get('cycle_times', {}).get('p95', 0)
            
            f.write(f"- **Significant Bottlenecks:** {bottlenecks}\n")
            f.write(f"- **Median Cycle Time:** {median_cycle:.2f} hours\n")
            f.write(f"- **95th Percentile Cycle Time:** {p95_cycle:.2f} hours\n\n")
            
            if 'variants' in analysis_results:
                variant_count = analysis_results['variants']['count']
                top_pct = analysis_results['variants']['top_variant_percentage']
                f.write(f"- **Process Variants:** {variant_count}\n")
                f.write(f"- **Top Variant Coverage:** {top_pct:.1f}%\n")
            
            if 'conformance' in analysis_results:
                conformance = analysis_results['conformance']
                if 'error' not in conformance:
                    conf_pct = conformance.get('conformance', 0) * 100
                    f.write(f"- **Process Conformance:** {conf_pct:.1f}%\n")
                    f.write(f"- **Deviant Traces:** {conformance.get('deviant_traces', 0)}\n")
            
            f.write(f"\n## Generated Artifacts\n\n")
            f.write(f"- **Models:** {args.model_type.replace('_', ' ').title()}\n")
            f.write(f"- **Visualizations:** Confusion matrices, process flow, bottlenecks, transitions, Sankey diagram\n")
            f.write(f"- **Analysis:** Bottlenecks, cycle times, transitions, variants\n")
            if not args.skip_rl:
                f.write(f"- **Policies:** RL optimization policies\n")
            if args.run_ablation:
                f.write(f"- **Ablation:** Component contribution analysis\n")
        
        print(colored(f"\n✅ Execution summary saved to {report_path}", "green"))
        
        return report_path
        
    except Exception as e:
        logger.error(f"Error creating summary report: {e}", exc_info=True)
        return None