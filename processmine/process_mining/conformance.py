#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Conformance checking utilities for process mining
"""

import pandas as pd
import time
import logging

logger = logging.getLogger(__name__)

# Check for PM4Py availability
try:
    from pm4py.objects.log.util import dataframe_utils
    from pm4py.objects.conversion.log import converter as log_converter
    from pm4py.algo.discovery.inductive import algorithm as inductive_miner
    from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
    PM4PY_AVAILABLE = True
except ImportError:
    PM4PY_AVAILABLE = False
    logger.warning("PM4Py not available. Conformance checking functionality will be limited.")

def perform_conformance_checking(df):
    """
    Perform conformance checking using inductive miner and token replay
    
    Args:
        df: Process data dataframe
        
    Returns:
        Tuple of (replayed_traces, num_deviant_traces)
    """
    if not PM4PY_AVAILABLE:
        print("\033[93mPM4Py not available. Skipping conformance checking.\033[0m")
        return [], 0
    
    print("\n==== Performing Conformance Checking ====")
    start_time = time.time()
    
    print("Preparing event log...")
    # Prepare dataframe for PM4Py
    df_pm = df[["case_id","task_name","timestamp"]].rename(columns={
        "case_id": "case:concept:name",
        "task_name": "concept:name",
        "timestamp": "time:timestamp"
    })
    
    df_pm = dataframe_utils.convert_timestamp_columns_in_df(df_pm)
    
    print("Converting to event log format...")
    event_log = log_converter.apply(df_pm)
    
    print("Discovering process model...")
    process_tree = inductive_miner.apply(event_log)
    from pm4py.objects.conversion.process_tree import converter as pt_converter
    net, im, fm = pt_converter.apply(process_tree)
    
    print("Performing token replay...")
    # Add progress bar for token replay if possible
    try:
        from pm4py.objects.conversion.process_tree import converter as pt_converter
        replayed = token_replay.apply(event_log, net, im, fm)
    except Exception as e:
        print(f"\033[91mError during token replay: {e}\033[0m")
        print("Falling back to simplified conformance checking...")
        # Simple fallback - just count variants
        variants = df.groupby("case_id")["task_name"].agg(lambda x: tuple(x)).value_counts()
        top_variant_count = variants.iloc[0]
        total_cases = len(variants)
        conformance = top_variant_count / total_cases
        print(f"Top variant covers {conformance:.1%} of cases ({top_variant_count}/{total_cases})")
        return [], 0
    
    # Count non-conforming traces
    n_deviant = sum(1 for t in replayed if not t["trace_is_fit"])
    fit_percentage = (len(replayed) - n_deviant) / len(replayed) * 100 if replayed else 0
    
    print(f"\033[1mConformance Results\033[0m:")
    print(f"  Model-conforming traces: \033[96m{len(replayed) - n_deviant}\033[0m ({fit_percentage:.1f}%)")
    print(f"  Deviant traces: \033[93m{n_deviant}\033[0m ({100-fit_percentage:.1f}%)")
    print(f"  Total traces: {len(replayed)}")
    print(f"Analysis completed in \033[96m{time.time() - start_time:.2f}s\033[0m")
    
    return replayed, n_deviant

def identify_conformance_violations(df, replayed_traces):
    """
    Identify specific conformance violations in the process
    
    Args:
        df: Process data dataframe
        replayed_traces: Replayed traces from token replay
        
    Returns:
        Dataframe of conformance violations
    """
    if not PM4PY_AVAILABLE or not replayed_traces:
        return pd.DataFrame()
    
    print("\n==== Identifying Conformance Violations ====")
    
    # Extract non-conforming traces
    non_conforming = [t for t in replayed_traces if not t["trace_is_fit"]]
    
    # Extract case IDs from non-conforming traces
    non_conforming_cases = []
    for trace in non_conforming:
        trace_attributes = trace["trace_attributes"]
        case_id = trace_attributes["concept:name"]
        non_conforming_cases.append(case_id)
    
    # Filter dataframe for non-conforming cases
    violations_df = df[df["case_id"].isin(non_conforming_cases)].copy()
    
    # Add violation flag
    violations_df["is_violation"] = True
    
    print(f"Identified {len(non_conforming_cases)} non-conforming cases")
    
    return violations_df