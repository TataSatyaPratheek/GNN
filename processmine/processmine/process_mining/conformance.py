# processmine/process_mining/conformance.py (UPDATED FILE)

"""
Comprehensive conformance checking for process mining
"""

import pandas as pd
import numpy as np
import time
import logging
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)

# Check for PM4Py availability
try:
    from pm4py.objects.log.util import dataframe_utils
    from pm4py.objects.conversion.log import converter as log_converter
    from pm4py.algo.discovery.inductive import algorithm as inductive_miner
    from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
    from pm4py.algo.conformance.alignments import algorithm as alignments
    PM4PY_AVAILABLE = True
except ImportError:
    PM4PY_AVAILABLE = False
    logger.warning("PM4Py not available. Conformance checking functionality will be limited.")


class ViolationType(Enum):
    """Types of conformance violations"""
    WRONG_ACTIVITY = "wrong_activity"
    SKIPPED_ACTIVITY = "skipped_activity"
    DUPLICATE_ACTIVITY = "duplicate_activity"
    WRONG_SEQUENCE = "wrong_sequence"
    INCOMPLETE_CASE = "incomplete_case"
    INVALID_TRANSITION = "invalid_transition"
    OTHER = "other"


class ConformanceChecker:
    """Class for comprehensive conformance checking"""
    
    def __init__(self, df: pd.DataFrame, use_token_replay: bool = True, use_alignments: bool = False):
        """
        Initialize conformance checker
        
        Args:
            df: Process dataframe
            use_token_replay: Whether to use token replay
            use_alignments: Whether to use alignments (more accurate but slower)
        """
        self.df = df
        self.use_token_replay = use_token_replay
        self.use_alignments = use_alignments
        
        if not PM4PY_AVAILABLE:
            logger.warning("PM4Py not available. Falling back to simplified conformance checking.")
        
        # Initialize results containers
        self.process_model = None
        self.conformance_results = None
        self.conforming_cases = []
        self.non_conforming_cases = []
        self.violations = pd.DataFrame()
        
        # Store original column names to handle different naming conventions
        self.orig_cols = df.columns.tolist()
    
    def check_conformance(self) -> Dict[str, Any]:
        """
        Check conformance using most appropriate method
        
        Returns:
            Dictionary of conformance results
        """
        start_time = time.time()
        
        if PM4PY_AVAILABLE:
            return self._check_with_pm4py()
        else:
            return self._check_simplified()
    
    def _check_with_pm4py(self) -> Dict[str, Any]:
        """
        Check conformance using PM4Py
        
        Returns:
            Dictionary of conformance results
        """
        # Prepare dataframe for PM4Py
        df_pm = self._prepare_pm4py_dataframe()
        
        try:
            # Convert to event log
            event_log = log_converter.apply(df_pm)
            
            # Discover process model
            logger.info("Discovering process model...")
            process_tree = inductive_miner.apply(event_log)
            self.process_model = process_tree
            
            # Convert to Petri net
            from pm4py.objects.conversion.process_tree import converter as pt_converter
            net, im, fm = pt_converter.apply(process_tree)
            
            # Perform conformance checking
            logger.info("Performing conformance checking...")
            
            if self.use_token_replay:
                # Token replay (faster)
                self.conformance_results = token_replay.apply(event_log, net, im, fm)
                return self._process_token_replay_results()
            
            elif self.use_alignments:
                # Alignments (more accurate but slower)
                self.conformance_results = alignments.apply(event_log, net, im, fm)
                return self._process_alignment_results()
            
        except Exception as e:
            logger.error(f"Error during conformance checking: {e}")
            # Fall back to simplified method
            return self._check_simplified()
    
    def _prepare_pm4py_dataframe(self) -> pd.DataFrame:
        """
        Prepare dataframe for PM4Py
        
        Returns:
            Formatted dataframe
        """
        df_pm = self.df.copy()
        
        # Map columns to PM4Py expected format
        col_map = {}
        
        # Find case ID column
        case_id_candidates = ['case_id', 'case:concept:name', 'case:id', 'caseid']
        for col in case_id_candidates:
            if col in df_pm.columns:
                col_map['case:concept:name'] = col
                break
        
        # Find activity column
        activity_candidates = ['task_name', 'concept:name', 'activity', 'event']
        for col in activity_candidates:
            if col in df_pm.columns:
                col_map['concept:name'] = col
                break
        
        # Find timestamp column
        time_candidates = ['timestamp', 'time:timestamp', 'time', 'date']
        for col in time_candidates:
            if col in df_pm.columns:
                col_map['time:timestamp'] = col
                break
        
        # Rename columns
        for pm4py_col, df_col in col_map.items():
            if df_col in df_pm.columns and pm4py_col != df_col:
                df_pm = df_pm.rename(columns={df_col: pm4py_col})
        
        # Ensure necessary columns exist
        required_columns = ['case:concept:name', 'concept:name', 'time:timestamp']
        missing_columns = [col for col in required_columns if col not in df_pm.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert timestamp to datetime
        if 'time:timestamp' in df_pm.columns:
            df_pm['time:timestamp'] = pd.to_datetime(df_pm['time:timestamp'])
        
        # Execute PM4Py preprocessing
        df_pm = dataframe_utils.convert_timestamp_columns_in_df(df_pm)
        
        return df_pm
    
    def _process_token_replay_results(self) -> Dict[str, Any]:
        """
        Process token replay results
        
        Returns:
            Dictionary of conformance results
        """
        # Extract conforming and non-conforming cases
        conforming = []
        non_conforming = []
        violations_list = []
        
        for trace_result in self.conformance_results:
            case_id = trace_result["trace_attributes"]["concept:name"]
            is_fit = trace_result["trace_is_fit"]
            
            if is_fit:
                conforming.append(case_id)
            else:
                non_conforming.append(case_id)
                
                # Analyze violations
                if "missing_tokens" in trace_result and trace_result["missing_tokens"] > 0:
                    # Missing tokens indicate skipped activities
                    violations_list.append({
                        "case_id": case_id,
                        "violation_type": ViolationType.SKIPPED_ACTIVITY.value,
                        "count": trace_result["missing_tokens"],
                        "fitness": trace_result.get("trace_fitness", 0)
                    })
                
                if "remaining_tokens" in trace_result and trace_result["remaining_tokens"] > 0:
                    # Remaining tokens indicate additional activities
                    violations_list.append({
                        "case_id": case_id,
                        "violation_type": ViolationType.DUPLICATE_ACTIVITY.value,
                        "count": trace_result["remaining_tokens"],
                        "fitness": trace_result.get("trace_fitness", 0)
                    })
                
                if "produced_tokens" in trace_result and "consumed_tokens" in trace_result:
                    # Compare produced and consumed tokens for sequence violations
                    if trace_result["produced_tokens"] > trace_result["consumed_tokens"]:
                        violations_list.append({
                            "case_id": case_id,
                            "violation_type": ViolationType.WRONG_SEQUENCE.value,
                            "count": trace_result["produced_tokens"] - trace_result["consumed_tokens"],
                            "fitness": trace_result.get("trace_fitness", 0)
                        })
        
        # Store results
        self.conforming_cases = conforming
        self.non_conforming_cases = non_conforming
        
        if violations_list:
            self.violations = pd.DataFrame(violations_list)
        
        # Calculate statistics
        total_cases = len(conforming) + len(non_conforming)
        conformance_ratio = len(conforming) / total_cases if total_cases > 0 else 0
        
        # Count violation types
        violation_counts = {}
        if not self.violations.empty:
            violation_counts = self.violations['violation_type'].value_counts().to_dict()
        
        return {
            "total_cases": total_cases,
            "conforming_cases": len(conforming),
            "non_conforming_cases": len(non_conforming),
            "conformance_ratio": conformance_ratio,
            "violations": violation_counts
        }
    
    def _process_alignment_results(self) -> Dict[str, Any]:
        """
        Process alignment results
        
        Returns:
            Dictionary of conformance results
        """
        # Parse alignment results
        alignments = self.conformance_results
        
        conforming = []
        non_conforming = []
        violations_list = []
        
        # Trace index to case ID mapping
        case_id_mapping = {}
        for idx, group in enumerate(self.df.groupby('case_id')):
            case_id_mapping[idx] = group[0]  # group[0] is the case_id
        
        for idx, alignment in enumerate(alignments):
            case_id = case_id_mapping.get(idx, f"Case_{idx}")
            
            # Check if alignment is perfect
            alignment_cost = alignment['cost']
            is_fit = alignment_cost == 0
            
            if is_fit:
                conforming.append(case_id)
            else:
                non_conforming.append(case_id)
                
                # Analyze alignment for violations
                alignment_steps = alignment.get('alignment', [])
                for step in alignment_steps:
                    if step[0][0] == '>>':  # Model move (skipped activity)
                        violations_list.append({
                            "case_id": case_id,
                            "violation_type": ViolationType.SKIPPED_ACTIVITY.value,
                            "activity": step[0][1],
                            "cost": alignment_cost
                        })
                    
                    elif step[1][0] == '>>':  # Log move (unexpected activity)
                        violations_list.append({
                            "case_id": case_id,
                            "violation_type": ViolationType.WRONG_ACTIVITY.value,
                            "activity": step[1][1],
                            "cost": alignment_cost
                        })
        
        # Store results
        self.conforming_cases = conforming
        self.non_conforming_cases = non_conforming
        
        if violations_list:
            self.violations = pd.DataFrame(violations_list)
        
        # Calculate statistics
        total_cases = len(conforming) + len(non_conforming)
        conformance_ratio = len(conforming) / total_cases if total_cases > 0 else 0
        
        # Count violation types
        violation_counts = {}
        if not self.violations.empty:
            violation_counts = self.violations['violation_type'].value_counts().to_dict()
        
        return {
            "total_cases": total_cases,
            "conforming_cases": len(conforming),
            "non_conforming_cases": len(non_conforming),
            "conformance_ratio": conformance_ratio,
            "violations": violation_counts
        }
    
    def _check_simplified(self) -> Dict[str, Any]:
        """
        Simple variant-based conformance checking when PM4Py is not available
        
        Returns:
            Dictionary of conformance results
        """
        logger.info("Using simplified conformance checking")
        
        # Identify process variants
        case_sequences = {}
        
        # Check column availability
        case_id_col = 'case_id'
        task_col = 'task_name'
        time_col = 'timestamp'
        
        # Find appropriate columns
        if case_id_col not in self.df.columns:
            candidates = [col for col in self.df.columns if 'case' in col.lower()]
            case_id_col = candidates[0] if candidates else None
        
        if task_col not in self.df.columns:
            candidates = [col for col in self.df.columns if 'task' in col.lower() or 'activity' in col.lower()]
            task_col = candidates[0] if candidates else None
        
        if time_col not in self.df.columns:
            candidates = [col for col in self.df.columns if 'time' in col.lower() or 'date' in col.lower()]
            time_col = candidates[0] if candidates else None
        
        if not all([case_id_col, task_col, time_col]):
            logger.error("Missing required columns for conformance checking")
            return {
                "error": "Missing required columns",
                "conformance_ratio": 0.0
            }
        
        # Get case sequences
        for case_id, case_df in self.df.groupby(case_id_col):
            # Sort by timestamp
            case_df = case_df.sort_values(time_col)
            # Convert sequence to tuple for hashing
            sequence = tuple(case_df[task_col].values)
            case_sequences[case_id] = sequence
        
        # Count sequence frequencies
        sequence_counts = {}
        for sequence in case_sequences.values():
            sequence_counts[sequence] = sequence_counts.get(sequence, 0) + 1
        
        # Sort variants by frequency
        sorted_variants = sorted(sequence_counts.items(), key=lambda x: x[1], reverse=True)
        
        # The most frequent variant is considered the "happy path"
        if sorted_variants:
            happy_path = sorted_variants[0][0]
            
            # Cases following the happy path are conforming
            self.conforming_cases = [
                case_id for case_id, sequence in case_sequences.items()
                if sequence == happy_path
            ]
            
            # Other cases are non-conforming
            self.non_conforming_cases = [
                case_id for case_id, sequence in case_sequences.items()
                if sequence != happy_path
            ]
            
            # Identify violation types
            violations_list = []
            
            for case_id, sequence in case_sequences.items():
                if sequence != happy_path:
                    # Compare with happy path to identify violations
                    if len(sequence) < len(happy_path):
                        violations_list.append({
                            "case_id": case_id,
                            "violation_type": ViolationType.INCOMPLETE_CASE.value,
                            "count": len(happy_path) - len(sequence)
                        })
                    elif len(sequence) > len(happy_path):
                        violations_list.append({
                            "case_id": case_id,
                            "violation_type": ViolationType.DUPLICATE_ACTIVITY.value,
                            "count": len(sequence) - len(happy_path)
                        })
                    else:
                        violations_list.append({
                            "case_id": case_id,
                            "violation_type": ViolationType.WRONG_SEQUENCE.value,
                            "count": sum(1 for i in range(len(sequence)) if sequence[i] != happy_path[i])
                        })
            
            if violations_list:
                self.violations = pd.DataFrame(violations_list)
            
            # Calculate statistics
            total_cases = len(case_sequences)
            conformance_ratio = len(self.conforming_cases) / total_cases
            
            # Count violation types
            violation_counts = {}
            if not self.violations.empty:
                violation_counts = self.violations['violation_type'].value_counts().to_dict()
            
            return {
                "total_cases": total_cases,
                "conforming_cases": len(self.conforming_cases),
                "non_conforming_cases": len(self.non_conforming_cases),
                "conformance_ratio": conformance_ratio,
                "violations": violation_counts,
                "happy_path_frequency": sequence_counts[happy_path],
                "total_variants": len(sequence_counts)
            }
        else:
            return {
                "total_cases": 0,
                "conforming_cases": 0,
                "non_conforming_cases": 0,
                "conformance_ratio": 0.0,
                "violations": {}
            }
    
    def get_violations_dataframe(self) -> pd.DataFrame:
        """
        Get violations as a dataframe
        
        Returns:
            DataFrame with violations
        """
        if self.violations.empty:
            return pd.DataFrame(columns=['case_id', 'violation_type', 'count'])
        return self.violations
    
    def get_violating_cases(self) -> pd.DataFrame:
        """
        Get events from cases with violations
        
        Returns:
            DataFrame with events from violating cases
        """
        if not self.non_conforming_cases:
            return pd.DataFrame()
        
        case_id_col = 'case_id'
        if case_id_col not in self.df.columns:
            candidates = [col for col in self.df.columns if 'case' in col.lower()]
            case_id_col = candidates[0] if candidates else None
        
        if not case_id_col:
            return pd.DataFrame()
        
        return self.df[self.df[case_id_col].isin(self.non_conforming_cases)].copy()