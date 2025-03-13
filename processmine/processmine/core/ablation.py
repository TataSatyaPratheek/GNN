"""
Comprehensive ablation study module for evaluating model components
with parallel execution, structured reporting, and visualization capabilities.
"""
import os
import time
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Callable, Optional, Union
from pathlib import Path
import concurrent.futures
from copy import deepcopy

logger = logging.getLogger(__name__)

class AblationStudy:
    """
    Comprehensive ablation study manager for evaluating model components
    
    Allows systematic testing of different model configurations by enabling/disabling
    components, running experiments in parallel, and comparing performance.
    """
    
    def __init__(
        self,
        base_config: Dict[str, Any],
        output_dir: Union[str, Path],
        experiment_name: str = "ablation_study",
        device: Optional[Any] = None,
        save_models: bool = True,
        parallel: bool = False,
        max_workers: int = 4,
        random_seeds: Optional[List[int]] = None
    ):
        """
        Initialize ablation study manager
        
        Args:
            base_config: Base configuration dictionary for experiments
            output_dir: Directory to save results
            experiment_name: Name for this ablation study
            device: Computing device to use
            save_models: Whether to save model checkpoints
            parallel: Whether to run experiments in parallel
            max_workers: Maximum number of parallel workers
            random_seeds: List of random seeds for each experiment (for reproducibility)
        """
        self.base_config = base_config.copy()
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.device = device
        self.save_models = save_models
        self.parallel = parallel
        self.max_workers = max_workers
        self.random_seeds = random_seeds or [42]
        
        # Create experiments directory
        self.study_dir = self.output_dir / experiment_name
        os.makedirs(self.study_dir, exist_ok=True)
        
        # Experiments to run
        self.experiments = {}
        
        # Results storage
        self.results = {}
        
        # Save base configuration
        self._save_config(self.base_config, "base_config.json")
        
        logger.info(f"Initialized ablation study '{experiment_name}' with {len(self.random_seeds)} random seeds")
    
    def _save_config(self, config: Dict[str, Any], filename: str) -> None:
        """Save config to file"""
        file_path = self.study_dir / filename
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def add_experiment(self, name: str, config_mods: Dict[str, Any]) -> None:
        """
        Add experiment with configuration modifications
        
        Args:
            name: Experiment name
            config_mods: Configuration modifications to apply to base config
        """
        if name in self.experiments:
            logger.warning(f"Experiment '{name}' already exists and will be overwritten")
        
        # Create experiment config by applying modifications to base config
        experiment_config = deepcopy(self.base_config)
        for key, value in config_mods.items():
            # Handle nested keys with dot notation
            if '.' in key:
                parts = key.split('.')
                current = experiment_config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                experiment_config[key] = value
        
        # Add experiment name to config
        experiment_config['name'] = name
        
        # Store experiment
        self.experiments[name] = {
            'config': experiment_config,
            'config_mods': config_mods
        }
        
        logger.info(f"Added experiment '{name}' with {len(config_mods)} configuration modifications")
    
    def add_ablation_experiments(
        self, 
        components: List[str], 
        disable: bool = True,
        include_combinations: bool = False
    ) -> None:
        """
        Add experiments by enabling/disabling specific components
        
        Args:
            components: List of component names to ablate
            disable: Whether to disable (True) or enable (False) components
            include_combinations: Whether to include combinatorial experiments
        """
        from itertools import combinations
        
        # Add individual component ablations
        for component in components:
            # Create experiment name
            name = f"no_{component}" if disable else f"only_{component}"
            
            # Create config modification
            config_mod = {component: not disable}
            
            # Add experiment
            self.add_experiment(name, config_mod)
        
        # Add combinations if requested
        if include_combinations and len(components) > 1:
            for r in range(2, len(components) + 1):
                for combo in combinations(components, r):
                    # Create experiment name
                    combo_name = "_".join(combo)
                    name = f"no_{combo_name}" if disable else f"only_{combo_name}"
                    
                    # Create config modifications
                    config_mods = {c: not disable for c in combo}
                    
                    # Add experiment
                    self.add_experiment(name, config_mods)
    
    def add_grid_search(self, param_grid: Dict[str, List[Any]]) -> None:
        """
        Add experiments for grid search over parameters
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values
        """
        import itertools
        
        # Get all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for values in itertools.product(*param_values):
            # Create config modifications
            config_mods = {name: value for name, value in zip(param_names, values)}
            
            # Create experiment name
            name_parts = [f"{name}_{value}" for name, value in config_mods.items()]
            name = "grid_" + "_".join(name_parts)
            
            # Add experiment
            self.add_experiment(name, config_mods)
        
        logger.info(f"Added grid search with {len(self.experiments)} parameter combinations")
    
    def run_experiments(
        self,
        run_function: Callable,
        aggregate_metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run all registered experiments
        
        Args:
            run_function: Function to run each experiment (takes config, returns metrics)
            aggregate_metrics: List of metrics to report across seeds (mean and std)
            
        Returns:
            Dictionary of experiment results
        """
        start_time = time.time()
        logger.info(f"Starting ablation study with {len(self.experiments)} experiments")
        
        if not self.experiments:
            logger.warning("No experiments registered. Add experiments before running.")
            return {}
        
        # Create directory for experiment results
        results_dir = self.study_dir / "experiments"
        os.makedirs(results_dir, exist_ok=True)
        
        # Run experiments
        if self.parallel and len(self.experiments) > 1:
            # Run experiments in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for exp_name, exp_info in self.experiments.items():
                    for seed_idx, seed in enumerate(self.random_seeds):
                        # Create experiment config with seed
                        config = exp_info['config'].copy()
                        config['seed'] = seed
                        
                        # Submit experiment
                        future = executor.submit(self._run_single_experiment, run_function, config, self.device)
                        futures[(exp_name, seed_idx)] = future
                
                # Collect results
                for (exp_name, seed_idx), future in futures.items():
                    try:
                        result = future.result()
                        if exp_name not in self.results:
                            self.results[exp_name] = []
                        self.results[exp_name].append(result)
                    except Exception as e:
                        logger.error(f"Error in experiment {exp_name} (seed {seed_idx}): {e}")
        else:
            # Run experiments sequentially
            for exp_name, exp_info in self.experiments.items():
                logger.info(f"Running experiment '{exp_name}'")
                self.results[exp_name] = []
                
                for seed_idx, seed in enumerate(self.random_seeds):
                    # Create experiment config with seed
                    config = exp_info['config'].copy()
                    config['seed'] = seed
                    
                    # Run experiment
                    try:
                        result = self._run_single_experiment(run_function, config, self.device)
                        self.results[exp_name].append(result)
                    except Exception as e:
                        logger.error(f"Error in experiment {exp_name} (seed {seed_idx}): {e}")
        
        # Aggregate results across seeds
        aggregated_results = self._aggregate_results(aggregate_metrics)
        
        # Save results
        self._save_results(aggregated_results)
        
        # Create visualizations
        if aggregate_metrics:
            self._create_visualizations(aggregated_results, aggregate_metrics)
        
        duration = time.time() - start_time
        logger.info(f"Ablation study completed in {duration:.2f}s")
        
        return aggregated_results
    
    def _run_single_experiment(
        self, 
        run_function: Callable, 
        config: Dict[str, Any],
        device: Any
    ) -> Dict[str, Any]:
        """Run a single experiment with the given config"""
        start_time = time.time()
        
        # Create experiment output directory
        exp_name = config.get('name', 'unnamed')
        exp_dir = self.study_dir / "experiments" / exp_name
        os.makedirs(exp_dir, exist_ok=True)
        
        # Add output directory to config if not present
        if 'output_dir' not in config:
            config['output_dir'] = str(exp_dir)
        
        # Add device to config if not present
        if 'device' not in config and device is not None:
            config['device'] = device
        
        # Run experiment
        result = run_function(config, device)
        
        # Save experiment config
        self._save_config(config, f"experiments/{exp_name}/config.json")
        
        # Log completion
        duration = time.time() - start_time
        logger.info(f"Experiment '{exp_name}' completed in {duration:.2f}s")
        
        return result
    
    def _aggregate_results(self, metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Aggregate results across random seeds
        
        Args:
            metrics: List of metric names to aggregate
            
        Returns:
            Dictionary of aggregated results
        """
        if not metrics:
            # Just return the raw results
            return {name: results for name, results in self.results.items()}
        
        # Initialize aggregated results
        aggregated = {}
        
        for exp_name, exp_results in self.results.items():
            exp_agg = {'name': exp_name}
            
            # Skip if no results
            if not exp_results:
                aggregated[exp_name] = exp_agg
                continue
            
            # Get modifications for this experiment
            exp_agg['modifications'] = self.experiments[exp_name]['config_mods']
            
            # Add number of seeds
            exp_agg['num_seeds'] = len(exp_results)
            
            # Aggregate metrics
            for metric in metrics:
                # Get metric values across seeds
                try:
                    metric_values = [result.get(metric, float('nan')) for result in exp_results]
                    metric_values = [v for v in metric_values if not np.isnan(v)]
                    
                    if metric_values:
                        exp_agg[f"{metric}_mean"] = float(np.mean(metric_values))
                        exp_agg[f"{metric}_std"] = float(np.std(metric_values))
                        exp_agg[f"{metric}_values"] = metric_values
                except Exception as e:
                    logger.warning(f"Error aggregating metric '{metric}' for experiment '{exp_name}': {e}")
            
            # Add to aggregated results
            aggregated[exp_name] = exp_agg
        
        return aggregated
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save results to file"""
        # Save as JSON
        results_path = self.study_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.float32, np.float64)) 
                                                           else (int(o) if isinstance(o, (np.int32, np.int64)) 
                                                                 else str(o)))
        
        # Try to save as CSV for easier analysis
        try:
            # Convert to dataframe
            rows = []
            for exp_name, exp_results in results.items():
                row = {'experiment': exp_name}
                row.update(exp_results.get('modifications', {}))
                
                # Add metrics
                for k, v in exp_results.items():
                    if k not in ['name', 'modifications', 'num_seeds', 'values']:
                        row[k] = v
                
                rows.append(row)
            
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(self.study_dir / "results.csv", index=False)
        except Exception as e:
            logger.warning(f"Error saving results as CSV: {e}")
    
    def _create_visualizations(
        self, 
        results: Dict[str, Any],
        metrics: List[str]
    ) -> None:
        """
        Create visualizations of ablation study results
        
        Args:
            results: Dictionary of aggregated results
            metrics: List of metrics to visualize
        """
        try:
            # Create visualizations directory
            viz_dir = self.study_dir / "visualizations"
            os.makedirs(viz_dir, exist_ok=True)
            
            # Prepare data for plotting
            exp_names = list(results.keys())
            
            # Create metric comparison plot
            for metric in metrics:
                plt.figure(figsize=(12, 6))
                
                # Collect metric values and errors
                mean_key = f"{metric}_mean"
                std_key = f"{metric}_std"
                
                means = []
                errors = []
                labels = []
                
                for exp_name in exp_names:
                    exp_results = results[exp_name]
                    if mean_key in exp_results and std_key in exp_results:
                        means.append(exp_results[mean_key])
                        errors.append(exp_results[std_key])
                        labels.append(exp_name)
                
                # Sort by mean value
                sorted_indices = np.argsort(means)
                means = [means[i] for i in sorted_indices]
                errors = [errors[i] for i in sorted_indices]
                labels = [labels[i] for i in sorted_indices]
                
                # Create bar chart
                plt.barh(labels, means, xerr=errors, alpha=0.7)
                plt.xlabel(f"{metric.capitalize()} (mean ± std)")
                plt.ylabel("Experiment")
                plt.title(f"Ablation Study Results - {metric.capitalize()}")
                plt.grid(axis='x', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                # Save plot
                plt.savefig(viz_dir / f"{metric}_comparison.png", dpi=100)
                plt.close()
                
                # Create detailed plots with all seeds
                plt.figure(figsize=(12, 6))
                
                positions = np.arange(len(labels))
                for i, exp_name in enumerate(reversed(labels)):
                    exp_results = results[exp_name]
                    values_key = f"{metric}_values"
                    if values_key in exp_results:
                        # Plot individual seed values
                        values = exp_results[values_key]
                        plt.scatter([len(labels) - 1 - i] * len(values), values, 
                                   alpha=0.5, marker='o')
                
                # Plot means with error bars
                plt.barh(positions, list(reversed(means)), xerr=list(reversed(errors)), 
                        alpha=0.3, color='gray')
                
                plt.yticks(positions, list(reversed(labels)))
                plt.xlabel(f"{metric.capitalize()}")
                plt.ylabel("Experiment")
                plt.title(f"Ablation Study Results - {metric.capitalize()} (All Seeds)")
                plt.grid(axis='x', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                # Save plot
                plt.savefig(viz_dir / f"{metric}_detailed.png", dpi=100)
                plt.close()
            
            # Create component importance plot
            if len(metrics) > 0:
                # Use first metric for component importance
                metric = metrics[0]
                mean_key = f"{metric}_mean"
                
                # Get baseline experiment (usually named 'baseline')
                baseline_exp = None
                for name in ['baseline', 'full']:
                    if name in results:
                        baseline_exp = name
                        break
                
                if baseline_exp and mean_key in results[baseline_exp]:
                    baseline_value = results[baseline_exp][mean_key]
                    
                    # Calculate component importance
                    importance = {}
                    
                    for exp_name, exp_results in results.items():
                        # Skip baseline
                        if exp_name == baseline_exp:
                            continue
                        
                        # Get metric value
                        if mean_key in exp_results:
                            # Get the component being ablated
                            mods = exp_results.get('modifications', {})
                            if len(mods) == 1:
                                # Single component experiment
                                component = list(mods.keys())[0]
                                value = exp_results[mean_key]
                                
                                # Calculate importance (decrease from baseline)
                                importance[component] = baseline_value - value
                    
                    if importance:
                        # Create component importance plot
                        plt.figure(figsize=(10, 6))
                        
                        # Sort by importance
                        components = sorted(importance.keys(), key=lambda x: abs(importance[x]), reverse=True)
                        importance_values = [importance[c] for c in components]
                        
                        # Create bar chart
                        bars = plt.bar(components, importance_values, alpha=0.7)
                        
                        # Color bars by sign (positive = good, negative = bad)
                        for i, v in enumerate(importance_values):
                            if v < 0:
                                bars[i].set_color('red')
                            else:
                                bars[i].set_color('green')
                        
                        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                        plt.xlabel("Component")
                        plt.ylabel(f"Importance (Δ{metric})")
                        plt.title(f"Component Importance - Effect on {metric.capitalize()}")
                        plt.xticks(rotation=45, ha='right')
                        plt.grid(axis='y', linestyle='--', alpha=0.7)
                        plt.tight_layout()
                        
                        # Save plot
                        plt.savefig(viz_dir / "component_importance.png", dpi=100)
                        plt.close()
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            import traceback
            traceback.print_exc()