import unittest
import time
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
import torch
import logging
import gc
import psutil
import sys
from contextlib import redirect_stdout
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Duration thresholds for performance tests
SHORT_DURATION_THRESHOLD = 5.0  # seconds
MEDIUM_DURATION_THRESHOLD = 15.0  # seconds
LONG_DURATION_THRESHOLD = 30.0  # seconds

# Helper function for generating large test datasets
def generate_large_dataset(path, n_cases, events_per_case, n_activities=20, n_resources=10):
    """Generate a large dataset for performance testing."""
    # Create output directory if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Track total number of events
    total_events = n_cases * events_per_case
    logger.info(f"Generating dataset with {n_cases:,} cases, {total_events:,} events...")
    
    # Generate in chunks to avoid memory issues
    chunk_size = 100000
    all_chunks = []
    
    # Generate case IDs
    case_ids = np.repeat(np.arange(1, n_cases + 1), events_per_case)
    
    # Generate random task and resource IDs
    task_ids = np.random.randint(1, n_activities + 1, total_events)
    task_names = [f"Activity_{i}" for i in task_ids]
    resource_ids = np.random.randint(1, n_resources + 1, total_events)
    resource_names = [f"Resource_{i}" for i in resource_ids]
    
    # Generate timestamps
    timestamps = []
    start_date = pd.Timestamp('2023-01-01')
    for case_id in range(1, n_cases + 1):
        case_start = start_date + pd.Timedelta(hours=case_id % 8760)  # Spread over a year
        for i in range(events_per_case):
            timestamps.append(case_start + pd.Timedelta(minutes=30 * i))
    
    # Create DataFrame
    df = pd.DataFrame({
        'case_id': case_ids,
        'task_id': task_ids,
        'task_name': task_names,
        'resource_id': resource_ids,
        'resource': resource_names,
        'timestamp': timestamps
    })
    
    # Add next_task column
    next_task = np.zeros_like(task_ids)
    for i in range(n_cases):
        start_idx = i * events_per_case
        end_idx = start_idx + events_per_case
        next_task[start_idx:end_idx-1] = task_ids[start_idx+1:end_idx]
        next_task[end_idx-1] = -1  # End of case
    
    df['next_task'] = next_task
    df = df[df['next_task'] >= 0]  # Remove last events
    
    # Write to CSV
    df.to_csv(path, index=False)
    logger.info(f"Dataset generated and written to {path}")
    
    return df

class MemoryTracker:
    """Utility for tracking memory usage."""
    
    def __init__(self):
        """Initialize tracker."""
        self.start_ram = psutil.virtual_memory().used / (1024 * 1024 * 1024)  # GB
        if torch.cuda.is_available():
            self.start_gpu = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # GB
        else:
            self.start_gpu = 0
        
        self.peak_ram = self.start_ram
        self.peak_gpu = self.start_gpu
        self.snapshots = []
    
    def snapshot(self, label):
        """Take a memory snapshot with a label."""
        ram = psutil.virtual_memory().used / (1024 * 1024 * 1024)  # GB
        if torch.cuda.is_available():
            gpu = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # GB
        else:
            gpu = 0
        
        self.peak_ram = max(self.peak_ram, ram)
        self.peak_gpu = max(self.peak_gpu, gpu)
        
        snapshot = {
            'label': label,
            'ram_gb': ram,
            'ram_delta_gb': ram - self.start_ram,
            'gpu_gb': gpu,
            'gpu_delta_gb': gpu - self.start_gpu
        }
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def summary(self):
        """Get a summary of memory usage."""
        return {
            'start_ram_gb': self.start_ram,
            'peak_ram_gb': self.peak_ram,
            'ram_increase_gb': self.peak_ram - self.start_ram,
            'start_gpu_gb': self.start_gpu,
            'peak_gpu_gb': self.peak_gpu,
            'gpu_increase_gb': self.peak_gpu - self.start_gpu,
            'snapshots': self.snapshots
        }
    
    def print_summary(self):
        """Print a summary of memory usage."""
        summary = self.summary()
        
        print("\nMEMORY USAGE SUMMARY:")
        print(f"  Initial RAM: {summary['start_ram_gb']:.2f} GB")
        print(f"  Peak RAM: {summary['peak_ram_gb']:.2f} GB (increase: {summary['ram_increase_gb']:.2f} GB)")
        
        if torch.cuda.is_available():
            print(f"  Initial GPU: {summary['start_gpu_gb']:.2f} GB")
            print(f"  Peak GPU: {summary['peak_gpu_gb']:.2f} GB (increase: {summary['gpu_increase_gb']:.2f} GB)")
        
        print("\nMEMORY SNAPSHOTS:")
        for snap in summary['snapshots']:
            print(f"  {snap['label']}:")
            print(f"    RAM: {snap['ram_gb']:.2f} GB (delta: {snap['ram_delta_gb']:.2f} GB)")
            if torch.cuda.is_available():
                print(f"    GPU: {snap['gpu_gb']:.2f} GB (delta: {snap['gpu_delta_gb']:.2f} GB)")


class ProcessMinePerformanceTests(unittest.TestCase):
    """Performance tests for ProcessMine package."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        # Create a temporary directory for test data
        cls.test_dir = tempfile.mkdtemp()
        
        # Create small dataset for quick tests
        cls.small_data_path = os.path.join(cls.test_dir, 'small_data.csv')
        cls.small_df = generate_large_dataset(
            cls.small_data_path, 
            n_cases=100, 
            events_per_case=10
        )
        
        # Create medium dataset for more intensive tests
        cls.medium_data_path = os.path.join(cls.test_dir, 'medium_data.csv')
        cls.medium_df = generate_large_dataset(
            cls.medium_data_path, 
            n_cases=1000, 
            events_per_case=10
        )
        
        # Large dataset will be generated on demand
        cls.large_data_path = os.path.join(cls.test_dir, 'large_data.csv')
        
        # Check if package is available
        try:
            import processmine
            cls.is_available = True
        except ImportError:
            cls.is_available = False
            logger.warning("ProcessMine package not available, tests will be skipped")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test resources."""
        shutil.rmtree(cls.test_dir)
    
    def setUp(self):
        """Set up for each test."""
        if not self.is_available:
            self.skipTest("ProcessMine package not available")
            
        # Clean up before each test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def test_data_loading_performance(self):
        """Test data loading performance."""
        from processmine.data.loader import load_and_preprocess_data
        from processmine.utils.memory import clear_memory
        
        # Initialize memory tracker
        tracker = MemoryTracker()
        
        # Test with small dataset
        start_time = time.time()
        df_small, task_encoder, resource_encoder = load_and_preprocess_data(
            self.small_data_path,
            norm_method='l2',
            use_dtypes=True
        )
        small_duration = time.time() - start_time
        tracker.snapshot(f"Small dataset ({len(df_small):,} events)")
        
        # Should be quick for small data
        self.assertLess(small_duration, SHORT_DURATION_THRESHOLD)
        
        # Clean up
        del df_small
        clear_memory(full_clear=True)
        
        # Test with medium dataset
        start_time = time.time()
        df_medium, task_encoder, resource_encoder = load_and_preprocess_data(
            self.medium_data_path,
            norm_method='l2',
            use_dtypes=True
        )
        medium_duration = time.time() - start_time
        tracker.snapshot(f"Medium dataset ({len(df_medium):,} events)")
        
        # Medium dataset should be reasonably fast
        self.assertLess(medium_duration, MEDIUM_DURATION_THRESHOLD)
        
        # Report times
        print(f"\nData loading performance:")
        print(f"  Small dataset ({len(df_small):,} events): {small_duration:.2f}s")
        print(f"  Medium dataset ({len(df_medium):,} events): {medium_duration:.2f}s")
        
        # Print memory usage
        tracker.print_summary()
    
    def test_graph_building_performance(self):
        """Test graph building performance."""
        from processmine.data.loader import load_and_preprocess_data
        from processmine.data.graphs import build_graph_data
        from processmine.utils.memory import clear_memory
        
        # Initialize memory tracker
        tracker = MemoryTracker()
        
        # Initialize memory tracker
        tracker = MemoryTracker()
    
        # Test with small dataset and memory limit
        start_time = time.time()
        df_small, task_encoder, resource_encoder = load_and_preprocess_data(
            self.small_data_path,
            norm_method='l2',
            use_dtypes=True,
            memory_limit_gb=0.1  # Test small memory limit
        )
        small_duration = time.time() - start_time
        tracker.snapshot(f"Small dataset with memory limit ({len(df_small):,} events)")
        
        # Clean up
        del df_small
        clear_memory(full_clear=True)
        
        # Test with different chunk sizes
        start_time = time.time()
        df_chunked, _, _ = load_and_preprocess_data(
            self.medium_data_path, 
            norm_method='l2', 
            chunk_size=1000,  # Force small chunks
            use_dtypes=True
        )
        chunked_duration = time.time() - start_time
        tracker.snapshot(f"Medium dataset with small chunks ({len(df_chunked):,} events)")
        
        # Report times
        print(f"\nData loading performance:")
        print(f"  Small dataset with memory limit: {small_duration:.2f}s")
        print(f"  Medium dataset with chunking: {chunked_duration:.2f}s")
        
        # Print memory usage
        tracker.print_summary()
        
         # Test with auto mode (new)
        start_time = time.time()
        graphs_auto = build_graph_data(
            df_chunked,
            enhanced=True,
            batch_size=100,
            mode='auto'  # Auto mode
        )
        auto_duration = time.time() - start_time
        tracker.snapshot("After building graphs (auto mode)")
        
        # Test with sparse mode (new)
        start_time = time.time()
        graphs_sparse = build_graph_data(
            df_chunked,
            enhanced=True,
            batch_size=100,
            mode='sparse'  # Sparse mode
        )
        sparse_duration = time.time() - start_time
        tracker.snapshot("After building graphs (sparse mode)")
        
        # Report times
        print(f"\nGraph building performance:")
        print(f"  Auto mode: {auto_duration:.2f}s")
        print(f"  Sparse mode: {sparse_duration:.2f}s")
    
    def test_analysis_performance(self):
        """Test analysis performance."""
        from processmine.data.loader import load_and_preprocess_data
        from processmine.process_mining.analysis import (
            analyze_bottlenecks,
            analyze_cycle_times,
            analyze_transition_patterns,
            identify_process_variants
        )
        from processmine.utils.memory import clear_memory
        
        # Initialize memory tracker
        tracker = MemoryTracker()
        
        # Load medium dataset
        df_medium, task_encoder, resource_encoder = load_and_preprocess_data(
            self.medium_data_path,
            norm_method='l2',
            use_dtypes=True
        )
        tracker.snapshot("After loading medium dataset")
        
        # Test bottleneck analysis
        start_time = time.time()
        bottleneck_stats, significant_bottlenecks = analyze_bottlenecks(
            df_medium,
            freq_threshold=5,
            percentile_threshold=90.0
        )
        bottleneck_duration = time.time() - start_time
        tracker.snapshot("After bottleneck analysis")
        
        # Test cycle time analysis
        start_time = time.time()
        case_stats, long_cases, p95 = analyze_cycle_times(df_medium)
        cycle_time_duration = time.time() - start_time
        tracker.snapshot("After cycle time analysis")
        
        # Test transition pattern analysis
        start_time = time.time()
        transitions, trans_count, prob_matrix = analyze_transition_patterns(df_medium)
        transition_duration = time.time() - start_time
        tracker.snapshot("After transition analysis")
        
        # Test process variant analysis
        start_time = time.time()
        variant_stats, variant_sequences = identify_process_variants(
            df_medium,
            max_variants=10
        )
        variant_duration = time.time() - start_time
        tracker.snapshot("After variant analysis")
        
        # Time expectations
        self.assertLess(bottleneck_duration, SHORT_DURATION_THRESHOLD)
        self.assertLess(cycle_time_duration, SHORT_DURATION_THRESHOLD)
        self.assertLess(transition_duration, SHORT_DURATION_THRESHOLD)
        self.assertLess(variant_duration, SHORT_DURATION_THRESHOLD)
        
        # Report times
        print(f"\nAnalysis performance:")
        print(f"  Bottleneck analysis: {bottleneck_duration:.2f}s")
        print(f"  Cycle time analysis: {cycle_time_duration:.2f}s")
        print(f"  Transition analysis: {transition_duration:.2f}s")
        print(f"  Variant analysis: {variant_duration:.2f}s")
        
        # Print memory usage
        tracker.print_summary()
    
    def test_model_performance(self):
        """Test model performance."""
        from processmine.data.loader import load_and_preprocess_data
        from processmine.data.graphs import build_graph_data
        from processmine.models.factory import create_model
        from processmine.utils.memory import clear_memory
        
        # Check if DGL is available
        try:
            from dgl.dataloading import GraphDataLoader
            dgl_available = True
        except ImportError:
            dgl_available = False
            self.skipTest("DGL not available")
        
        # Initialize memory tracker
        tracker = MemoryTracker()
        
        # Load small dataset
        df_small, task_encoder, resource_encoder = load_and_preprocess_data(
            self.small_data_path,
            norm_method='l2',
            use_dtypes=True
        )
        
        # Build graphs
        graphs = build_graph_data(
            df_small,
            enhanced=True,
            batch_size=50
        )
        tracker.snapshot("After building graphs")
        
        # Create data loaders
        import torch
        from dgl.dataloading import GraphDataLoader
        
        indices = np.arange(len(graphs))
        np.random.shuffle(indices)
        
        train_size = int(0.7 * len(indices))
        val_size = int(0.15 * len(indices))
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        # Create data loaders
        train_loader = GraphDataLoader([graphs[i] for i in train_idx], batch_size=16, shuffle=True)
        val_loader = GraphDataLoader([graphs[i] for i in val_idx], batch_size=16)
        tracker.snapshot("After creating data loaders")
        
        # Create different model types
        model_types = ['gnn', 'enhanced_gnn']
        model_durations = {}
        
        for model_type in model_types:
            clear_memory(full_clear=True)
            
            # Create model
            start_time = time.time()
            model = create_model(
                model_type=model_type,
                input_dim=len([col for col in df_small.columns if col.startswith("feat_")]),
                hidden_dim=32,
                output_dim=len(task_encoder.classes_)
            )
            creation_time = time.time() - start_time
            tracker.snapshot(f"After creating {model_type} model")
            
            # Test forward pass
            device = torch.device('cpu')  # Use CPU for consistency
            model = model.to(device)
            
            start_time = time.time()
            # Forward pass on a few batches
            model.train()
            forward_count = 0
            
            for batch in train_loader:
                batch = batch.to(device)
                _ = model(batch)
                forward_count += 1
                if forward_count >= 5:  # Limit to 5 batches for performance test
                    break
            
            forward_time = time.time() - start_time
            tracker.snapshot(f"After {model_type} forward passes")
            
            model_durations[model_type] = {
                'creation': creation_time,
                'forward_per_batch': forward_time / max(1, forward_count)
            }
            
            # Clean up
            del model
            clear_memory(full_clear=True)
        
        # Report times
        print(f"\nModel performance:")
        for model_type, durations in model_durations.items():
            print(f"  {model_type}:")
            print(f"    Creation time: {durations['creation']:.2f}s")
            print(f"    Forward pass (per batch): {durations['forward_per_batch']:.4f}s")
        
        # Print memory usage
        tracker.print_summary()
    
    def test_training_performance(self):
        """Test model training performance."""
        from processmine.data.loader import load_and_preprocess_data
        from processmine.data.graphs import build_graph_data
        from processmine.models.factory import create_model
        from processmine.utils.memory import clear_memory
        from processmine.core.training import train_model, compute_class_weights
        
        # Check if DGL is available
        try:
            from dgl.dataloading import GraphDataLoader
            dgl_available = True
        except ImportError:
            dgl_available = False
            self.skipTest("DGL not available")
        
        # Initialize memory tracker
        tracker = MemoryTracker()
        
        # Load small dataset
        df_small, task_encoder, resource_encoder = load_and_preprocess_data(
            self.small_data_path,
            norm_method='l2',
            use_dtypes=True
        )
        
        # Build graphs
        graphs = build_graph_data(
            df_small,
            enhanced=True,
            batch_size=50
        )
        tracker.snapshot("After building graphs")
        
        # Create data loaders
        import torch
        from dgl.dataloading import GraphDataLoader
        
        indices = np.arange(len(graphs))
        np.random.shuffle(indices)
        
        train_size = int(0.7 * len(indices))
        val_size = int(0.15 * len(indices))
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        
        # Create data loaders
        train_loader = GraphDataLoader([graphs[i] for i in train_idx], batch_size=16, shuffle=True)
        val_loader = GraphDataLoader([graphs[i] for i in val_idx], batch_size=16)
        
        # Compare memory-efficient vs regular mode
        model_configs = [
            {'name': 'Standard', 'memory_efficient': False},
            {'name': 'Memory-efficient', 'memory_efficient': True}
        ]
        
        training_stats = {}
        
        for config in model_configs:
            clear_memory(full_clear=True)
            tracker.snapshot(f"Before {config['name']} training")
            
            # Create model
            model = create_model(
                model_type='enhanced_gnn',
                input_dim=len([col for col in df_small.columns if col.startswith("feat_")]),
                hidden_dim=32,
                output_dim=len(task_encoder.classes_)
            )
            
            # Compute class weights
            class_weights = compute_class_weights(df_small, len(task_encoder.classes_))
            
            # Configure optimizer and loss
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            
            # Train model (with minimal epochs for testing)
            device = torch.device('cpu')  # Use CPU for consistency
            
            # Capture stdout to suppress training output
            with io.StringIO() as buf, redirect_stdout(buf):
                start_time = time.time()
                model, metrics = train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    epochs=3,  # Minimal for testing
                    patience=5,
                    memory_efficient=config['memory_efficient']
                )
                training_time = time.time() - start_time
            
            tracker.snapshot(f"After {config['name']} training")
            
            training_stats[config['name']] = {
                'training_time': training_time,
                'time_per_epoch': training_time / 3  # 3 epochs
            }
            
            # Clean up
            del model, optimizer, criterion
            clear_memory(full_clear=True)
        
        # Report times
        print(f"\nTraining performance (3 epochs):")
        for name, stats in training_stats.items():
            print(f"  {name} mode:")
            print(f"    Total training time: {stats['training_time']:.2f}s")
            print(f"    Time per epoch: {stats['time_per_epoch']:.2f}s")
        
        # Print memory usage
        tracker.print_summary()
    
    def test_visualization_performance(self):
        """Test visualization performance."""
        from processmine.data.loader import load_and_preprocess_data
        from processmine.process_mining.analysis import (
            analyze_bottlenecks,
            analyze_cycle_times,
            analyze_transition_patterns,
            identify_process_variants
        )
        from processmine.visualization.viz import ProcessVisualizer
        from processmine.utils.memory import clear_memory
        
        # Initialize memory tracker
        tracker = MemoryTracker()
        
        # Load medium dataset
        df_medium, task_encoder, resource_encoder = load_and_preprocess_data(
            self.medium_data_path,
            norm_method='l2',
            use_dtypes=True
        )
        
        # Run analysis to get visualization data
        bottleneck_stats, significant_bottlenecks = analyze_bottlenecks(
            df_medium,
            freq_threshold=5,
            percentile_threshold=90.0
        )
        
        case_stats, long_cases, p95 = analyze_cycle_times(df_medium)
        
        transitions, trans_count, prob_matrix = analyze_transition_patterns(df_medium)
        
        tracker.snapshot("After analysis")
        
        # Create visualizer
        viz_dir = os.path.join(self.test_dir, 'viz_performance')
        os.makedirs(viz_dir, exist_ok=True)
        
        visualizer = ProcessVisualizer(output_dir=viz_dir)
        
        # Test different visualization functions
        viz_times = {}
        
        # Cycle time distribution
        start_time = time.time()
        visualizer.cycle_time_distribution(
            case_stats['duration_h'].values,
            filename='cycle_time_distribution.png'
        )
        cycle_time_viz_time = time.time() - start_time
        viz_times['cycle_time'] = cycle_time_viz_time
        tracker.snapshot("After cycle time visualization")
        
        # Bottleneck analysis
        start_time = time.time()
        visualizer.bottleneck_analysis(
            bottleneck_stats,
            significant_bottlenecks,
            task_encoder,
            filename='bottleneck_analysis.png'
        )
        bottleneck_viz_time = time.time() - start_time
        viz_times['bottleneck'] = bottleneck_viz_time
        tracker.snapshot("After bottleneck visualization")
        
        # Process flow
        start_time = time.time()
        visualizer.process_flow(
            bottleneck_stats,
            task_encoder,
            significant_bottlenecks,
            filename='process_flow.png'
        )
        process_flow_viz_time = time.time() - start_time
        viz_times['process_flow'] = process_flow_viz_time
        tracker.snapshot("After process flow visualization")
        
        # Transition heatmap
        start_time = time.time()
        visualizer.transition_heatmap(
            transitions,
            task_encoder,
            filename='transition_heatmap.png'
        )
        heatmap_viz_time = time.time() - start_time
        viz_times['heatmap'] = heatmap_viz_time
        tracker.snapshot("After heatmap visualization")
        
        # Time expectations - visualization can be slow, especially for large graphs
        self.assertLess(cycle_time_viz_time, SHORT_DURATION_THRESHOLD)
        self.assertLess(bottleneck_viz_time, SHORT_DURATION_THRESHOLD)
        self.assertLess(heatmap_viz_time, MEDIUM_DURATION_THRESHOLD)
        # Process flow can take longer due to layout algorithms
        self.assertLess(process_flow_viz_time, LONG_DURATION_THRESHOLD)
        
        # Report times
        print(f"\nVisualization performance:")
        print(f"  Cycle time distribution: {cycle_time_viz_time:.2f}s")
        print(f"  Bottleneck analysis: {bottleneck_viz_time:.2f}s")
        print(f"  Process flow: {process_flow_viz_time:.2f}s")
        print(f"  Transition heatmap: {heatmap_viz_time:.2f}s")
        
        # Print memory usage
        tracker.print_summary()
    
    @unittest.skip("Skip large-scale test by default")
    def test_large_scale_performance(self):
        """Test performance with a large dataset."""
        # Only run this test if explicitly enabled (it's resource intensive)
        if not os.environ.get('RUN_LARGE_SCALE_TEST'):
            self.skipTest("Large-scale test not enabled")
        
        from processmine.data.loader import load_and_preprocess_data
        from processmine.utils.memory import clear_memory, get_memory_stats
        
        # Generate large dataset if needed
        if not os.path.exists(self.large_data_path):
            self.large_df = generate_large_dataset(
                self.large_data_path,
                n_cases=10000,  # 10,000 cases
                events_per_case=15  # 15 events per case
            )
        
        # Initialize memory tracker
        tracker = MemoryTracker()
        
        # Load with memory optimization
        start_time = time.time()
        df_large, task_encoder, resource_encoder = load_and_preprocess_data(
            self.large_data_path,
            norm_method='l2',
            chunk_size=100000,  # Process in chunks
            use_dtypes=True,
            memory_limit_gb=2.0  # Limit memory usage
        )
        loading_time = time.time() - start_time
        tracker.snapshot(f"After loading large dataset ({len(df_large):,} events)")
        
        # Run bottleneck analysis
        from processmine.process_mining.analysis import analyze_bottlenecks
        
        start_time = time.time()
        bottleneck_stats, significant_bottlenecks = analyze_bottlenecks(
            df_large,
            freq_threshold=10,
            percentile_threshold=90.0,
            vectorized=True  # Use vectorized implementation
        )
        bottleneck_time = time.time() - start_time
        tracker.snapshot("After bottleneck analysis")
        
        # Clean up
        del df_large, bottleneck_stats, significant_bottlenecks
        clear_memory(full_clear=True)
        tracker.snapshot("After cleanup")
        
        # Report times
        print(f"\nLarge-scale performance:")
        print(f"  Loading time: {loading_time:.2f}s")
        print(f"  Bottleneck analysis time: {bottleneck_time:.2f}s")
        
        # Print memory usage
        tracker.print_summary()


if __name__ == '__main__':
    unittest.main()