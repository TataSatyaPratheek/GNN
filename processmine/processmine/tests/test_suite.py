import unittest
import pandas as pd
import numpy as np
import torch
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import random
import sys
import io
from contextlib import redirect_stdout

# Add coverage reporting
try:
    import coverage
    USE_COVERAGE = True
except ImportError:
    USE_COVERAGE = False

# Path for storing test results
TEST_RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'test_results')
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

# Test data generation utilities
def generate_process_data(n_cases=20, n_activities=5, n_resources=3, max_events_per_case=10):
    """Generate synthetic process log data for testing."""
    data = []
    
    for case_id in range(1, n_cases + 1):
        # Determine number of events for this case
        n_events = random.randint(3, max_events_per_case)
        
        # Generate timestamps with increasing times
        start_time = pd.Timestamp('2023-01-01') + pd.Timedelta(hours=case_id)
        timestamps = [start_time + pd.Timedelta(minutes=30*i) for i in range(n_events)]
        
        # Generate tasks and resources
        tasks = [random.randint(1, n_activities) for _ in range(n_events)]
        task_names = [f"Activity_{t}" for t in tasks]
        resources = [random.randint(1, n_resources) for _ in range(n_events)]
        resource_names = [f"Resource_{r}" for r in resources]
        
        # Add events to data
        for i in range(n_events):
            event = {
                'case_id': case_id,
                'task_id': tasks[i],
                'task_name': task_names[i],
                'resource_id': resources[i],
                'resource': resource_names[i],
                'timestamp': timestamps[i],
                'next_task': tasks[i+1] if i < n_events-1 else None
            }
            data.append(event)
    
    # Create dataframe
    df = pd.DataFrame(data)
    return df

def generate_test_csv(path, n_cases=20):
    """Generate a test CSV file with process data."""
    df = generate_process_data(n_cases=n_cases)
    df.to_csv(path, index=False)
    return df

class TestDataLoaderComprehensive(unittest.TestCase):
    """Comprehensive tests for data loading and preprocessing."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data that can be reused across tests."""
        # Create a temporary directory for test data
        cls.test_dir = tempfile.mkdtemp()
        
        # Generate a test CSV with more complex data
        cls.test_csv = os.path.join(cls.test_dir, "test_log.csv")
        cls.test_df = generate_test_csv(cls.test_csv, n_cases=20)
        
        # Import necessary modules
        try:
            from processmine.data.loader import load_and_preprocess_data
            from processmine.data.graphs import build_graph_data
            
            # Preprocess data once for reuse
            cls.df, cls.task_encoder, cls.resource_encoder = load_and_preprocess_data(
                cls.test_csv, norm_method='l2', use_dtypes=True
            )
            
            # Build graphs once for reuse
            cls.graphs = build_graph_data(cls.df, enhanced=True, batch_size=5)
            
            cls.imports_successful = True
        except ImportError:
            cls.imports_successful = False
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test resources."""
        shutil.rmtree(cls.test_dir)
    
    def setUp(self):
        """Skip tests if imports failed."""
        if not self.imports_successful:
            self.skipTest("ProcessMine package not available in test environment")
    
    def test_load_data_with_chunking(self):
        """Test data loading with chunking enabled."""
        from processmine.data.loader import load_and_preprocess_data
        
        # Test with small chunk size
        df, task_encoder, resource_encoder = load_and_preprocess_data(
            self.test_csv, 
            norm_method='l2', 
            chunk_size=5,  # Small chunk size to force chunking
            use_dtypes=True
        )
        
        # Verify data loaded correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue("next_task" in df.columns)
        self.assertTrue("feat_task_id" in df.columns)
    
    def test_load_data_with_different_normalization(self):
        """Test data loading with different normalization methods."""
        from processmine.data.loader import load_and_preprocess_data
        
        # Test with standard normalization
        df_std, _, _ = load_and_preprocess_data(
            self.test_csv, 
            norm_method='standard',
            use_dtypes=True
        )
        
        # Test with minmax normalization
        df_minmax, _, _ = load_and_preprocess_data(
            self.test_csv, 
            norm_method='minmax',
            use_dtypes=True
        )
        
        # Test with no normalization
        df_none, _, _ = load_and_preprocess_data(
            self.test_csv, 
            norm_method=None,
            use_dtypes=True
        )
        
        # Verify feature differences between normalization methods
        if 'feat_task_id' in df_std.columns and 'feat_task_id' in df_minmax.columns:
            # Standard should have mean around 0
            self.assertAlmostEqual(df_std['feat_task_id'].mean(), 0, delta=0.5)
            
            # Minmax should be between 0 and 1
            self.assertTrue((df_minmax['feat_task_id'] >= 0).all())
            self.assertTrue((df_minmax['feat_task_id'] <= 1).all())
            
            # No normalization should just use original values
            if 'task_id' in df_none.columns and 'feat_task_id' in df_none.columns:
                self.assertEqual(df_none['task_id'].equals(df_none['feat_task_id']), True)
    
    def test_build_graph_data_options(self):
        """Test graph building with different options."""
        from processmine.data.graphs import build_graph_data
        
        # Test with basic options
        graphs_basic = build_graph_data(
            self.df, 
            enhanced=False, 
            batch_size=5
        )
        
        # Test with bidirectional=False (new)
        graphs_directed = build_graph_data(
            self.df, 
            enhanced=True, 
            bidirectional=False,
            batch_size=5
        )
        
        # Test with limit_nodes parameter (new)
        graphs_limited = build_graph_data(
            self.df,
            enhanced=True,
            batch_size=5,
            limit_nodes=5  # Limit nodes per graph
        )
        
        # Test with mode parameter (new)
        graphs_sparse = build_graph_data(
            self.df,
            enhanced=True,
            batch_size=5,
            mode='sparse'  # Use sparse implementation
        )
        
        # Verify differences
        self.assertTrue(len(graphs_basic) > 0)
        self.assertTrue(len(graphs_directed) > 0)
        self.assertTrue(len(graphs_limited) > 0)
        self.assertTrue(len(graphs_sparse) > 0)
        
        # Limited graphs should have maximum 5 nodes per graph
        if len(graphs_limited) > 0:
            max_nodes = max(g.x.shape[0] for g in graphs_limited)
            self.assertLessEqual(max_nodes, 5)
            
    def test_build_heterogeneous_graph(self):
        """Test building heterogeneous graphs."""
        from processmine.data.graphs import build_heterogeneous_graph
        
        # Test building heterogeneous graphs
        het_graphs = build_heterogeneous_graph(
            self.df,
            batch_size=5,
            use_edge_attr=True  # Test with edge attributes
        )
        
        # Verify structure
        self.assertTrue(len(het_graphs) > 0)
        
        # Check structure of first graph
        first_graph = het_graphs[0]
        self.assertIn('node_features', first_graph)
        self.assertIn('edge_indices', first_graph)
    
    def test_build_heterogeneous_graph(self):
        """Test building heterogeneous graphs."""
        from processmine.data.graphs import build_heterogeneous_graph
        
        # Test building heterogeneous graphs
        het_graphs = build_heterogeneous_graph(
            self.df,
            batch_size=5
        )
        
        # Verify structure
        self.assertTrue(len(het_graphs) > 0)
        
        # Check structure of first graph
        first_graph = het_graphs[0]
        self.assertIn('node_features', first_graph)
        self.assertIn('edge_indices', first_graph)
        if 'task' in first_graph['node_features']:
            self.assertTrue(isinstance(first_graph['node_features']['task'], torch.Tensor))

    def test_memory_optimization(self):
        """Test memory optimization during data loading."""
        # Capture memory usage output
        from processmine.data.loader import load_and_preprocess_data
        from processmine.utils.memory import get_memory_stats, clear_memory
        
        # Get initial memory stats
        initial_stats = get_memory_stats()
        
        # Load data with memory optimization
        with redirect_stdout(io.StringIO()) as f:
            df, _, _ = load_and_preprocess_data(
                self.test_csv, 
                norm_method='l2',
                memory_limit_gb=0.1,  # Very small limit to force chunking
                use_dtypes=True
            )
        
        # Clear memory
        clear_memory(full_clear=True)
        
        # Get final memory stats
        final_stats = get_memory_stats()
        
        # Verify memory usage is reasonable
        if 'cpu_used_gb' in initial_stats and 'cpu_used_gb' in final_stats:
            # Memory increase should be reasonable (less than 1GB for test data)
            self.assertLess(final_stats['cpu_used_gb'] - initial_stats['cpu_used_gb'], 1.0)
    
    def test_normalize_features(self):
        """Test the new normalize_features function."""
        from processmine.data.features import normalize_features
        
        # Create test feature array
        features = np.random.rand(10, 5)
        
        # Test L2 normalization
        l2_features = normalize_features(features, method='l2')
        
        # Test standard normalization
        std_features = normalize_features(features, method='standard')
        
        # Test minmax normalization
        minmax_features = normalize_features(features, method='minmax')
        
        # Test return as tensor
        tensor_features = normalize_features(features, method='l2', return_tensor=True)
        
        # Verify shapes
        self.assertEqual(l2_features.shape, features.shape)
        self.assertEqual(std_features.shape, features.shape)
        self.assertEqual(minmax_features.shape, features.shape)
        self.assertEqual(tensor_features.shape, features.shape)
        
        # Verify L2 normalization
        l2_norms = np.sqrt(np.sum(l2_features**2, axis=1))
        self.assertTrue(np.allclose(l2_norms, np.ones(10), atol=1e-6))
        
        # Verify minmax normalization in [0, 1]
        self.assertTrue((minmax_features >= 0).all())
        self.assertTrue((minmax_features <= 1).all())
        
        # Verify tensor return type
        self.assertIsInstance(tensor_features, torch.Tensor)


class TestProcessMiningAnalysis(unittest.TestCase):
    """Test the process mining analysis functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data that can be reused across tests."""
        # Import modules
        try:
            from processmine.process_mining.analysis import (
                analyze_bottlenecks,
                analyze_cycle_times,
                analyze_transition_patterns,
                identify_process_variants,
                analyze_resource_workload,
                analyze_resource_performance,
                perform_conformance_checking,
                run_complete_analysis
            )
            from processmine.data.loader import load_and_preprocess_data
            
            cls.imports_successful = True
            
            # Create test dataframe with realistic process mining data
            cls.test_df = generate_process_data(n_cases=50, n_activities=10, n_resources=5, max_events_per_case=15)
            
            # Ensure timestamp is datetime
            cls.test_df['timestamp'] = pd.to_datetime(cls.test_df['timestamp'])
            
            # Remove None/NaN values from next_task
            cls.test_df = cls.test_df.dropna(subset=['next_task'])
            cls.test_df['next_task'] = cls.test_df['next_task'].astype(int)
            
            # Add next_timestamp column for bottleneck analysis
            cls.test_df['next_timestamp'] = cls.test_df.groupby('case_id')['timestamp'].shift(-1)
            
        except ImportError:
            cls.imports_successful = False
    
    def setUp(self):
        """Skip tests if imports failed."""
        if not self.imports_successful:
            self.skipTest("ProcessMine package not available in test environment")
    
    def test_analyze_bottlenecks(self):
        """Test bottleneck analysis with different parameters."""
        from processmine.process_mining.analysis import analyze_bottlenecks
        
        # Test with different thresholds
        for percentile in [50, 75, 90]:
            bottleneck_stats, significant_bottlenecks = analyze_bottlenecks(
                self.test_df, 
                freq_threshold=2, 
                percentile_threshold=float(percentile),
                vectorized=True  # Test new vectorized parameter
            )
            
            # Verify output structure
            self.assertIsInstance(bottleneck_stats, pd.DataFrame)
            self.assertIsInstance(significant_bottlenecks, pd.DataFrame)
            
            # Check that the required columns exist
            for col in ['task_id', 'next_task_id', 'count', 'mean', 'mean_hours']:
                self.assertIn(col, bottleneck_stats.columns)
                
            # Check that higher percentile yields fewer significant bottlenecks
            if percentile < 90:
                prev_bottlenecks = significant_bottlenecks
            else:
                self.assertLessEqual(len(significant_bottlenecks), len(prev_bottlenecks))
        
        # Test with memory_efficient parameter
        bottleneck_stats, significant_bottlenecks = analyze_bottlenecks(
            self.test_df, 
            freq_threshold=2,
            percentile_threshold=90.0,
            memory_efficient=True  # Test new memory_efficient parameter
        )
        
        # Verify output
        self.assertIsInstance(bottleneck_stats, pd.DataFrame)
        self.assertIsInstance(significant_bottlenecks, pd.DataFrame)
    
    def test_analyze_cycle_times(self):
        """Test cycle time analysis."""
        from processmine.process_mining.analysis import analyze_cycle_times
        
        # Test with default parameters
        case_stats, long_cases, p95 = analyze_cycle_times(self.test_df)
        
        # Verify output structure
        self.assertIsInstance(case_stats, pd.DataFrame)
        self.assertIsInstance(long_cases, pd.DataFrame)
        self.assertIsInstance(p95, float)
        
        # Check columns in case_stats
        for col in ['duration_h', 'duration_days']:
            self.assertIn(col, case_stats.columns)
        
        # Verify that p95 is a valid percentile
        self.assertGreaterEqual(p95, 0)
        self.assertTrue(case_stats['duration_h'].quantile(0.95) <= p95 * 1.01)  # Allow for small numerical differences
        
        # Test with memory_efficient parameter
        case_stats, long_cases, p95 = analyze_cycle_times(
            self.test_df,
            memory_efficient=True  # Test new memory_efficient parameter
        )
        
        # Verify output
        self.assertIsInstance(case_stats, pd.DataFrame)
        self.assertIsInstance(long_cases, pd.DataFrame)
    
    def test_analyze_transition_patterns(self):
        """Test transition pattern analysis."""
        from processmine.process_mining.analysis import analyze_transition_patterns
        
        # Test transition analysis
        transitions, trans_count, prob_matrix = analyze_transition_patterns(self.test_df)
        
        # Verify output structure
        self.assertIsInstance(transitions, pd.DataFrame)
        self.assertIsInstance(trans_count, pd.DataFrame)
        self.assertIsInstance(prob_matrix, pd.DataFrame)
        
        # Check that probability matrix sums to 1 for each row
        for idx, row in prob_matrix.iterrows():
            row_sum = row.sum()
            if not pd.isna(row_sum) and row_sum > 0:
                self.assertAlmostEqual(row_sum, 1.0, places=6)
        
        # Test with memory_efficient parameter
        transitions, trans_count, prob_matrix = analyze_transition_patterns(
            self.test_df,
            memory_efficient=True  # Test new memory_efficient parameter
        )
        
        # Verify output
        self.assertIsInstance(transitions, pd.DataFrame)
        self.assertIsInstance(trans_count, pd.DataFrame)
        self.assertIsInstance(prob_matrix, pd.DataFrame)
    
    def test_identify_process_variants(self):
        """Test process variant identification."""
        from processmine.process_mining.analysis import identify_process_variants
        
        # Test with different max_variants
        for max_var in [5, 10]:
            variant_stats, variant_sequences = identify_process_variants(
                self.test_df,
                max_variants=max_var
            )
            
            # Verify output structure
            self.assertIsInstance(variant_stats, pd.DataFrame)
            self.assertIsInstance(variant_sequences, dict)
            
            # Check that the number of variants is correct
            self.assertLessEqual(len(variant_sequences), max_var)
            
            # Check that percentages sum to 100%
            if 'percentage' in variant_stats.columns:
                total_pct = variant_stats['percentage'].sum()
                self.assertAlmostEqual(total_pct, 100.0, places=1)
        
        # Test with memory_efficient parameter
        variant_stats, variant_sequences = identify_process_variants(
            self.test_df,
            max_variants=5,
            memory_efficient=True  # Test new memory_efficient parameter
        )
        
        # Verify output
        self.assertIsInstance(variant_stats, pd.DataFrame)
        self.assertIsInstance(variant_sequences, dict)
    
    def test_analyze_resource_workload(self):
        """Test resource workload analysis."""
        from processmine.process_mining.analysis import analyze_resource_workload
        
        # Test resource analysis
        resource_stats = analyze_resource_workload(self.test_df)
        
        # Verify output structure
        self.assertIsInstance(resource_stats, pd.DataFrame)
        
        # Check columns
        for col in ['activity_count', 'case_count', 'workload_percentage']:
            self.assertIn(col, resource_stats.columns)
        
        # Check that workload percentages sum to 100%
        if 'workload_percentage' in resource_stats.columns:
            total_pct = resource_stats['workload_percentage'].sum()
            self.assertAlmostEqual(total_pct, 100.0, places=1)
        
        # Test with memory_efficient parameter
        resource_stats = analyze_resource_workload(
            self.test_df,
            memory_efficient=True  # Test new memory_efficient parameter
        )
        
        # Verify output
        self.assertIsInstance(resource_stats, pd.DataFrame)
    
        def test_analyze_resource_performance(self):
            """Test resource performance analysis (new function)."""
            from processmine.process_mining.analysis import analyze_resource_performance
            
            # Test resource performance analysis
            resource_perf = analyze_resource_performance(self.test_df)
            
            # Verify output structure
            self.assertIsInstance(resource_perf, pd.DataFrame)
            
            # Check columns in resource_perf
            for col in ['task_duration_mean', 'task_duration_median', 'throughput']:
                self.assertIn(col, resource_perf.columns)
        
        def test_perform_conformance_checking(self):
            """Test conformance checking."""
            from processmine.process_mining.analysis import perform_conformance_checking
            
            # Test conformance checking
            results = perform_conformance_checking(self.test_df)
            
            # Verify output structure
            self.assertIsInstance(results, dict)
            self.assertIn('conformance_ratio', results)
            self.assertIn('variant_coverage', results)
    
    def test_run_complete_analysis(self):
        """Test the complete analysis pipeline."""
        from processmine.process_mining.analysis import run_complete_analysis
        
        # Test complete analysis
        results = run_complete_analysis(self.test_df)
        
        # Verify output structure
        self.assertIsInstance(results, dict)
        
        # Check for key components
        expected_keys = [
            'bottleneck_stats', 'significant_bottlenecks', 
            'case_stats', 'transitions', 'variant_stats', 
            'resource_stats', 'metrics'
        ]
        
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check metrics structure
        self.assertIn('cases', results['metrics'])
        self.assertIn('events', results['metrics'])
        self.assertIn('activities', results['metrics'])
        self.assertIn('resources', results['metrics'])
        self.assertIn('perf', results['metrics'])


class TestModelArchitectures(unittest.TestCase):
    """Test the model architectures for process mining."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data that can be reused across tests."""
        # Import modules
        try:
            import torch
            from processmine.models.gnn.architectures import (
                MemoryEfficientGNN, 
                PositionalGATConv, 
                DiverseGATConv, 
                CombinedGATConv
            )
            from processmine.models.sequence.lstm import NextActivityLSTM, EnhancedProcessRNN
            from processmine.processmine import create_model
            
            cls.imports_successful = True
            
            # Set up test tensors
            cls.node_features = torch.randn(20, 8)  # 20 nodes, 8 features
            cls.edge_index = torch.tensor([
                [0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9],  # Source nodes
                [1, 0, 2, 1, 3, 2, 5, 4, 7, 6, 9, 8]   # Target nodes
            ], dtype=torch.long)
            
            # Create mock PyG data
            try:
                from torch_geometric.data import Data
                from torch_geometric.loader import DataLoader
                
                # Create a list of graphs
                cls.graphs = []
                for i in range(5):
                    # Create a graph with random nodes and edges
                    n_nodes = random.randint(5, 10)
                    x = torch.randn(n_nodes, 8)  # Node features
                    
                    # Create edge indices
                    n_edges = n_nodes * 2
                    edge_src = torch.randint(0, n_nodes, (n_edges,))
                    edge_dst = torch.randint(0, n_nodes, (n_edges,))
                    edge_index = torch.stack([edge_src, edge_dst])
                    
                    # Create edge attributes
                    edge_attr = torch.randn(n_edges, 1)
                    
                    # Create graph target
                    y = torch.tensor([i % 3], dtype=torch.long)
                    
                    # Create batch index
                    batch = torch.zeros(n_nodes, dtype=torch.long)
                    
                    # Create Data object
                    data = Data(
                        x=x, 
                        edge_index=edge_index, 
                        edge_attr=edge_attr, 
                        y=y,
                        batch=batch
                    )
                    cls.graphs.append(data)
                
                # Create dataloader
                cls.loader = DataLoader(cls.graphs, batch_size=2)
                cls.pyg_available = True
                
            except ImportError:
                cls.pyg_available = False
                
        except ImportError:
            cls.imports_successful = False
    
    def setUp(self):
        """Skip tests if imports failed."""
        if not self.imports_successful:
            self.skipTest("ProcessMine package not available in test environment")
    
    def test_gnn_model_creation(self):
        """Test GNN model creation with different configurations."""
        from processmine.models.gnn.architectures import MemoryEfficientGNN
        
        # Test creating models with different configurations
        models = []
        
        # Test basic attention
        models.append(MemoryEfficientGNN(
            input_dim=8,
            hidden_dim=16,
            output_dim=3,
            num_layers=2,
            heads=2,
            dropout=0.1,
            attention_type="basic"
        ))
        
        # Test positional attention
        models.append(MemoryEfficientGNN(
            input_dim=8,
            hidden_dim=16,
            output_dim=3,
            num_layers=2,
            heads=2,
            dropout=0.1,
            attention_type="positional",
            pos_enc_dim=4
        ))
        
        # Test diverse attention
        models.append(MemoryEfficientGNN(
            input_dim=8,
            hidden_dim=16,
            output_dim=3,
            num_layers=2,
            heads=2,
            dropout=0.1,
            attention_type="diverse",
            diversity_weight=0.1
        ))
        
        # Test combined attention
        models.append(MemoryEfficientGNN(
            input_dim=8,
            hidden_dim=16,
            output_dim=3,
            num_layers=2,
            heads=2,
            dropout=0.1,
            attention_type="combined",
            pos_enc_dim=4,
            diversity_weight=0.1
        ))
        
        # Test new parameters: sparse_attention
        models.append(MemoryEfficientGNN(
            input_dim=8,
            hidden_dim=16,
            output_dim=3,
            num_layers=2,
            heads=2,
            dropout=0.1,
            attention_type="basic",
            sparse_attention=True  # New parameter
        ))
        
        # Test new parameters: use_checkpointing
        models.append(MemoryEfficientGNN(
            input_dim=8,
            hidden_dim=16,
            output_dim=3,
            num_layers=2,
            heads=2,
            dropout=0.1,
            attention_type="basic",
            use_checkpointing=True  # New parameter
        ))
        
        # Test new parameters: pooling
        for pooling in ["mean", "max", "sum", "combined", "attention"]:
            models.append(MemoryEfficientGNN(
                input_dim=8,
                hidden_dim=16,
                output_dim=3,
                num_layers=2,
                heads=2,
                dropout=0.1,
                attention_type="basic",
                pooling=pooling  # New parameter with different options
            ))
        
        # Verify models were created successfully
        for model in models:
            self.assertIsInstance(model, MemoryEfficientGNN)
            
            # Test memory usage calculation
            mem_usage = model.memory_usage()
            self.assertIsInstance(mem_usage, dict)
            self.assertIn('parameters_mb', mem_usage)
    
    def test_gnn_forward_pass(self):
        """Test GNN forward pass."""
        if not self.pyg_available:
            self.skipTest("PyTorch Geometric not available")
        
        from processmine.models.gnn.architectures import MemoryEfficientGNN
        
        # Create a model
        model = MemoryEfficientGNN(
            input_dim=8,
            hidden_dim=16,
            output_dim=3,
            num_layers=2,
            heads=2,
            dropout=0.1,
            attention_type="basic"
        )
        
        # Test forward pass on a batch
        for batch in self.loader:
            output = model(batch)
            
            # Verify output shape and structure
            self.assertIsInstance(output, dict)
            self.assertIn('task_pred', output)
            self.assertEqual(output['task_pred'].shape[1], 3)  # 3 output classes
            break  # Only test the first batch
        
        # Test get_embeddings method
        for batch in self.loader:
            embeddings, batch_indices = model.get_embeddings(batch)
            
            # Verify embeddings shape
            self.assertEqual(embeddings.dim(), 2)
            self.assertEqual(embeddings.shape[1], 16 * 2)  # hidden_dim * heads
            break
        
        # Test get_attention_weights method
        for batch in self.loader:
            attn_weights = model.get_attention_weights(batch)
            
            # Verify weights shape
            self.assertIsInstance(attn_weights, list)
            # AttentionGNN should have attention weights for each layer
            self.assertEqual(len(attn_weights), 2)  # 2 layers
            break
    
    def test_lstm_model_creation(self):
        """Test LSTM model creation."""
        from processmine.models.sequence.lstm import NextActivityLSTM, EnhancedProcessRNN
        
        # Test creating basic LSTM
        basic_lstm = NextActivityLSTM(
            num_cls=3,
            emb_dim=16,
            hidden_dim=32,
            num_layers=1,
            dropout=0.1
        )
        
        # Test new parameters for basic LSTM
        basic_lstm_with_options = NextActivityLSTM(
            num_cls=3,
            emb_dim=16,
            hidden_dim=32,
            num_layers=1,
            dropout=0.1,
            bidirectional=True,  # Test bidirectional
            use_attention=True,   # Test attention
            use_layer_norm=True,  # Test layer norm
            mem_efficient=True    # Test memory efficiency
        )
        
        # Test creating enhanced RNN
        enhanced_rnn = EnhancedProcessRNN(
            num_cls=3,
            emb_dim=16,
            hidden_dim=32,
            num_layers=2,
            dropout=0.1,
            use_gru=False,
            use_transformer=True,
            num_heads=4
        )
        
        # Test new parameters for enhanced RNN
        enhanced_rnn_with_options = EnhancedProcessRNN(
            num_cls=3,
            emb_dim=16,
            hidden_dim=32,
            num_layers=2,
            dropout=0.1,
            use_gru=True,              # Test GRU option
            use_transformer=True,
            num_heads=4,
            use_time_features=True,    # Test time features
            time_encoding_dim=8,       # Test time encoding dimension
            mem_efficient=True         # Test memory efficiency
        )
        
        # Verify models were created successfully
        self.assertIsInstance(basic_lstm, NextActivityLSTM)
        self.assertIsInstance(basic_lstm_with_options, NextActivityLSTM)
        self.assertIsInstance(enhanced_rnn, EnhancedProcessRNN)
        self.assertIsInstance(enhanced_rnn_with_options, EnhancedProcessRNN)
    
    def test_lstm_forward_pass(self):
        """Test LSTM forward pass."""
        from processmine.models.sequence.lstm import NextActivityLSTM
        import torch
        
        # Create a model
        model = NextActivityLSTM(
            num_cls=3,
            emb_dim=16,
            hidden_dim=32,
            num_layers=1,
            dropout=0.1
        )
        
        # Create a batch of sequences
        seq_batch = torch.randint(0, 3, (5, 10))  # 5 sequences, length 10
        seq_lengths = torch.tensor([10, 8, 6, 9, 7])  # Variable lengths
        
        # Test forward pass
        output = model(seq_batch, seq_lengths)
        
        # Verify output shape
        self.assertIsInstance(output, dict)
        self.assertIn('task_pred', output)
        self.assertEqual(output['task_pred'].shape, (5, 3))  # 5 sequences, 3 classes
        
        # Test processing graph data
        if self.pyg_available:
            # Create a model that can handle graph data
            model = NextActivityLSTM(
                num_cls=3,
                emb_dim=16,
                hidden_dim=32,
                num_layers=1,
                dropout=0.1
            )
            
            # Test _process_graph_data method
            for batch in self.loader:
                output = model(batch)
                
                # Verify output
                self.assertIsInstance(output, dict)
                self.assertIn('task_pred', output)
                break
    
    def test_model_factory(self):
        """Test model factory function."""
        from processmine.processmine import create_model
        
        # Test creating different model types
        models = []
        
        # Test GNN
        models.append(create_model(
            model_type='gnn',
            input_dim=8,
            hidden_dim=16,
            output_dim=3
        ))
        
        # Test enhanced GNN
        models.append(create_model(
            model_type='enhanced_gnn',
            input_dim=8,
            hidden_dim=16,
            output_dim=3
        ))
        
        # Test LSTM
        models.append(create_model(
            model_type='lstm',
            num_cls=3,
            emb_dim=16,
            hidden_dim=32
        ))
        
        # Test enhanced LSTM
        models.append(create_model(
            model_type='enhanced_lstm',
            num_cls=3,
            emb_dim=16,
            hidden_dim=32
        ))
        
        # Test random forest
        try:
            models.append(create_model(
                model_type='random_forest',
                n_estimators=10
            ))
        except ImportError:
            pass  # Skip if sklearn not available
        
        # Test XGBoost
        try:
            models.append(create_model(
                model_type='xgboost',
                n_estimators=10,
                max_depth=3
            ))
        except ImportError:
            pass  # Skip if xgboost not available
        
        # Verify models were created successfully
        expected_types = [
            'MemoryEfficientGNN', 'MemoryEfficientGNN', 
            'NextActivityLSTM', 'EnhancedProcessRNN',
            'RandomForestClassifier', 'XGBClassifier'
        ]
        
        for i, model in enumerate(models):
            # Check model type matches expected
            if i < len(expected_types):
                model_type = model.__class__.__name__
                self.assertTrue(
                    model_type == expected_types[i] or 
                    expected_types[i] in str(type(model)),
                    f"Expected {expected_types[i]}, got {model_type}"
                )


class TestVisualizationComprehensive(unittest.TestCase):
    """Comprehensive tests for visualization utilities."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data that can be reused across tests."""
        # Create a temporary directory for visualizations
        cls.test_dir = tempfile.mkdtemp()
        
        # Import modules
        try:
            from processmine.visualization.viz import ProcessVisualizer
            
            cls.imports_successful = True
            
            # Create test data
            cls.cycle_times = np.random.exponential(scale=10, size=200)
            
            # Create bottleneck data
            cls.bottleneck_stats = pd.DataFrame({
                'task_id': [1, 1, 2, 2, 3, 3, 4, 4],
                'next_task_id': [2, 3, 3, 4, 4, 5, 5, 1],
                'count': [20, 15, 12, 10, 8, 6, 5, 3],
                'mean': [100, 200, 150, 50, 80, 120, 90, 30],
                'median': [90, 180, 130, 40, 70, 110, 80, 25],
                'std': [20, 40, 30, 10, 15, 25, 20, 5],
                'min': [50, 100, 80, 30, 40, 60, 50, 20],
                'max': [150, 300, 220, 70, 120, 180, 130, 40],
                'mean_hours': [0.03, 0.06, 0.04, 0.01, 0.02, 0.03, 0.025, 0.008]
            })
            
            # Create significant bottlenecks
            cls.significant_bottlenecks = cls.bottleneck_stats[cls.bottleneck_stats['count'] > 8].copy()
            cls.significant_bottlenecks['bottleneck_score'] = [2.0, 3.0, 1.5, 1.2]
            
            # Create mock task encoder
            cls.task_encoder = MagicMock()
            cls.task_encoder.inverse_transform = lambda x: [f"Activity_{i}" for i in x]
            
            # Create mock process flow transitions
            cls.transitions = pd.DataFrame({
                'task_id': [1, 1, 2, 2, 3, 3, 4],
                'next_task_id': [2, 3, 3, 4, 4, 5, 5],
                'count': [20, 15, 12, 10, 8, 6, 5]
            })
            
            # Create visualizer
            cls.visualizer = ProcessVisualizer(output_dir=cls.test_dir)
            
        except ImportError:
            cls.imports_successful = False
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test resources."""
        shutil.rmtree(cls.test_dir)
    
    def setUp(self):
        """Skip tests if imports failed."""
        if not self.imports_successful:
            self.skipTest("ProcessMine package not available in test environment")
    
    def test_cycle_time_distribution_options(self):
        """Test cycle time distribution with different options."""
        # Test with default settings
        self.visualizer.cycle_time_distribution(
            self.cycle_times, 
            filename="cycle_time_default.png"
        )
        
        # Test with custom bins
        self.visualizer.cycle_time_distribution(
            self.cycle_times, 
            filename="cycle_time_custom_bins.png",
            bins=20
        )
        
        # Test without KDE
        self.visualizer.cycle_time_distribution(
            self.cycle_times, 
            filename="cycle_time_no_kde.png",
            include_kde=False
        )
        
        # Test with show_percentiles parameter
        self.visualizer.cycle_time_distribution(
            self.cycle_times, 
            filename="cycle_time_no_percentiles.png",
            show_percentiles=False  # New parameter
        )
        
        # Verify files were created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "cycle_time_default.png")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "cycle_time_custom_bins.png")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "cycle_time_no_kde.png")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "cycle_time_no_percentiles.png")))
    
    def test_bottleneck_analysis_options(self):
        """Test bottleneck analysis with different options."""
        # Test with default settings
        self.visualizer.bottleneck_analysis(
            self.bottleneck_stats,
            self.significant_bottlenecks,
            self.task_encoder,
            filename="bottleneck_default.png"
        )
        
        # Test with limited top_n
        self.visualizer.bottleneck_analysis(
            self.bottleneck_stats,
            self.significant_bottlenecks,
            self.task_encoder,
            filename="bottleneck_top3.png",
            top_n=3
        )
        
        # Verify files were created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "bottleneck_default.png")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "bottleneck_top3.png")))
    
    def test_process_flow_options(self):
        """Test process flow visualization with different options."""
        # Test with default settings
        self.visualizer.process_flow(
            self.bottleneck_stats,
            self.task_encoder,
            self.significant_bottlenecks,
            filename="process_flow_default.png"
        )
        
        # Test with different layout
        self.visualizer.process_flow(
            self.bottleneck_stats,
            self.task_encoder,
            self.significant_bottlenecks,
            filename="process_flow_circular.png",
            layout='circular'
        )
        
        # Test with limited nodes
        self.visualizer.process_flow(
            self.bottleneck_stats,
            self.task_encoder,
            self.significant_bottlenecks,
            filename="process_flow_limited.png",
            max_nodes=4
        )
        
        # Verify files were created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "process_flow_default.png")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "process_flow_circular.png")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "process_flow_limited.png")))
    
    def test_transition_heatmap(self):
        """Test transition heatmap visualization."""
        # First create transition matrix
        trans_count = pd.crosstab(
            self.transitions["task_id"], 
            self.transitions["next_task_id"],
            normalize=False
        )
        
        # Calculate probability matrix
        row_sums = trans_count.sum(axis=1)
        prob_matrix = trans_count.div(row_sums, axis=0).fillna(0)
        
        # Test visualization
        self.visualizer.transition_heatmap(
            prob_matrix,
            self.task_encoder,
            filename="transition_heatmap.png"
        )
        
        # Test with max_activities parameter
        self.visualizer.transition_heatmap(
            prob_matrix,
            self.task_encoder,
            filename="transition_heatmap_limited.png",
            max_activities=3  # Limit to 3 activities
        )
        
        # Verify files were created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "transition_heatmap.png")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "transition_heatmap_limited.png")))
    
    def test_dashboard_creation(self):
        """Test dashboard creation."""
        # Skip if not interactive mode
        if self.visualizer.force_static:
            self.skipTest("Visualizer is in static mode, dashboard requires interactive mode")
        
        # Generate case data
        case_stats = pd.DataFrame({
            'case_id': range(1, 101),
            'duration_h': self.cycle_times[:100],
            'task_id_count': np.random.randint(5, 15, 100),
            'task_id_nunique': np.random.randint(3, 8, 100),
            'resource_id_nunique': np.random.randint(1, 4, 100)
        })
        
        # Create dashboard
        self.visualizer.create_dashboard(
            df=generate_process_data(n_cases=30),
            cycle_times=self.cycle_times,
            bottleneck_stats=self.bottleneck_stats,
            significant_bottlenecks=self.significant_bottlenecks,
            task_encoder=self.task_encoder,
            case_stats=case_stats,
            filename="dashboard.html"
        )
        
        # Verify file was created
        dashboard_file = os.path.join(self.test_dir, "dashboard.html")
        if os.path.exists(dashboard_file):
            # Check that the file is not empty
            self.assertGreater(os.path.getsize(dashboard_file), 1000)
    
    def test_visualizer_init_options(self):
        """Test visualizer initialization with different options."""
        # Test with different styles
        from processmine.visualization.viz import ProcessVisualizer
        
        for style in ['default', 'dark', 'light']:
            visualizer = ProcessVisualizer(
                output_dir=self.test_dir,
                style=style
            )
            
            # Create a simple visualization to verify it works
            visualizer.cycle_time_distribution(
                self.cycle_times, 
                filename=f"cycle_time_{style}.png"
            )
            
            # Verify file was created
            self.assertTrue(os.path.exists(os.path.join(self.test_dir, f"cycle_time_{style}.png")))
        
        # Test with memory_efficient parameter
        memory_efficient_viz = ProcessVisualizer(
            output_dir=self.test_dir,
            memory_efficient=True,
            sampling_threshold=50,  # Low threshold to force sampling
            max_plot_points=25       # Low max points to test sampling
        )
        
        # Create visualization with large dataset
        memory_efficient_viz.cycle_time_distribution(
            np.random.exponential(scale=10, size=100),  # Size > sampling_threshold
            filename="cycle_time_memory_efficient.png"
        )
        
        # Verify file was created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "cycle_time_memory_efficient.png")))
    
        def test_dashboard_creation(self):
            """Test dashboard creation."""
            # Skip if not interactive mode
            if self.visualizer.force_static:
                self.skipTest("Visualizer is in static mode, dashboard requires interactive mode")
            
            # Generate case data
            case_stats = pd.DataFrame({
                'case_id': range(1, 101),
                'duration_h': self.cycle_times[:100],
                'task_id_count': np.random.randint(5, 15, 100),
                'task_id_nunique': np.random.randint(3, 8, 100),
                'resource_id_nunique': np.random.randint(1, 4, 100)
            })
            
            # Create dashboard
            self.visualizer.create_dashboard(
                df=generate_process_data(n_cases=30),
                cycle_times=self.cycle_times,
                bottleneck_stats=self.bottleneck_stats,
                significant_bottlenecks=self.significant_bottlenecks,
                task_encoder=self.task_encoder,
                case_stats=case_stats,
                filename="dashboard.html"
            )
            
            # Verify file was created
            dashboard_file = os.path.join(self.test_dir, "dashboard.html")
            if os.path.exists(dashboard_file):
                # Check that the file is not empty
                self.assertGreater(os.path.getsize(dashboard_file), 1000)


class TestCLIFunctionality(unittest.TestCase):
    """Test command-line interface functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data that can be reused across tests."""
        # Create a temporary directory for outputs
        cls.test_dir = tempfile.mkdtemp()
        
        # Create a test CSV file
        cls.test_csv = os.path.join(cls.test_dir, "test_log.csv")
        cls.test_df = generate_test_csv(cls.test_csv, n_cases=10)
        
        # Import modules
        try:
            import processmine.processmine.cli as cli
            cls.imports_successful = True
        except ImportError:
            cls.imports_successful = False
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test resources."""
        shutil.rmtree(cls.test_dir)
    
    def setUp(self):
        """Skip tests if imports failed."""
        if not self.imports_successful:
            self.skipTest("ProcessMine CLI not available in test environment")
    
    @patch('sys.argv')
    def test_cli_parse_arguments(self, mock_argv):
        """Test command-line argument parsing."""
        import processmine.processmine.cli as cli
        
        # Test analyze mode
        mock_argv.__getitem__.side_effect = lambda idx: [
            'processmine',
            self.test_csv,
            'analyze',
            '--output-dir', self.test_dir,
            '--viz-format', 'static',
            '--seed', '42'
        ][idx]
        mock_argv.__len__.return_value = 8
        
        args = cli.parse_arguments()
        
        # Verify parsed arguments
        self.assertEqual(args.data_path, self.test_csv)
        self.assertEqual(args.mode, 'analyze')
        self.assertEqual(args.output_dir, self.test_dir)
        self.assertEqual(args.viz_format, 'static')
        self.assertEqual(args.seed, 42)
    
    @patch('sys.argv')
    def test_cli_parse_model_specific_args(self, mock_argv):
        """Test parsing model-specific arguments (new feature)."""
        import processmine.processmine.cli as cli
        
        # Test train mode with model-specific args
        mock_argv.__getitem__.side_effect = lambda idx: [
            'processmine',
            self.test_csv,
            'train',
            '--model', 'enhanced_gnn',
            '--hidden_dim', '32',
            '--heads', '4',
            '--attention_type', 'combined'
        ][idx]
        mock_argv.__len__.return_value = 10
        
        args = cli.parse_arguments()
        
        # Verify model-specific args were parsed
        self.assertEqual(args.model, 'enhanced_gnn')
        self.assertEqual(args.hidden_dim, 32)
        self.assertEqual(args.heads, 4)
        self.assertEqual(args.attention_type, 'combined')
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_cli_setup_environment(self, mock_cuda):
        """Test environment setup."""
        import processmine.processmine.cli as cli
        
        # Create mock args
        args = MagicMock()
        args.seed = 42
        args.device = None
        args.output_dir = self.test_dir
        args.debug = True
        
        # Test function
        device, output_dir = cli.setup_environment(args)
        
        # Verify setup
        self.assertEqual(device.type, 'cpu')
        self.assertTrue(str(output_dir).startswith(self.test_dir))
    
    @patch('processmine.core.runner.run_analysis')
    def test_cli_main_function(self, mock_run_analysis):
        """Test main function with different modes."""
        import processmine.processmine.cli as cli
        
        # Mock run functions
        mock_run_analysis.return_value = {"result": "success"}
        
        # Create mock args
        args = MagicMock()
        args.data_path = self.test_csv
        args.mode = 'analyze'
        args.output_dir = self.test_dir
        args.debug = False
        
        # Patch parse_arguments
        with patch('processmine.processmine.cli.parse_arguments', return_value=args):
            # Test main function
            exit_code = cli.main()
            
            # Verify run_analysis was called
            mock_run_analysis.assert_called_once()
            self.assertEqual(exit_code, 0)


def run_tests_with_coverage():
    """Run tests with coverage reporting."""
    if USE_COVERAGE:
        cov = coverage.Coverage(source=["processmine"])
        cov.start()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoaderComprehensive))
    suite.addTests(loader.loadTestsFromTestCase(TestProcessMiningAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestModelArchitectures))
    suite.addTests(loader.loadTestsFromTestCase(TestVisualizationComprehensive))
    suite.addTests(loader.loadTestsFromTestCase(TestCLIFunctionality))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate coverage report if available
    if USE_COVERAGE:
        cov.stop()
        cov.save()
        
        # Print report
        print("\nCoverage Report:")
        cov.report()
        
        # Generate HTML report
        html_dir = os.path.join(TEST_RESULTS_DIR, 'coverage_html')
        os.makedirs(html_dir, exist_ok=True)
        cov.html_report(directory=html_dir)
        print(f"HTML coverage report saved to {html_dir}")
    
    return result


if __name__ == '__main__':
    result = run_tests_with_coverage()
    sys.exit(not result.wasSuccessful())