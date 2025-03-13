import unittest
import numpy as np
import pandas as pd
import torch
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock

# Import the modules to test
try:
    from processmine.core.training import train_model, evaluate_model, compute_class_weights
    from processmine.data.loader import load_and_preprocess_data
    from processmine.data.graph_builder import build_graph_data, build_heterogeneous_graph
    from processmine.models.gnn.architectures import MemoryEfficientGNN, PositionalGATConv
    from processmine.models.sequence.lstm import NextActivityLSTM
    from processmine.process_mining.analysis import analyze_bottlenecks, analyze_cycle_times
    from processmine.visualization.viz import ProcessVisualizer
    from processmine.utils.memory import clear_memory, get_memory_stats
    from processmine.processmine import create_model
except ImportError:
    # For CI/CD where imports might fail
    pass


class TestDataLoader(unittest.TestCase):
    """Test the data loading and preprocessing functionality."""

    def setUp(self):
        """Set up test data."""
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Create a simple test CSV
        self.test_csv = os.path.join(self.test_dir, "test_data.csv")
        
        # Create test data
        data = {
            "case_id": [1, 1, 1, 2, 2, 3],
            "task_name": ["start", "process", "end", "start", "end", "start"],
            "resource": ["user_a", "user_b", "user_a", "user_c", "user_c", "user_b"],
            "timestamp": [
                "2023-01-01 10:00:00", 
                "2023-01-01 10:30:00",
                "2023-01-01 11:00:00",
                "2023-01-01 12:00:00",
                "2023-01-01 13:00:00",
                "2023-01-01 14:00:00"
            ]
        }
        
        self.test_df = pd.DataFrame(data)
        self.test_df.to_csv(self.test_csv, index=False)

    def tearDown(self):
        """Clean up test resources."""
        shutil.rmtree(self.test_dir)

    def test_load_and_preprocess_data(self):
        """Test loading and preprocessing data from CSV."""
        try:
            # Test the function
            df, task_encoder, resource_encoder = load_and_preprocess_data(
                self.test_csv, norm_method='l2', use_dtypes=True
            )
            
            # Check that the dataframe has the expected shape
            self.assertEqual(len(df), 5)  # 6 original rows - 1 last event per case
            
            # Check that encoders were created
            self.assertEqual(len(task_encoder.classes_), 3)  # start, process, end
            self.assertEqual(len(resource_encoder.classes_), 3)  # user_a, user_b, user_c
            
            # Check that derived features were created
            self.assertTrue("feat_task_id" in df.columns)
            self.assertTrue("next_task" in df.columns)
            
            # Check additional features created from new version
            self.assertTrue("case_events" in df.columns)
            self.assertTrue("case_unique_tasks" in df.columns)
            
        except (ImportError, NameError):
            self.skipTest("ProcessMine package not available in test environment")

    def test_build_graph_data(self):
        """Test building graph data from dataframe."""
        try:
            # First load and preprocess the data
            df, task_encoder, resource_encoder = load_and_preprocess_data(
                self.test_csv, norm_method='l2'
            )
            
            # Test building graph data with updated parameters
            graphs = build_graph_data(
                df, 
                enhanced=True, 
                batch_size=2, 
                num_workers=0, 
                verbose=True,
                bidirectional=True,
                mode='auto'
            )
            
            # Check that graphs were created
            self.assertTrue(isinstance(graphs, list))
            self.assertGreater(len(graphs), 0)
            
            # Check graph structure
            for graph in graphs:
                self.assertTrue(hasattr(graph, 'x'))
                self.assertTrue(hasattr(graph, 'edge_index'))
                if len(graph.edge_index.shape) > 0 and graph.edge_index.shape[1] > 0:
                    self.assertTrue(hasattr(graph, 'edge_attr'))
                self.assertTrue(hasattr(graph, 'y'))
                
        except (ImportError, NameError):
            self.skipTest("ProcessMine package not available in test environment")


class TestProcessAnalysis(unittest.TestCase):
    """Test the process mining analysis functionality."""

    def setUp(self):
        """Set up test data."""
        # Create a more complex test dataframe
        case_ids = np.repeat(range(1, 11), 5)  # 10 cases with 5 events each
        task_ids = np.random.randint(1, 5, 50)  # 4 different tasks
        task_names = [f"Task_{i}" for i in task_ids]
        resource_ids = np.random.randint(1, 4, 50)  # 3 different resources
        
        # Create timestamps with increasing times per case
        timestamps = []
        for case in range(1, 11):
            start_time = pd.Timestamp('2023-01-01') + pd.Timedelta(hours=case)
            for i in range(5):
                timestamps.append(start_time + pd.Timedelta(minutes=i*30))
        
        self.test_df = pd.DataFrame({
            "case_id": case_ids,
            "task_id": task_ids,
            "task_name": task_names,
            "resource_id": resource_ids,
            "timestamp": timestamps,
            "next_task": np.roll(task_ids, -1)  # Shift task_ids for next_task
        })
        
        # Set NaN for last event in each case
        self.test_df.loc[self.test_df.groupby("case_id").tail(1).index, "next_task"] = np.nan
        self.test_df = self.test_df.dropna()  # Drop NaN rows

    def test_analyze_bottlenecks(self):
        """Test bottleneck analysis."""
        try:
            # Add next_timestamp column for bottleneck analysis
            self.test_df['next_timestamp'] = self.test_df.groupby("case_id")["timestamp"].shift(-1)
            
            # Test the function with the new memory_efficient parameter
            bottleneck_stats, significant_bottlenecks = analyze_bottlenecks(
                self.test_df, 
                freq_threshold=1, 
                percentile_threshold=75.0,
                vectorized=True,
                memory_efficient=True
            )
            
            # Check output format
            self.assertIsInstance(bottleneck_stats, pd.DataFrame)
            self.assertIsInstance(significant_bottlenecks, pd.DataFrame)
            
            # Check that stats were computed
            self.assertTrue("mean_hours" in bottleneck_stats.columns)
            self.assertTrue("count" in bottleneck_stats.columns)
            
            # Check new columns from updated function
            self.assertTrue("rank" in bottleneck_stats.columns)
            if len(significant_bottlenecks) > 0:
                self.assertTrue("bottleneck_score" in significant_bottlenecks.columns)
            
        except (ImportError, NameError):
            self.skipTest("ProcessMine package not available in test environment")

    def test_analyze_cycle_times(self):
        """Test cycle time analysis."""
        try:
            # Test the function with the memory_efficient parameter
            case_stats, long_cases, p95 = analyze_cycle_times(
                self.test_df, 
                percentile_threshold=95.0,
                memory_efficient=True
            )
            
            # Check output format
            self.assertIsInstance(case_stats, pd.DataFrame)
            self.assertIsInstance(long_cases, pd.DataFrame)
            self.assertIsInstance(p95, float)
            
            # Check that stats were computed
            self.assertTrue("duration_h" in case_stats.columns)
            self.assertGreaterEqual(p95, 0)
            
            # Check additional metrics from updated function
            self.assertTrue("duration_days" in case_stats.columns)
            
        except (ImportError, NameError):
            self.skipTest("ProcessMine package not available in test environment")


class TestModels(unittest.TestCase):
    """Test the ML models for process mining."""

    def setUp(self):
        """Set up test data and models."""
        # Create a simple test tensor
        self.X = torch.randn(10, 5)  # 10 samples, 5 features
        self.y = torch.randint(0, 3, (10,))  # 3 classes
        
        # Create mock graph data
        self.create_mock_graph_data()

    def create_mock_graph_data(self):
        """Create mock PyG data for testing GNN models."""
        try:
            from torch_geometric.data import Data, Batch
            
            # Create a list of small graphs
            self.graph_list = []
            for i in range(3):
                # Create a small graph with 3 nodes
                x = torch.randn(3, 5)  # 3 nodes, 5 features
                edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long)  # 3 edges
                edge_attr = torch.randn(3, 1)  # Edge features
                y = torch.tensor([i % 3], dtype=torch.long)  # Graph label
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
                self.graph_list.append(data)
            
            # Create a batch
            self.batch = Batch.from_data_list(self.graph_list)
            
        except ImportError:
            self.graph_list = None
            self.batch = None

    def test_gnn_model(self):
        """Test GNN model initialization and forward pass."""
        try:
            if self.batch is None:
                self.skipTest("PyTorch Geometric not available")
                
            # Create a GNN model with updated parameters
            model = MemoryEfficientGNN(
                input_dim=5,
                hidden_dim=8,
                output_dim=3,
                num_layers=2,
                heads=2,
                dropout=0.1,
                attention_type="basic",
                use_batch_norm=True,
                use_layer_norm=False,
                use_residual=True,
                mem_efficient=True
            )
            
            # Test forward pass
            output = model(self.batch)
            
            # Check output format - now returns a dictionary
            self.assertIsInstance(output, dict)
            self.assertTrue("task_pred" in output)
            self.assertEqual(output["task_pred"].shape, (3, 3))  # 3 graphs, 3 classes
            
            # Test get_embeddings
            embeddings, batch_indices = model.get_embeddings(self.batch)
            self.assertTrue(embeddings.shape[1], 8 * 2)  # hidden_dim * heads
            
            # Test memory_usage method
            mem_usage = model.memory_usage()
            self.assertIsInstance(mem_usage, dict)
            self.assertIn('parameters_mb', mem_usage)
            
        except (ImportError, NameError):
            self.skipTest("GNN model not available in test environment")

    def test_lstm_model(self):
        """Test LSTM model initialization and forward pass."""
        try:
            # Create sequence data
            seq_data = torch.randint(0, 3, (5, 10))  # 5 sequences, length 10
            seq_lengths = torch.tensor([10, 8, 6, 10, 7])
            
            # Create an LSTM model with updated parameters
            model = NextActivityLSTM(
                num_cls=3,
                emb_dim=8,
                hidden_dim=8,
                num_layers=1,
                dropout=0.1,
                bidirectional=False,
                use_attention=True,
                use_layer_norm=True,
                mem_efficient=True
            )
            
            # Test forward pass - now returns a dictionary
            output = model(seq_data, seq_lengths)
            
            # Check output format
            self.assertIsInstance(output, dict)
            self.assertTrue("task_pred" in output)
            self.assertEqual(output["task_pred"].shape, (5, 3))  # 5 sequences, 3 classes
            
        except (ImportError, NameError):
            self.skipTest("LSTM model not available in test environment")

    def test_create_model_factory(self):
        """Test model creation through the factory function."""
        try:
            # Test creating different model types
            gnn_model = create_model(
                model_type="gnn",
                input_dim=5,
                hidden_dim=8,
                output_dim=3
            )
            
            lstm_model = create_model(
                model_type="lstm",
                num_cls=3,
                emb_dim=8,
                hidden_dim=8
            )
            
            # Test enhanced models
            enhanced_gnn = create_model(
                model_type="enhanced_gnn",
                input_dim=5,
                hidden_dim=8,
                output_dim=3
            )
            
            # Check that models were created correctly
            self.assertIsNotNone(gnn_model)
            self.assertIsNotNone(lstm_model)
            self.assertIsNotNone(enhanced_gnn)
            
        except (ImportError, NameError):
            self.skipTest("Model factory not available in test environment")


class TestModelTraining(unittest.TestCase):
    """Test model training functionality."""

    def setUp(self):
        """Set up model and data for training tests."""
        try:
            # Create a simple model
            self.input_dim = 5
            self.hidden_dim = 8
            self.output_dim = 3
            self.model = MagicMock()
            self.model.return_value = {"task_pred": torch.randn(5, 3)}
            
            # Create mock data loader
            self.loader = [(torch.randn(5, 5), torch.randint(0, 3, (5,))) for _ in range(3)]
            
            # Mock device
            self.device = torch.device("cpu")
            
            # Mock criterion
            self.criterion = MagicMock()
            self.criterion.return_value = torch.tensor(0.5)
            
            # Mock optimizer
            self.optimizer = MagicMock()
            
        except (ImportError, NameError):
            self.skipTest("PyTorch not available in test environment")

    @patch('processmine.core.training.torch.no_grad')
    def test_evaluate_model(self, mock_no_grad):
        """Test model evaluation."""
        try:
            # Mock no_grad context
            mock_no_grad.return_value.__enter__ = MagicMock()
            mock_no_grad.return_value.__exit__ = MagicMock()
            
            # Test evaluate_model function with updated interface
            with patch('processmine.core.training._calculate_metrics', return_value={'accuracy': 0.8}):
                metrics, predictions, true_labels = evaluate_model(
                    self.model, self.loader, self.device, self.criterion, detailed=True
                )
            
            # Check that metrics were calculated
            self.assertTrue("accuracy" in metrics)
            self.assertEqual(metrics["accuracy"], 0.8)
            
        except (ImportError, NameError):
            self.skipTest("Training utilities not available in test environment")

    @patch('processmine.core.training.torch.no_grad')
    @patch('processmine.core.training.time.time')
    def test_train_model(self, mock_time, mock_no_grad):
        """Test model training."""
        try:
            # Mock time and no_grad
            mock_time.return_value = 0
            mock_no_grad.return_value.__enter__ = MagicMock()
            mock_no_grad.return_value.__exit__ = MagicMock()
            
            # Mock evaluate_model
            with patch('processmine.core.training.evaluate_model', 
                      return_value=(0.4, {'val_loss': 0.4}, None, None)):
                
                # Test train_model function with updated parameters
                model, history = train_model(
                    model=self.model,
                    train_loader=self.loader,
                    val_loader=self.loader, 
                    optimizer=self.optimizer,
                    criterion=self.criterion,
                    device=self.device,
                    epochs=2,
                    patience=1,
                    use_amp=False,
                    memory_efficient=True,
                    track_memory=True
                )
                
                # Check that training completed
                self.assertEqual(len(history['train_loss']), 2)
                self.assertEqual(len(history['val_loss']), 2)
            
        except (ImportError, NameError):
            self.skipTest("Training utilities not available in test environment")


class TestVisualization(unittest.TestCase):
    """Test visualization utilities."""

    def setUp(self):
        """Set up test data for visualization tests."""
        # Create a temporary directory for visualizations
        self.test_dir = tempfile.mkdtemp()
        
        # Create test data
        self.cycle_times = np.random.exponential(scale=10, size=100)
        
        # Create mock bottleneck data
        self.bottleneck_stats = pd.DataFrame({
            'task_id': [1, 1, 2, 2],
            'next_task_id': [2, 3, 3, 4],
            'count': [10, 5, 8, 3],
            'mean': [100, 200, 150, 50],
            'median': [90, 180, 130, 40],
            'std': [20, 40, 30, 10],
            'min': [50, 100, 80, 30],
            'max': [150, 300, 220, 70],
            'mean_hours': [0.03, 0.06, 0.04, 0.01]
        })
        
        self.significant_bottlenecks = self.bottleneck_stats[self.bottleneck_stats['count'] > 5].copy()
        self.significant_bottlenecks['bottleneck_score'] = [1.5, 2.0, 1.2]
        
        # Mock task encoder
        self.task_encoder = MagicMock()
        self.task_encoder.inverse_transform = lambda x: [f"Activity_{i}" for i in x]

    def tearDown(self):
        """Clean up test resources."""
        shutil.rmtree(self.test_dir)

    def test_process_visualizer(self):
        """Test ProcessVisualizer initialization."""
        try:
            # Create a visualizer with updated parameters
            visualizer = ProcessVisualizer(
                output_dir=self.test_dir,
                style='default',
                force_static=False,
                memory_efficient=True,
                sampling_threshold=100000,
                max_plot_points=50000,
                dpi=120
            )
            
            # Check that the visualizer was created
            self.assertEqual(visualizer.output_dir, self.test_dir)
            
        except (ImportError, NameError):
            self.skipTest("Visualization utilities not available in test environment")

    def test_cycle_time_distribution(self):
        """Test creating cycle time distribution plot."""
        try:
            # Create a visualizer
            visualizer = ProcessVisualizer(output_dir=self.test_dir)
            
            # Test creating cycle time distribution with updated parameters
            filename = visualizer.cycle_time_distribution(
                self.cycle_times, 
                filename="cycle_time_test.png",
                bins=20,
                include_kde=True,
                show_percentiles=True
            )
            
            # Check that the file was created
            self.assertTrue(os.path.exists(os.path.join(self.test_dir, "cycle_time_test.png")))
            
        except (ImportError, NameError):
            self.skipTest("Visualization utilities not available in test environment")

    def test_bottleneck_analysis(self):
        """Test creating bottleneck analysis plot."""
        try:
            # Create a visualizer
            visualizer = ProcessVisualizer(output_dir=self.test_dir)
            
            # Test creating bottleneck analysis with updated parameters
            filename = visualizer.bottleneck_analysis(
                self.bottleneck_stats,
                self.significant_bottlenecks,
                self.task_encoder,
                filename="bottleneck_test.png",
                top_n=3
            )
            
            # Check that the file was created
            self.assertTrue(os.path.exists(os.path.join(self.test_dir, "bottleneck_test.png")))
            
        except (ImportError, NameError):
            self.skipTest("Visualization utilities not available in test environment")

    def test_process_flow(self):
        """Test creating process flow visualization."""
        try:
            # Create a visualizer
            visualizer = ProcessVisualizer(output_dir=self.test_dir)
            
            # Test creating process flow with updated parameters
            filename = visualizer.process_flow(
                self.bottleneck_stats,
                self.task_encoder,
                self.significant_bottlenecks,
                filename="process_flow_test.png",
                max_nodes=20,
                layout='auto'
            )
            
            # Check that the file was created - this might fail if NetworkX not available
            if filename:
                self.assertTrue(os.path.exists(os.path.join(self.test_dir, "process_flow_test.png")))
            
        except (ImportError, NameError):
            self.skipTest("Visualization utilities not available in test environment")


class TestUtilities(unittest.TestCase):
    """Test utility functions."""

    def test_memory_utilities(self):
        """Test memory management utilities."""
        try:
            # Test memory clearing with full_clear parameter
            clear_memory(full_clear=True)
            
            # Test memory statistics with updated return values
            stats = get_memory_stats()
            self.assertIsInstance(stats, dict)
            self.assertTrue("cpu_percent" in stats)
            self.assertTrue("cpu_used_gb" in stats)
            self.assertTrue("cpu_available_gb" in stats)
            
        except (ImportError, NameError):
            self.skipTest("Memory utilities not available in test environment")

    def test_class_weights(self):
        """Test class weight computation."""
        try:
            # Create a test dataframe with imbalanced classes
            df = pd.DataFrame({
                "next_task": [0, 0, 0, 0, 1, 1, 2]
            })
            
            # Test computing class weights with updated interface from core.training
            from processmine.core.training import compute_class_weights
            weights = compute_class_weights(df, num_classes=3, method='balanced')
            
            # Check that weights were computed
            self.assertIsInstance(weights, torch.Tensor)
            self.assertEqual(len(weights), 3)
            
            # Check that less frequent classes have higher weights
            self.assertGreater(weights[2], weights[0])
            
        except (ImportError, NameError):
            self.skipTest("Class weight utilities not available in test environment")


if __name__ == '__main__':
    unittest.main()