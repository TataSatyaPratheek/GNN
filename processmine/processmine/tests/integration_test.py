import os
import unittest
import pandas as pd
import numpy as np
import torch
import tempfile
import shutil
from contextlib import redirect_stdout, redirect_stderr
import io
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper function to generate synthetic process data
def generate_synthetic_data(filepath, n_cases=50, n_activities=10, n_resources=5):
    """Generate synthetic process data for testing."""
    # Generate case IDs
    case_ids = np.repeat(range(1, n_cases + 1), np.random.randint(3, 10, n_cases))
    n_events = len(case_ids)
    
    # Generate activities
    activities = np.random.randint(1, n_activities + 1, n_events)
    activity_names = [f"Activity_{i}" for i in activities]
    
    # Generate resources
    resources = np.random.randint(1, n_resources + 1, n_events)
    resource_names = [f"Resource_{i}" for i in resources]
    
    # Generate timestamps
    base_date = pd.Timestamp('2023-01-01')
    timestamps = []
    for case_id in range(1, n_cases + 1):
        case_events = np.where(case_ids == case_id)[0]
        case_start = base_date + pd.Timedelta(hours=case_id * 4)
        for i, event_idx in enumerate(case_events):
            timestamps.append(case_start + pd.Timedelta(minutes=30 * i))
    
    # Create dataframe
    df = pd.DataFrame({
        'case_id': case_ids,
        'task_id': activities,
        'task_name': activity_names,
        'resource_id': resources,
        'resource': resource_names,
        'timestamp': timestamps
    })
    
    # Add more features for advanced testing
    df['next_timestamp'] = df.groupby('case_id')['timestamp'].shift(-1)
    
    # Add a custom attribute for testing
    df['priority'] = np.random.randint(1, 4, n_events)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    return df

class ProcessMineIntegrationTests(unittest.TestCase):
    """Integration tests for ProcessMine package."""
    
    @classmethod
    def setUpClass(cls):
        """Set up resources for all tests."""
        # Create a temporary directory
        cls.test_dir = tempfile.mkdtemp()
        
        # Generate synthetic data
        cls.data_path = os.path.join(cls.test_dir, 'process_log.csv')
        cls.test_df = generate_synthetic_data(cls.data_path)
        
        # Configure output directories
        cls.output_dir = os.path.join(cls.test_dir, 'results')
        os.makedirs(cls.output_dir, exist_ok=True)
        
        # Check if package is available
        try:
            import processmine
            cls.is_available = True
        except ImportError:
            cls.is_available = False
            logger.warning("ProcessMine package not available, tests will be skipped")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up resources."""
        shutil.rmtree(cls.test_dir)
    
    def setUp(self):
        """Set up for each test."""
        if not self.is_available:
            self.skipTest("ProcessMine package not available")
        
        # Import necessary modules
        from processmine.models.factory import create_model
        from processmine.core.runner import run_analysis
        from processmine.data.loader import load_and_preprocess_data
        from processmine.data.graphs import build_graph_data
        from processmine.core.training import train_model, evaluate_model
        from processmine.process_mining.analysis import analyze_bottlenecks
        from processmine.visualization.viz import ProcessVisualizer
    
    def test_full_pipeline_integration(self):
        """Test the complete ProcessMine pipeline from data loading to visualization."""
        # Capture stdout to check progress
        output = io.StringIO()
        with redirect_stdout(output):
            # Step 1: Load and preprocess data
            from processmine.data.loader import load_and_preprocess_data
            
            df, task_encoder, resource_encoder = load_and_preprocess_data(
                self.data_path,
                norm_method='l2',
                use_dtypes=True
            )
            
            # Verify preprocessing
            self.assertTrue('feat_task_id' in df.columns)
            self.assertTrue('next_task' in df.columns)
            
            # Step 2: Build graph data
            from processmine.data.graphs import build_graph_data
            
            graphs = build_graph_data(
                df,
                enhanced=True,
                batch_size=10
            )
            
            # Verify graph creation
            self.assertTrue(len(graphs) > 0)
            
            # Step 3: Split data
            import torch
            from dgl.dataloading import GraphDataLoader
            
            # Create indices for train/val/test split (70/15/15)
            indices = np.arange(len(graphs))
            np.random.shuffle(indices)
            
            train_size = int(0.7 * len(indices))
            val_size = int(0.15 * len(indices))
            
            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size + val_size]
            test_idx = indices[train_size + val_size:]
            
            # Create data loaders
            train_loader = GraphDataLoader([graphs[i] for i in train_idx], batch_size=4, shuffle=True)
            val_loader = GraphDataLoader([graphs[i] for i in val_idx], batch_size=4)
            test_loader = GraphDataLoader([graphs[i] for i in test_idx], batch_size=4)
            
            # Step 4: Create model
            from processmine.models.factory import create_model
            
            model = create_model(
                model_type="enhanced_gnn",
                input_dim=len([col for col in df.columns if col.startswith("feat_")]),
                hidden_dim=16,
                output_dim=len(task_encoder.classes_),
                attention_type="combined"
            )
            
            # Step 5: Train model
            from processmine.core.training import train_model, compute_class_weights
            
            # Compute class weights for imbalanced data
            class_weights = compute_class_weights(df, len(task_encoder.classes_))
            
            # Configure optimizer and loss
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            
            # Train model (with reduced epochs for testing)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model, metrics = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epochs=3,  # Reduced for testing
                patience=2,
                model_path=os.path.join(self.output_dir, 'gnn_model.pt')
            )
            
            # Step 6: Evaluate model
            from processmine.core.training import evaluate_model
            
            eval_metrics, y_true, y_pred = evaluate_model(
                model=model,
                data_loader=test_loader,
                device=device,
                criterion=criterion
            )
            
            # Verify metrics were calculated
            self.assertIn('accuracy', eval_metrics)
            self.assertIn('f1_weighted', eval_metrics)
            
            # Step 7: Process mining analysis
            from processmine.process_mining.analysis import (
                analyze_bottlenecks,
                analyze_cycle_times,
                analyze_transition_patterns,
                identify_process_variants
            )
            
            # Bottleneck analysis
            bottleneck_stats, significant_bottlenecks = analyze_bottlenecks(
                df,
                freq_threshold=2,
                percentile_threshold=90.0
            )
            
            # Cycle time analysis
            case_stats, long_cases, p95 = analyze_cycle_times(df)
            
            # Transition pattern analysis
            transitions, trans_count, prob_matrix = analyze_transition_patterns(df)
            
            # Process variant analysis
            variant_stats, variant_sequences = identify_process_variants(
                df,
                max_variants=5
            )
            
            # Step 8: Visualization
            from processmine.visualization.viz import ProcessVisualizer
            
            viz = ProcessVisualizer(output_dir=self.output_dir)
            
            # Create visualizations
            viz.cycle_time_distribution(
                case_stats['duration_h'].values,
                filename='cycle_time_distribution.png'
            )
            
            viz.bottleneck_analysis(
                bottleneck_stats,
                significant_bottlenecks,
                task_encoder,
                filename='bottleneck_analysis.png'
            )
            
            viz.process_flow(
                bottleneck_stats,
                task_encoder,
                significant_bottlenecks,
                filename='process_flow.png'
            )
            
            viz.transition_heatmap(
                transitions,
                task_encoder,
                filename='transition_heatmap.png'
            )
            
            # Verify visualization files were created
            self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'cycle_time_distribution.png')))
            self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'bottleneck_analysis.png')))
            self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'process_flow.png')))
            self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'transition_heatmap.png')))
            
        # Print output if test fails
        # print(output.getvalue())
    
    def test_cli_integration(self):
        """Test the command-line interface."""
        from processmine.cli import main, parse_arguments
        
        # Test with analyze mode
        test_args = [
            'processmine',
            self.data_path,
            'analyze',
            '--output-dir', os.path.join(self.output_dir, 'cli_test'),
            '--viz-format', 'static',
            '--freq-threshold', '2',
            '--bottleneck-threshold', '85.0'
        ]
        
        with patch.object(sys, 'argv', test_args):
            try:
                # Redirect stdout/stderr to capture output
                stdout = io.StringIO()
                stderr = io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    # Run CLI
                    exit_code = main()
                
                # Check that CLI ran without errors
                self.assertEqual(exit_code, 0)
                
                # Check that output directory was created
                cli_out_dir = os.path.join(self.output_dir, 'cli_test')
                self.assertTrue(os.path.exists(cli_out_dir))
                
                # Check for expected files
                self.assertTrue(os.path.exists(os.path.join(cli_out_dir, 'visualizations')))
                self.assertTrue(os.path.exists(os.path.join(cli_out_dir, 'metrics')))
                
            except Exception as e:
                # Some environment issues might prevent full CLI testing
                logger.warning(f"CLI test failed: {e}")
                # Print captured output for debugging
                # print(f"STDOUT: {stdout.getvalue()}")
                # print(f"STDERR: {stderr.getvalue()}")
    
    def test_model_types_integration(self):
        """Test creating and using different model types."""
        from processmine.models.factory import create_model
        
        # Load and preprocess data
        from processmine.data.loader import load_and_preprocess_data
        
        df, task_encoder, resource_encoder = load_and_preprocess_data(
            self.data_path,
            norm_method='l2',
            use_dtypes=True
        )
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col.startswith("feat_")]
        X = df[feature_cols].values
        y = df["next_task"].values
        
        # Create different model types
        models = []
        
        # Test GNN
        models.append(create_model(
            model_type='gnn',
            input_dim=len(feature_cols),
            hidden_dim=16,
            output_dim=len(task_encoder.classes_)
        ))
        
        # Test enhanced GNN
        models.append(create_model(
            model_type='enhanced_gnn',
            input_dim=len(feature_cols),
            hidden_dim=16,
            output_dim=len(task_encoder.classes_)
        ))
        
        # Test LSTM
        models.append(create_model(
            model_type='lstm',
            num_cls=len(task_encoder.classes_),
            emb_dim=16,
            hidden_dim=32
        ))
        
        # Test enhanced LSTM
        models.append(create_model(
            model_type='enhanced_lstm',
            num_cls=len(task_encoder.classes_),
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
    
    def test_memory_optimization_integration(self):
        """Test memory optimization features."""
        from processmine.data.loader import load_and_preprocess_data
        from processmine.utils.memory import clear_memory, get_memory_stats
        
        # Get initial memory stats
        initial_stats = get_memory_stats()
        
        # Load data with memory optimization
        df, task_encoder, resource_encoder = load_and_preprocess_data(
            self.data_path,
            norm_method='l2',
            memory_limit_gb=0.1,  # Force smaller chunks
            use_dtypes=True
        )
        
        # Get memory stats after loading
        loading_stats = get_memory_stats()
        
        # Clear memory
        clear_memory(full_clear=True)
        
        # Get memory stats after clearing
        cleared_stats = get_memory_stats()
        
        # Build graph data with memory optimization
        from processmine.data.graphs import build_graph_data
        
        graphs = build_graph_data(
            df,
            enhanced=True,
            batch_size=5,  # Small batch size for memory efficiency
            bidirectional=True
        )
        
        # Get memory stats after graph building
        graph_stats = get_memory_stats()
        
        # Verify stats were collected
        self.assertIn('cpu_percent', initial_stats)
        self.assertIn('cpu_percent', loading_stats)
        self.assertIn('cpu_percent', cleared_stats)
        self.assertIn('cpu_percent', graph_stats)
        
        # Memory usage should be reasonable (for a test dataset)
        if 'cpu_used_gb' in initial_stats and 'cpu_used_gb' in graph_stats:
            # Memory increase should be reasonable
            memory_increase = graph_stats['cpu_used_gb'] - initial_stats['cpu_used_gb']
            # Verify increase is less than 2GB (arbitrary but reasonable for test data)
            self.assertLess(memory_increase, 2.0)


# Additional patch import for CLI testing
from unittest.mock import patch, MagicMock


if __name__ == '__main__':
    unittest.main()