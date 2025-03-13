import unittest
import torch
import numpy as np
import pandas as pd
import dgl
import tempfile
import os
import shutil
from datetime import datetime, timedelta

class TestDGLGraphCreation(unittest.TestCase):
    """Test cases for DGL graph creation from process data"""
    
    def setUp(self):
        """Set up test data"""
        # Create a simple test dataframe
        self.test_df = pd.DataFrame({
            'case_id': [1, 1, 1, 2, 2, 2],
            'task_id': [0, 1, 2, 0, 2, 3],
            'task_name': ['start', 'process', 'end', 'start', 'process', 'review'],
            'resource_id': [0, 1, 0, 2, 1, 2],
            'resource': ['user1', 'user2', 'user1', 'user3', 'user2', 'user3'],
            'timestamp': [
                datetime.now(),
                datetime.now() + timedelta(hours=1),
                datetime.now() + timedelta(hours=2),
                datetime.now() + timedelta(days=1),
                datetime.now() + timedelta(days=1, hours=1),
                datetime.now() + timedelta(days=1, hours=2)
            ],
            'next_task': [1, 2, -1, 2, 3, -1],
            'feat_task_id': [0.1, 0.2, 0.3, 0.1, 0.3, 0.4],
            'feat_resource_id': [0.5, 0.6, 0.5, 0.7, 0.6, 0.7]
        })
        
        # Remove rows with next_task == -1
        self.test_df = self.test_df[self.test_df['next_task'] != -1]
        
        # Import the graph building function
        try:
            from processmine.data.graphs import build_graph_data
            self.build_graph_data = build_graph_data
            self.imports_successful = True
        except ImportError:
            self.imports_successful = False
            
    def test_graph_building_basic(self):
        """Test basic graph building functionality"""
        if not self.imports_successful:
            self.skipTest("Required imports not available")
            
        # Build graphs with basic configuration
        graphs = self.build_graph_data(
            self.test_df,
            enhanced=False,
            batch_size=2,
            verbose=False
        )
        
        # Verify graphs were created correctly
        self.assertEqual(len(graphs), 2)  # 2 cases
        
        # Check first graph
        g1 = graphs[0]
        self.assertIsInstance(g1, dgl.DGLGraph)
        self.assertEqual(g1.num_nodes(), 2)  # 2 nodes in first case after removing end
        
        # Check node features
        self.assertTrue('feat' in g1.ndata)
        self.assertEqual(g1.ndata['feat'].shape, (2, 2))  # 2 nodes, 2 features
        
        # Check if labels are present
        self.assertTrue('label' in g1.ndata)
        
        # Check edge structure
        self.assertEqual(g1.num_edges(), 1)  # Sequential edges in case 1
        
    def test_graph_building_enhanced(self):
        """Test enhanced graph building with edge features"""
        if not self.imports_successful:
            self.skipTest("Required imports not available")
            
        # Build graphs with enhanced configuration
        graphs = self.build_graph_data(
            self.test_df,
            enhanced=True,
            batch_size=2,
            verbose=False
        )
        
        # Verify graphs were created correctly
        self.assertEqual(len(graphs), 2)  # 2 cases
        
        # Check edge features in first graph
        g1 = graphs[0]
        self.assertTrue('feat' in g1.edata)
        self.assertEqual(g1.edata['feat'].shape, (1, 1))  # 1 edge, 1 feature (time)
        
    def test_graph_building_bidirectional(self):
        """Test bidirectional graph building"""
        if not self.imports_successful:
            self.skipTest("Required imports not available")
            
        # Build graphs with bidirectional edges
        graphs = self.build_graph_data(
            self.test_df,
            enhanced=True,
            bidirectional=True,
            batch_size=2,
            verbose=False
        )
        
        # Verify graphs were created correctly
        self.assertEqual(len(graphs), 2)  # 2 cases
        
        # Check edge count in first graph
        g1 = graphs[0]
        self.assertEqual(g1.num_edges(), 2)  # Original + reverse edges
        
    def test_graph_building_limit_nodes(self):
        """Test limiting nodes per graph"""
        if not self.imports_successful:
            self.skipTest("Required imports not available")
            
        # Build graphs with node limit
        graphs = self.build_graph_data(
            self.test_df,
            enhanced=False,
            limit_nodes=1,  # Limit to 1 node per graph
            batch_size=2,
            verbose=False
        )
        
        # Verify node limits were applied
        for g in graphs:
            self.assertLessEqual(g.num_nodes(), 1)
    
    def test_heterogeneous_graph_building(self):
        """Test heterogeneous graph building"""
        if not self.imports_successful:
            self.skipTest("Required imports not available")
            
        # Import the heterogeneous graph building function
        try:
            from processmine.data.graphs import build_heterogeneous_graph
            
            # Build heterogeneous graphs
            het_graphs = build_heterogeneous_graph(
                self.test_df,
                batch_size=2,
                verbose=False
            )
            
            # Verify graphs were created correctly
            self.assertGreaterEqual(len(het_graphs), 1)
            
            # Check first graph
            g1 = het_graphs[0]
            self.assertIsInstance(g1, dgl.DGLGraph)
            
            # Check if graph has appropriate node types
            self.assertIn('task', g1.ntypes)
            self.assertIn('resource', g1.ntypes)
            
            # Check edge types
            self.assertGreaterEqual(len(g1.canonical_etypes), 1)
            
        except ImportError:
            self.skipTest("Heterogeneous graph building not available")


class TestDGLDataLoader(unittest.TestCase):
    """Test cases for DGL data loading utilities"""
    
    def setUp(self):
        """Set up test graphs"""
        # Create a few simple test graphs
        self.graphs = []
        for i in range(3):
            # Create a graph with i+2 nodes
            num_nodes = i + 2
            g = dgl.graph(([0], [1]))  # Start with a simple edge
            
            # Add more nodes if needed
            if num_nodes > 2:
                src = list(range(1, num_nodes-1))
                dst = list(range(2, num_nodes))
                g.add_edges(src, dst)
            
            # Add node features
            g.ndata['feat'] = torch.randn(num_nodes, 2)
            
            # Add node labels
            g.ndata['label'] = torch.tensor([i % 3] * num_nodes)
            
            self.graphs.append(g)
        
        # Import data loader utilities
        try:
            from dgl.dataloading import GraphDataLoader

            from processmine.utils.dataloader import (
                get_graph_dataloader,
                get_graph_targets,
                get_batch_graphs_from_indices,
                apply_to_nodes,
                apply_to_edges,
                create_node_masks
            )
            self.get_graph_dataloader = get_graph_dataloader
            self.get_graph_targets = get_graph_targets
            self.get_batch_graphs_from_indices = get_batch_graphs_from_indices
            self.apply_to_nodes = apply_to_nodes
            self.apply_to_edges = apply_to_edges
            self.create_node_masks = create_node_masks
            self.imports_successful = True
        except ImportError:
            self.imports_successful = False
    
    def test_graph_dataloader(self):
        """Test graph data loader creation"""
        if not self.imports_successful:
            self.skipTest("Required imports not available")
        
        # Create a dataloader
        loader = self.get_graph_dataloader(
            self.graphs,
            batch_size=2,
            shuffle=False
        )
        
        # Verify dataloader properties
        self.assertEqual(len(loader), 2)  # 3 graphs with batch_size=2
        
        # Iterate through dataloader
        batches = list(loader)
        self.assertEqual(len(batches), 2)
        
        # Check first batch
        batch1 = batches[0]
        self.assertIsInstance(batch1, dgl.DGLGraph)
        self.assertEqual(batch1.batch_size, 2)  # 2 graphs in this batch
    
    def test_get_graph_targets(self):
        """Test extracting graph-level targets"""
        if not self.imports_successful:
            self.skipTest("Required imports not available")
        
        # Batch the graphs
        batch = dgl.batch(self.graphs[:2])
        
        # Get targets
        targets = self.get_graph_targets(batch)
        
        # Verify targets
        self.assertIsInstance(targets, torch.Tensor)
        self.assertEqual(len(targets), 2)  # 2 graphs
        self.assertEqual(targets[0].item(), 0)  # First graph label
        self.assertEqual(targets[1].item(), 1)  # Second graph label
    
    def test_get_batch_graphs_from_indices(self):
        """Test getting graphs by indices"""
        if not self.imports_successful:
            self.skipTest("Required imports not available")
        
        # Get graphs by indices
        selected_graphs = self.get_batch_graphs_from_indices(self.graphs, [0, 2])
        
        # Verify selection
        self.assertEqual(len(selected_graphs), 2)
        self.assertEqual(selected_graphs[0].num_nodes(), 2)  # First graph has 2 nodes
        self.assertEqual(selected_graphs[1].num_nodes(), 4)  # Third graph has 4 nodes
    
    def test_apply_to_nodes(self):
        """Test applying functions to node features"""
        if not self.imports_successful:
            self.skipTest("Required imports not available")
        
        # Define a function to apply
        def double_features(features):
            return features * 2
        
        # Apply function to nodes
        g = self.graphs[0]
        original_features = g.ndata['feat'].clone()
        g_modified = self.apply_to_nodes(g, double_features)
        
        # Verify features were transformed
        self.assertTrue(torch.allclose(g_modified.ndata['feat'], original_features * 2))
    
    def test_create_node_masks(self):
        """Test creating node feature masks"""
        if not self.imports_successful:
            self.skipTest("Required imports not available")
        
        # Create masks
        g = self.graphs[0].clone()
        g_masked = self.create_node_masks(g, mask_ratio=0.5)
        
        # Verify masks were created
        self.assertTrue('mask' in g_masked.ndata)
        self.assertTrue('orig_feat' in g_masked.ndata)
        
        # Check that masked features are zero
        masked_nodes = g_masked.ndata['mask']
        self.assertEqual(masked_nodes.sum().item(), 1)  # 50% of 2 nodes = 1 node
        
        # Verify that masked features are zeroed out
        zeros = torch.zeros_like(g_masked.ndata['feat'][0])
        masked_idx = masked_nodes.nonzero().squeeze()
        self.assertTrue(torch.allclose(g_masked.ndata['feat'][masked_idx], zeros))


class TestDGLModels(unittest.TestCase):
    """Test cases for DGL-based models"""
    
    def setUp(self):
        """Set up test graphs and models"""
        # Create a few test graphs
        self.graphs = []
        for i in range(5):
            # Create a graph with random structure
            num_nodes = i + 3
            src = np.random.randint(0, num_nodes, size=num_nodes * 2)
            dst = np.random.randint(0, num_nodes, size=num_nodes * 2)
            g = dgl.graph((src, dst), num_nodes=num_nodes)
            
            # Add node features
            g.ndata['feat'] = torch.randn(num_nodes, 4)  # 4 features
            
            # Add node labels
            g.ndata['label'] = torch.tensor([i % 3] * num_nodes)
            
            self.graphs.append(g)
        
        # Import model factory
        try:
            from processmine.models.factory import create_model
            from processmine.utils.dataloader import get_graph_dataloader
            
            self.create_model = create_model
            self.get_graph_dataloader = get_graph_dataloader
            self.imports_successful = True
            
            # Create models
            self.gnn_model = create_model(
                'gnn',
                input_dim=4,
                hidden_dim=8,
                output_dim=3,
                num_layers=2
            )
            
            self.enhanced_gnn = create_model(
                'enhanced_gnn',
                input_dim=4,
                hidden_dim=8,
                output_dim=3,
                num_layers=2
            )
            
            # Create dataloader
            self.loader = get_graph_dataloader(
                self.graphs,
                batch_size=2,
                shuffle=False
            )
            
        except ImportError:
            self.imports_successful = False
    
    def test_gnn_model_init(self):
        """Test GNN model initialization"""
        if not self.imports_successful:
            self.skipTest("Required imports not available")
        
        # Verify model properties
        self.assertEqual(self.gnn_model.input_dim, 4)
        self.assertEqual(self.gnn_model.hidden_dim, 8)
        self.assertEqual(self.gnn_model.output_dim, 3)
        self.assertEqual(self.gnn_model.num_layers, 2)
        self.assertEqual(self.gnn_model.attention_type, "basic")
    
    def test_enhanced_gnn_model_init(self):
        """Test enhanced GNN model initialization"""
        if not self.imports_successful:
            self.skipTest("Required imports not available")
        
        # Verify model properties
        self.assertEqual(self.enhanced_gnn.input_dim, 4)
        self.assertEqual(self.enhanced_gnn.hidden_dim, 8)
        self.assertEqual(self.enhanced_gnn.output_dim, 3)
        self.assertEqual(self.enhanced_gnn.num_layers, 2)
        self.assertEqual(self.enhanced_gnn.attention_type, "combined")
    
    def test_gnn_forward_pass(self):
        """Test GNN forward pass"""
        if not self.imports_successful:
            self.skipTest("Required imports not available")
        
        # Get a batch
        for batch in self.loader:
            # Run forward pass
            outputs = self.gnn_model(batch)
            
            # Verify outputs
            self.assertIsInstance(outputs, dict)
            self.assertIn('task_pred', outputs)
            self.assertEqual(outputs['task_pred'].shape[0], batch.batch_size)
            self.assertEqual(outputs['task_pred'].shape[1], 3)  # 3 output classes
            break
    
    def test_enhanced_gnn_forward_pass(self):
        """Test enhanced GNN forward pass"""
        if not self.imports_successful:
            self.skipTest("Required imports not available")
        
        # Get a batch
        for batch in self.loader:
            # Run forward pass
            outputs = self.enhanced_gnn(batch)
            
            # Verify outputs
            self.assertIsInstance(outputs, dict)
            self.assertIn('task_pred', outputs)
            self.assertEqual(outputs['task_pred'].shape[0], batch.batch_size)
            self.assertEqual(outputs['task_pred'].shape[1], 3)  # 3 output classes
            
            # Check for diversity loss
            self.assertIn('diversity_loss', outputs)
            self.assertIn('diversity_weight', outputs)
            break
    
    def test_get_embeddings(self):
        """Test getting embeddings from model"""
        if not self.imports_successful:
            self.skipTest("Required imports not available")
        
        # Get a batch
        for batch in self.loader:
            # Get embeddings
            embeddings, batch_indices = self.gnn_model.get_embeddings(batch)
            
            # Verify outputs
            self.assertEqual(embeddings.shape[0], batch.num_nodes())
            self.assertEqual(embeddings.shape[1], 8 * 4)  # hidden_dim * heads
            self.assertEqual(len(batch_indices), batch.num_nodes())
            break
    
    def test_get_attention_weights(self):
        """Test getting attention weights from model"""
        if not self.imports_successful:
            self.skipTest("Required imports not available")
        
        # Get a batch
        for batch in self.loader:
            # Get attention weights
            attn_weights = self.gnn_model.get_attention_weights(batch)
            
            # Verify outputs
            self.assertIsInstance(attn_weights, list)
            self.assertEqual(len(attn_weights), 2)  # 2 layers
            break


class TestTrainingWithDGL(unittest.TestCase):
    """Test cases for training with DGL models"""
    
    def setUp(self):
        """Set up test data and models"""
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        
        # Create test graphs
        self.graphs = []
        for i in range(10):
            # Create a graph with random structure
            num_nodes = 5
            src = list(range(num_nodes-1))
            dst = list(range(1, num_nodes))
            g = dgl.graph((src, dst), num_nodes=num_nodes)
            
            # Add node features
            g.ndata['feat'] = torch.randn(num_nodes, 4)  # 4 features
            
            # Add node labels
            g.ndata['label'] = torch.tensor([i % 3] * num_nodes)
            
            self.graphs.append(g)
        
        # Import training utilities
        try:
            from processmine.models.factory import create_model
            from processmine.utils.dataloader import get_graph_dataloader
            from processmine.core.training import (
                train_model,
                evaluate_model,
                create_optimizer,
                create_lr_scheduler
            )
            
            self.create_model = create_model
            self.get_graph_dataloader = get_graph_dataloader
            self.train_model = train_model
            self.evaluate_model = evaluate_model
            self.create_optimizer = create_optimizer
            self.create_lr_scheduler = create_lr_scheduler
            
            self.imports_successful = True
            
            # Create model
            self.model = create_model(
                'gnn',
                input_dim=4,
                hidden_dim=8,
                output_dim=3,
                num_layers=2
            )
            
            # Split graphs into train/val/test
            indices = np.arange(len(self.graphs))
            np.random.shuffle(indices)
            
            train_idx = indices[:6]
            val_idx = indices[6:8]
            test_idx = indices[8:]
            
            # Get subsets
            from processmine.utils.dataloader import get_batch_graphs_from_indices
            self.train_graphs = get_batch_graphs_from_indices(self.graphs, train_idx)
            self.val_graphs = get_batch_graphs_from_indices(self.graphs, val_idx)
            self.test_graphs = get_batch_graphs_from_indices(self.graphs, test_idx)
            
            # Create data loaders
            self.train_loader = get_graph_dataloader(
                self.train_graphs,
                batch_size=2,
                shuffle=True
            )
            
            self.val_loader = get_graph_dataloader(
                self.val_graphs,
                batch_size=2
            )
            
            self.test_loader = get_graph_dataloader(
                self.test_graphs,
                batch_size=2
            )
            
            # Set up optimizer and loss
            self.optimizer = create_optimizer(
                self.model,
                optimizer_type='adam',
                lr=0.01
            )
            
            self.criterion = torch.nn.CrossEntropyLoss()
            
        except ImportError:
            self.imports_successful = False
    
    def tearDown(self):
        """Clean up resources"""
        shutil.rmtree(self.test_dir)
    
    def test_mini_training_run(self):
        """Test a minimal training run with DGL models"""
        if not self.imports_successful:
            self.skipTest("Required imports not available")
        
        # Define model path
        model_path = os.path.join(self.test_dir, "test_model.pt")
        
        # Create scheduler
        scheduler = self.create_lr_scheduler(
            self.optimizer,
            scheduler_type='step',
            epochs=3
        )
        
        # Train for a few epochs
        model, metrics = self.train_model(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            epochs=3,
            patience=5,
            model_path=model_path,
            lr_scheduler=scheduler
        )
        
        # Verify training metrics were collected
        self.assertIn('train_loss', metrics)
        self.assertEqual(len(metrics['train_loss']), 3)  # 3 epochs
        
        # Verify model was saved
        self.assertTrue(os.path.exists(model_path))
    
    def test_evaluation(self):
        """Test model evaluation"""
        if not self.imports_successful:
            self.skipTest("Required imports not available")
        
        # Evaluate model
        eval_metrics, predictions, true_labels = self.evaluate_model(
            model=self.model,
            data_loader=self.test_loader,
            criterion=self.criterion
        )
        
        # Verify evaluation metrics
        self.assertIsInstance(eval_metrics, dict)
        self.assertIn('accuracy', eval_metrics)
        self.assertIn('f1_weighted', eval_metrics)
        
        # Verify predictions
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), len(self.test_graphs))
        
        # Verify true labels
        self.assertIsInstance(true_labels, np.ndarray)
        self.assertEqual(len(true_labels), len(self.test_graphs))


if __name__ == '__main__':
    unittest.main()