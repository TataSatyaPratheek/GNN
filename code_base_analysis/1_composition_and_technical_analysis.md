# Codebase Composition and Technical Analysis

## Overview of Architecture

The codebase implements a comprehensive process mining and optimization framework using Graph Neural Networks (GNNs), sequence modeling, and reinforcement learning. It represents a multi-paradigm approach to business process analysis that combines structural, temporal, and optimization aspects within a unified architecture.

## Core Components

### 1. Application Entry Point (`main.py`)

The main orchestration script implements a complete workflow for process mining, following a clearly delineated execution pattern:

```python
# Simplified execution flow

run_dir = setup_results_dir()

df = load_and_preprocess_data(data_path)

df, le_task, le_resource = create_feature_representation(df, use_norm_features=True)

graphs = build_graph_data(df)

gat_model = train_gat_model(...)

y_true, y_pred, y_prob = evaluate_gat_model(gat_model, val_loader, device)

train_seq, test_seq = prepare_sequence_data(df)

lstm_model = train_lstm_model(...)

bottleneck_stats, significant_bottlenecks = analyze_bottlenecks(df)

case_merged, long_cases, cut95 = analyze_cycle_times(df)

transitions, trans_count, prob_matrix = analyze_transition_patterns(df)

cluster_labels = spectral_cluster_graph(adj_matrix, k=3)

env = ProcessEnv(df, le_task, dummy_resources)

q_table = run_q_learning(env, episodes=30)
```

This demonstrates a systematic execution pattern from data preparation through multiple modeling paradigms to final analysis and optimization.

### 2. Graph Neural Network Model (`models/gat_model.py`)

The GNN implementation uses Graph Attention Networks (GAT) with the following characteristics:

```python
class NextTaskGAT(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, heads=4, dropout=0.5):

        super().__init__()

        self.convs = nn.ModuleList()

        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, concat=True))

        for _ in range(num_layers-1):

            self.convs.append(GATConv(hidden_dim*heads, hidden_dim, heads=heads, concat=True))

        self.fc = nn.Linear(hidden_dim*heads, output_dim)

        self.dropout = dropout
```

Key technical aspects:

1. **Multi-layer Architecture**: Implements a variable-depth network with configurable number of layers
2. **Multi-head Attention**: Uses multiple attention heads (default: 4) with concatenation
3. **Regularization**: Employs dropout (0.5) for model regularization
4. **Activation Function**: Uses ELU activation between convolutional layers
5. **Readout Function**: Implements global mean pooling as the graph-level readout function
6. **Prediction Layer**: Uses a final linear layer for classification output

The training procedure employs:
- Class weight balancing to address imbalanced task distributions
- Early stopping based on validation loss
- AdamW optimizer with weight decay (5e-4)

### 3. Sequence Modeling Implementation (`models/lstm_model.py`)

The sequence model uses an LSTM architecture for temporal pattern recognition:

```python
class NextActivityLSTM(nn.Module):

    def __init__(self, num_cls, emb_dim=64, hidden_dim=64, num_layers=1):

        super().__init__()

        self.emb = nn.Embedding(num_cls+1, emb_dim, padding_idx=0)

        self.lstm = nn.LSTM(emb_dim, hidden_dim, 
        num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, num_cls)
```

Key technical aspects:

1. **Embedding Layer**: Uses learnable embeddings for task representation
2. **Variable Sequence Handling**: Properly handles variable-length sequences using PackedSequence
3. **Padding Mechanism**: Implements padding with masking for efficient batch processing
4. **Sequence Sorting**: Employs length-based sequence sorting for computational efficiency

The LSTM model serves as a complementary paradigm to the GNN, focusing on temporal patterns rather than structural relationships.

### 4. Data Preprocessing (`modules/data_preprocessing.py`)

The preprocessing module implements several key technical components:

```python
# Feature representation options

features_scaled = scaler.fit_transform(raw_features)  # MinMax scaling

features_normed = normalizer.fit_transform(raw_features)  # L2 normalization

combined_features = features_normed if use_norm_features else features_scaled
```

1. **Dual Normalization Approach**: Implements both MinMax scaling and L2 normalization with a configuration flag
2. **Feature Engineering**: Derives temporal features (day_of_week, hour_of_day)
3. **Graph Construction**: Converts sequential data to graph format with nodes and edges
4. **Class Weight Computation**: Calculates balanced class weights for training

The preprocessing pipeline is crucial as it implements the paper's normalization approach and prepares the data for both graph-based and sequence-based models.

### 5. Process Mining Analysis (`modules/process_mining.py`)

This module implements domain-specific process analysis techniques:

```python
def analyze_bottlenecks(df, freq_threshold=5):

    # ...

    bottleneck_stats["mean_hours"] = bottleneck_stats["mean"]/3600.0

    # ...
```

1. **Bottleneck Analysis**: Quantifies waiting times between activities
2. **Cycle Time Analysis**: Measures end-to-end process duration
3. **Conformance Checking**: Uses inductive mining and token replay
4. **Transition Pattern Analysis**: Computes transition matrices and probabilities
5. **Spectral Clustering**: Employs graph spectral methods for process segmentation

This module connects traditional process mining with the machine learning components by providing process-specific analysis techniques.

### 6. Reinforcement Learning Optimization (`modules/rl_optimization.py`)

The RL module implements:

```python
class ProcessEnv:

    # Environment for process optimization

    # ...

def run_q_learning(env, episodes=30, alpha=0.1, gamma=0.9, epsilon=0.1):

    # Q-learning implementation

    # ...
```

1. **Custom Environment**: Implements a process-specific OpenAI Gym-like environment
2. **State Representation**: Uses one-hot encoding of current task state
3. **Action Space**: Defines actions as (task, resource) pairs
4. **Reward Function**: Combines cost, delay, and resource efficiency components
5. **Q-learning Algorithm**: Implements tabular Q-learning with epsilon-greedy exploration

This component represents the optimization aspect of the framework, enabling decision-making for resource allocation and process execution.

### 7. Visualization Suite (`visualization/process_viz.py`)

The visualization module implements:

1. **Confusion Matrix**: Performance visualization for classification tasks
2. **Embedding Visualization**: Dimension reduction (t-SNE, UMAP) for embeddings
3. **Process Flow Visualization**: Graph-based visualization of process models
4. **Transition Heatmap**: Visual representation of process transitions
5. **Sankey Diagram**: Flow-based visualization of process paths

## Technical Implementation Aspects

### Computational Efficiency Considerations

1. **Device Agnosticism**: The code implements device detection for CPU, CUDA, and MPS (Apple Silicon)
2. **Batch Processing**: Implements efficient batch processing for both GNN and LSTM models
3. **Memory Management**: Uses variable sequence handling techniques to optimize memory usage
4. **Matrix Operations**: Vectorized operations for computational efficiency

### Software Engineering Practices

1. **Modular Design**: Clear separation of concerns between modules
2. **Configurability**: Parameterized implementation with sensible defaults
3. **Result Persistence**: Structured output directory with timestamped runs
4. **Error Handling**: Input validation and error messaging
5. **Documentation**: Function-level docstrings and code comments

### Mathematical Implementations

### Mathematical Implementations

1. **Graph Construction**: $G = (V, E, X, \mathbf{E})$ where:
   - $V$ represents tasks
   - $E \subseteq V \times V$ represents dependencies
   - $X \in \mathbb{R}^{n \times d}$ represents node features
   - $\mathbf{E}$ represents edge features

2. **Graph Attention Mechanism**:
   $$
   \alpha_{ij} = \frac{\exp(\text{LeakyReLU}(a^T[Wh_i || Wh_j]))}{\sum_{k \in \mathcal{N}(i)} \exp(\text{LeakyReLU}(a^T[Wh_i || Wh_k]))}
   $$

3. **Norm-Based Feature Representation**:
   $$
   \hat{x} = \frac{x}{||x||_2 + \epsilon}
   $$

4. **Spectral Clustering**:
   - Computes the graph Laplacian $L = D - A$
   - Finds eigenvectors of $L$
   - Uses Fiedler vector or k-means clustering on the embedding

5. **Q-learning Update Rule**:
   $$
   Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]
   $$
   
In summary, the codebase represents a comprehensive implementation that spans from data preparation through multiple modeling paradigms to process analysis and optimization, with attention to software engineering practices and computational efficiency.