# Architectural Integration and Component Interactions

## System Architecture Integration

The codebase represents a tightly integrated system that combines multiple paradigms for process analysis. Understanding how these components interact is essential for grasping the overall design philosophy.

## Data Flow and Component Interactions

### 1. Primary Execution Flow

The execution flow in `main.py` reveals the system's fundamental architecture:

```
Data Preprocessing → GNN Model → LSTM Model → Process Analysis → RL Optimization
```

This structure reflects a progression from representation to prediction to optimization, with each component building upon the outputs of previous stages.

### 2. Data Transformation Chain

Data undergoes several transformations as it flows through the system:

```
CSV Data → Pandas DataFrame → Feature Vectors → Graph Objects → Task Embeddings → Predictions
```

Each transformation preserves essential process information while adapting it to the requirements of different modeling paradigms:

1. **CSV to DataFrame**: Raw event log data is parsed into a structured format
2. **DataFrame to Features**: Raw attributes are converted to numerical features
3. **Features to Graphs**: Sequential data is restructured into graph objects
4. **Graphs to Embeddings**: GNN processes graph structure into latent representations
5. **Embeddings to Predictions**: Learned representations are used for predictive tasks

### 3. Model Interaction Pattern

The system implements a multi-model approach where different models capture complementary aspects of the process:

```
          ┌─────────────────┐
          │ Event Log Data  │
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │  Preprocessing  │
          └────────┬────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼───────┐    ┌────────▼─────────┐
│  Graph Data    │    │  Sequence Data   │
└───────┬───────┘    └────────┬─────────┘
        │                     │
┌───────▼───────┐    ┌────────▼─────────┐
│  GAT Model    │    │   LSTM Model     │
└───────┬───────┘    └────────┬─────────┘
        │                     │
┌───────▼───────────┴─────────▼─────────┐
│       Analysis & Optimization         │
└─────────────────────────────────────┬─┘
                                      │
                            ┌─────────▼──────────┐
                            │  Results & Visuals  │
                            └────────────────────┘
```

This architecture enables the system to capture both structural patterns (through the GNN) and temporal sequences (through the LSTM) simultaneously.

### 4. Integration Points Between Components

Several key integration points connect the different components:

#### 4.1. Graph Construction from Sequential Data

```python
def build_graph_data(df):

    """Convert preprocessed data into graph format for GNN"""

    graphs = []

    for cid, cdata in df.groupby("case_id"):

        # Create nodes with features

        x_data = torch.tensor(
            cdata[
            ["feat_task_id",
            "feat_resource_id",
            "feat_amount",
            "feat_day_of_week",
            "feat_hour_of_day"]
            ].values, 
            dtype=torch.float
            )
        
        # Create edges between sequential activities

        n_nodes = len(cdata)

        if n_nodes > 1:

            src = list(range(n_nodes-1))

            tgt = list(range(1,n_nodes))

            edge_index = torch.tensor(
                [src+tgt, tgt+src],
                 dtype=torch.long)

        else:
            edge_index = torch.empty(
                (2,0), 
                dtype=torch.long)
        
        # Add to graph collection

        data_obj = Data(x=x_data, edge_index=edge_index, y=y_data)

        graphs.append(data_obj)

    return graphs
```

This function transforms the sequential event log data into a collection of graph objects, creating a bridge between traditional process data and graph-based modeling.

#### 4.2. Feature Representation Integration

```python
raw_features = df[feature_cols].values

features_scaled = scaler.fit_transform(raw_features)

features_normed = normalizer.fit_transform(raw_features)

combined_features = features_normed if use_norm_features else features_scaled
```

This code segment implements the dual representation approach, allowing for either norm-based or scale-based feature representation based on a configuration flag.

#### 4.3. Predictive Models to Process Analysis

```python
# Evaluate GAT model

y_true, y_pred, y_prob = evaluate_gat_model(gat_model, val_loader, device)

# Process Mining Analysis

bottleneck_stats, significant_bottlenecks = analyze_bottlenecks(df)
```

The outputs from predictive models (GAT) are evaluated and then incorporated with domain-specific process analysis techniques, creating an integration point between machine learning and process mining.

#### 4.4. Process Analysis to RL Environment

```python
# Process analysis results inform RL environment

bottleneck_stats, significant_bottlenecks = analyze_bottlenecks(df)

# Initialize RL environment using process knowledge

env = ProcessEnv(df, le_task, dummy_resources)

q_table = run_q_learning(env, episodes=30)
```

The insights from process analysis are used to structure the reinforcement learning environment, creating a connection between analytical insights and optimization strategies.

## Data Structures as Integration Vehicles

### 1. Graph Data Objects

The PyTorch Geometric `Data` objects serve as a fundamental integration structure:

```python
Data(x=x_data, edge_index=edge_index, y=y_data)
```

These objects encapsulate:
- Node features (activity attributes)
- Edge structure (process flow)
- Target labels (next activities)

This unifies the representation needed for graph-based learning while preserving the essential process information.

### 2. Transition Matrices

Transition probability matrices connect statistical process analysis with machine learning:

```python
trans_count = transitions.groupby(["task_id","next_task_id"]).size().unstack(fill_value=0)

prob_matrix = trans_count.div(trans_count.sum(axis=1), axis=0)
```

These matrices provide:
- Statistical summary of process flows
- Input for visualization components
- Insights for reinforcement learning state transitions

### 3. Embedding Spaces

The learned embeddings represent a crucial integration point:

```python
# Node-level embeddings from GAT

x = global_mean_pool(x, batch)  # Readout function

# Sequence-level embeddings from LSTM

last_hidden = h_n[-1]  # Final hidden state
```

These embedding spaces link:
- Raw process data with higher-level abstractions
- Structural patterns with decision-making logic
- Model outputs with visualization techniques

## Component Coupling Analysis

The codebase exhibits a carefully designed coupling structure:

### 1. Loose Coupling Between Major Components

The major components (GNN, LSTM, Process Mining, RL) maintain loose coupling through:

- Clear interfaces between modules
- Independent model architectures
- Minimal direct dependencies

This allows each component to evolve independently and enables easier testing and maintenance.

### 2. Tighter Coupling in Data Representations

Data representation shows tighter coupling:

- Feature representation choices affect all downstream models
- Graph construction decisions impact GNN performance
- Preprocessing choices influence both predictive and analytical components

This reflects the centrality of data representation in the overall architecture.

### 3. Integration through Orchestration

The `main.py` script serves as an orchestrator that manages component interaction:

```python
# Data flows through components via orchestration

gat_model = train_gat_model(...)

y_true, y_pred, y_prob = evaluate_gat_model(gat_model, val_loader, device)

bottleneck_stats, significant_bottlenecks = analyze_bottlenecks(df)
```

This approach minimizes direct dependencies between components while ensuring proper data flow.

## Technical Integration Patterns

### 1. Pipeline Pattern

The overall system implements a pipeline pattern where data flows through sequential processing stages:

```
Data Loading → Preprocessing → Feature Engineering → Model Training → Analysis → Visualization
```

Each stage transforms the data for subsequent stages while maintaining independence.

### 2. Multiple Model Integration

The system integrates multiple modeling paradigms:

- **GNN**: Capturing structural relationships
- **LSTM**: Modeling sequential patterns
- **Reinforcement Learning**: Optimizing decisions

These models operate on the same underlying data but focus on different aspects of the process.

### 3. Shared State through Data Structures

Rather than direct method calls between components, integration occurs through shared data structures:

- Pandas DataFrames for tabular process data
- PyTorch Geometric Data objects for graph representation
- Numpy arrays for statistical results
- Tensor embeddings for learned representations

This creates a more maintainable and flexible architecture.

## Functional Integration Mechanisms

### 1. Label Encoders as Global References

Label encoders serve as consistent reference points across the system:

```python
df, le_task, le_resource = create_feature_representation(df, use_norm_features=True)

# Later used in visualization

plot_process_flow(bottleneck_stats, le_task, significant_bottlenecks.head(), ...)

# And in reinforcement learning
env = ProcessEnv(df, le_task, dummy_resources)

```

This ensures consistency in task and resource representations across different components.

### 2. Unified Loss Function

The loss function integrates different optimization objectives:

```python
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
```

By incorporating class weights, the system balances:
- Prediction accuracy objectives
- Class balance considerations
- Process-specific performance requirements

### 3. Shared Hyperparameters

Key hyperparameters connect different components:

```python
# GAT model

num_layers=2, heads=4, dropout=0.5

# LSTM model

emb_dim=64, hidden_dim=64, num_layers=1

# RL component

episodes=30, alpha=0.1, gamma=0.9, epsilon=0.1
```

These parameters represent crucial integration points that affect overall system behavior.

## Conclusion: A Multi-Paradigm Integration Architecture

The system architecture demonstrates a sophisticated multi-paradigm integration approach that combines:

1. **Graph-based representation** for structural patterns
2. **Sequence modeling** for temporal patterns
3. **Process mining techniques** for domain-specific analysis
4. **Reinforcement learning** for optimization

Rather than forcing a single modeling paradigm, the architecture allows each component to focus on its strengths while ensuring proper integration through orchestration, shared data structures, and consistent representations.

This reflects an understanding that process mining requires capturing multiple dimensions of process behavior - structural relationships, temporal patterns, and decision optimization - within a cohesive framework.