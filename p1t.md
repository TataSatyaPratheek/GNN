# Code Optimization Analysis for ProcessMine

After carefully reviewing your code, I've identified several areas where code can be deleted to improve maintainability and efficiency, as well as incomplete implementations that should be addressed.

## Code to Delete

### Entire Files
1. **processmine/visualization/visualizer.py** - This entire file (~600 lines) should be removed as `viz.py` provides the same functionality with better memory optimization.

2. **processmine/process_mining/bottlenecks.py** - The entire file can be deleted as it duplicates functionality already present in `analysis.py`. Functions like `analyze_bottlenecks_vectorized` and `analyze_cycle_times_vectorized` are redundant.

### Specific Code Sections

#### CLI Module
- **processmine/cli.py**:
  - Lines 55-115: The multiple `_add_*_arguments` functions can be consolidated into a configuration-based approach
  - Lines 190-252: `setup_environment` has excessive error handling and boilerplate
  - Lines 400-500: Duplicated progress tracking and logging code in multiple run functions

#### Core Module
- **processmine/core/training.py**:
  - Lines 242-281: `_evaluate_during_training` duplicates logic from `evaluate_model`
  - Lines 287-335: `_compute_loss` uses repeated conditional patterns
  - Lines 370-450: Multiple utility functions that should be centralized

#### Models Module
- **processmine/models/baseline/tree_models.py**:
  - The entire class hierarchy can be simplified; `DecisionTreeModel`, `RandomForestModel`, and `XGBoostModel` have ~70% duplicate code
  - `fit()`, `predict()`, and `evaluate()` methods follow identical patterns

- **processmine/models/gnn/architectures.py**:
  - Lines 340-400: The `_dense_forward` and `_sparse_forward` methods could be consolidated
  - Multiple attention implementations with duplicated code patterns

#### Data Module
- **processmine/data/loader.py**:
  - Lines 145-190: Redundant type handling in `_optimize_dtypes`
  - Lines 210-290: Excessive chunking logic that could be simplified

- **processmine/data/graph_builder.py**:
  - Lines 340-430: Duplicate matrix operations in `_build_graphs_sparse`

#### Utils Module
- **processmine/utils/memory.py**:
  - This entire file could be merged with similar functions from other modules

## Incomplete Implementations

1. **LSTM for Graph Data Processing**
   - **processmine/models/sequence/lstm.py**:
     - Line 467: `_process_graph_data` has incomplete handling of PyG Data objects
     - Line 491: Task ID extraction assumes fixed column positions

2. **Reinforcement Learning Implementation**
   - **processmine/process_mining/optimization.py**:
     - Line 242: `run_q_learning` implementation lacks proper exploration strategies
     - Line 59: Environment model uses simplistic reward function

3. **GNN Attention Mechanisms**
   - **processmine/models/gnn/architectures.py**:
     - Line 680: `DiverseGATConv` has incomplete diversity loss calculation
     - Line 750: `CombinedGATConv` doesn't fully integrate positional and diversity components

4. **Conformance Checking**
   - **processmine/process_mining/conformance.py**:
     - Line 65: Error handling is minimal with a very basic fallback
     - Line 112: `identify_conformance_violations` lacks proper violation categorization

5. **Heterogeneous Graph Building**
   - **processmine/data/graph_builder.py**:
     - Line 492: `build_heterogeneous_graph` doesn't properly handle edge attributes
     - Line 533: Node feature extraction is incomplete

Now that I've identified the redundant code and incomplete implementations, I'll proceed to create more comprehensive test cases for each key component of the ProcessMine package.

# ProcessMine Optimization Plan

Based on my analysis of the ProcessMine codebase, I'll now provide a detailed plan for implementing the improvements identified earlier. This will serve as a roadmap for refactoring and optimizing the package.

## Refactoring Priority Areas

### 1. Consolidate Duplicate Functionality

The most pressing issue is the duplicate functionality across multiple files. Here's an implementation plan:

#### Data Module

```python
# Create a centralized feature engineering function
def normalize_features(features, method='l2'):
    """
    Unified normalization function for all feature types.
    Consolidates duplicate code from loader.py and preprocessing.py
    """
    # Implementation that handles all normalization methods
    # This will replace _normalize_features in loader.py and the 
    # duplicate logic in preprocessing.py
```

#### Visualization Module

```python
# Replace visualizer.py with a single viz.py that has memory optimization
# The memory-efficient ProcessVisualizer class should be the only one used
```

#### Process Mining Module

```python
# Merge bottlenecks.py into analysis.py
# Use decorators for vectorization options:

@vectorize_option
def analyze_bottlenecks(df, freq_threshold=5, percentile_threshold=90.0):
    """
    Single implementation with vectorization option controlled by decorator
    """
```

### 2. Optimize Core Architecture

The core architecture needs several improvements:

```python
# Abstract GNN implementations to avoid duplication
class BaseGATConv(MessagePassing):
    """Base class for all GAT implementations to reduce code duplication"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(aggr='add', node_dim=0)
        # Common initialization code for all attention implementations
    
    def forward(self, x, edge_index, return_attention_weights=False):
        """Template method pattern for forward pass"""
        # 1. Common preprocessing
        # 2. Call specific attention implementation
        # 3. Common postprocessing
    
    def _compute_attention(self, x_i, x_j):
        """To be implemented by subclasses"""
        raise NotImplementedError
```

### 3. Fix Incomplete Implementations

#### LSTM Graph Processing

```python
def _process_graph_data(self, data):
    """
    Process PyG Data by extracting case-level sequences
    
    Args:
        data: PyG Data object
        
    Returns:
        Output logits [batch_size, num_classes]
    """
    # Extract batch indices
    batch = data.batch
    
    # Get unique batch indices (cases)
    unique_batches = torch.unique(batch)
    batch_size = unique_batches.size(0)
    
    # Extract sequence for each case
    sequences = []
    seq_lengths = []
    
    for b in unique_batches:
        # Get nodes for this case
        mask = (batch == b)
        case_nodes = data.x[mask]
        
        # Extract task IDs safely - don't assume column ordering
        if case_nodes.size(1) > 1:
            # Find the right feature column - look for task_id patterns
            # or use the first column as a fallback
            task_feature_idx = 0
            if hasattr(data, 'feature_names'):
                for i, name in enumerate(data.feature_names):
                    if 'task' in name.lower():
                        task_feature_idx = i
                        break
            
            task_ids = case_nodes[:, task_feature_idx].long()
        else:
            # Single feature node
            task_ids = case_nodes.squeeze(-1).long()
        
        # Store sequence and length
        sequences.append(task_ids)
        seq_lengths.append(len(task_ids))
    
    # Process sequences with existing method
    return self._process_sequence_list(sequences, seq_lengths)
```

#### Reinforcement Learning Optimization

```python
class ProcessEnv:
    """
    Environment for process optimization using RL
    """
    def _compute_reward(self, current_task, next_task, resource):
        """
        Improved reward function with multiple components
        
        Returns:
            Tuple of (total_reward, reward_components)
        """
        # 1. Transition cost (based on historical data)
        transition_cost = self._compute_transition_cost(current_task, next_task)
        
        # 2. Resource efficiency (workload balancing)
        resource_efficiency = self._compute_resource_efficiency(resource)
        
        # 3. Time efficiency (based on historical task-resource performance)
        time_efficiency = self._compute_time_efficiency(next_task, resource)
        
        # 4. Process conformance (reward staying on common paths)
        conformance = self._compute_conformance_reward(current_task, next_task)
        
        # 5. Goal orientation (reward approaching end activities)
        goal_reward = self._compute_goal_reward(next_task)
        
        # Calculate weighted total reward
        total_reward = (
            -transition_cost * self.weights['cost'] +
            resource_efficiency * self.weights['resource'] +
            time_efficiency * self.weights['time'] +
            conformance * self.weights['conformance'] +
            goal_reward * self.weights['goal']
        )
        
        # Return total and components for analysis
        return total_reward, {
            'transition_cost': transition_cost,
            'resource_efficiency': resource_efficiency,
            'time_efficiency': time_efficiency,
            'conformance': conformance,
            'goal_reward': goal_reward
        }
```

## Performance Improvement Recommendations

Based on the code analysis, here are key performance improvements:

### 1. Memory Management

```python
def estimate_memory_requirements(data_size):
    """
    Estimate memory requirements based on data size
    
    Args:
        data_size: Number of events in the dataset
        
    Returns:
        Dictionary with memory estimates
    """
    # Base memory per event
    bytes_per_event = 200  # Base size
    
    # Graph memory with overhead
    graph_overhead = 1.5  # Overhead factor for graph structures
    
    # Estimate memory for different operations
    estimates = {
        'loading': data_size * bytes_per_event / (1024**3),  # GB
        'graph_building': data_size * bytes_per_event * graph_overhead / (1024**3),  # GB
        'model_training': data_size * bytes_per_event * 0.2 / (1024**3)  # GB (per batch)
    }
    
    return estimates
```

### 2. Dynamic Chunking

```python
def adaptive_chunk_size(file_size, memory_limit_gb=4.0):
    """
    Adaptively determine chunk size based on file size and memory limit
    
    Args:
        file_size: Size of input file in bytes
        memory_limit_gb: Memory limit in GB
        
    Returns:
        Optimal chunk size in rows
    """
    # Estimate rows based on file size
    avg_row_size = 200  # Bytes per row (estimate)
    total_rows = file_size / avg_row_size
    
    # Available memory with safety margin (70% of limit)
    available_memory = memory_limit_gb * 0.7 * (1024**3)  # Bytes
    
    # Memory expansion factor during processing
    expansion_factor = 3.0
    
    # Calculate rows per chunk
    rows_per_chunk = int(available_memory / (avg_row_size * expansion_factor))
    
    # Set reasonable bounds
    min_chunk = 10000
    max_chunk = 500000
    
    return max(min_chunk, min(rows_per_chunk, max_chunk, int(total_rows)))
```

## Project Structure Improvements

Here's a recommended project structure improvement:

```
processmine/
├── core/
│   ├── __init__.py
│   ├── training.py  # Consolidated training functions
│   └── config.py    # Centralized configuration
├── data/
│   ├── __init__.py
│   ├── loader.py    # Data loading with memory optimization
│   ├── features.py  # Feature engineering (NEW)
│   └── graphs.py    # Graph building (renamed from graph_builder.py)
├── models/
│   ├── __init__.py
│   ├── factory.py   # Model factory (NEW)
│   ├── gnn/
│   │   ├── __init__.py
│   │   ├── attention.py  # Attention mechanisms (NEW)
│   │   └── models.py     # GNN models (renamed from architectures.py)
│   └── sequence/
│       ├── __init__.py
│       └── lstm.py
├── process_mining/
│   ├── __init__.py
│   ├── analysis.py  # Consolidated analysis functions
│   └── optimization.py
├── utils/
│   ├── __init__.py
│   ├── memory.py
│   └── metrics.py   # Evaluation metrics (NEW)
└── visualization/
    ├── __init__.py
    └── viz.py       # Single visualization module
```

## Documentation Improvements

Add comprehensive docstrings following Google style:

```python
def function_name(param1, param2):
    """One-line summary of function purpose.
    
    Detailed description of function functionality, including
    any algorithm details or implementation notes.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: Description of when/why exception is raised
        
    Examples:
        >>> function_name(1, 2)
        3
    """
```

## Implementation Schedule

Here's a recommended implementation schedule:

1. **Phase 1: Core Cleanup (1-2 weeks)**
   - Delete duplicate files (visualizer.py, bottlenecks.py)
   - Consolidate core functions
   - Fix incomplete implementations

2. **Phase 2: Architecture Refactoring (2-3 weeks)**
   - Implement base classes for models
   - Refactor attention mechanisms
   - Optimize memory management

3. **Phase 3: Testing & Performance (1-2 weeks)**
   - Implement test suite
   - Benchmark and optimize performance

4. **Phase 4: Documentation & Examples (1 week)**
   - Add comprehensive docstrings
   - Create example notebooks

## Potential Third-Party Dependencies to Consider

1. **Memory Optimization**
   - `memory_profiler` for monitoring memory usage during development
   - `psutil` for system resource monitoring (already using)

2. **Visualization Enhancement**
   - `plotly` for interactive visualizations
   - `holoviews` or `bokeh` for more advanced dashboards

3. **Parallel Processing**
   - `dask` for out-of-core processing of very large datasets
   - `ray` for distributed training of models

Let me know if you'd like me to develop any of these sections in more detail or provide specific code examples for implementing a particular improvement.

# Phase 1: Core Cleanup Implementation

Let's start implementing Phase 1 of our cleanup plan. Since you've already deleted the duplicate files (visualizer.py and bottlenecks.py), we'll focus on consolidating core functions and fixing incomplete implementations.

## 1. Consolidating Core Functions

Let's begin by consolidating duplicated functionality across the codebase:

### 1.1. Centralized Feature Normalization

First, let's create a centralized feature normalization function to replace duplicated code in both loader.py and preprocessing.py:
complete

### 1.2. Consolidate Graph Building Functions

Next, let's consolidate the graph building functions in graph_builder.py:
complete

### 1.3. Consolidate Training Evaluation Functions

Let's consolidate the duplicated evaluation logic in the training module:
complete

### 1.4. Centralize Memory Management

Now, let's centralize memory management utilities:
complete

## 2. Fixing Incomplete Implementations

Now, let's fix the incomplete implementations:




### 2.1. Fix LSTM for Graph Data Processing
complete
### 2.2. Fix Reinforcement Learning Implementation

complete

### 2.3. Fix GNN Attention Mechanisms
complete

### 2.4. Fix Conformance Checking
Complete

### 2.5. Fix Heterogeneous Graph Building

```python
# processmine/data/graph_builder.py (UPDATED FUNCTION)

def build_heterogeneous_graph(
    df: pd.DataFrame, 
    node_types: Optional[Dict[str, List[str]]] = None, 
    edge_types: Optional[List[str]] = None, 
    batch_size: int = 1000, 
    verbose: bool = True,
    use_edge_attr: bool = True
) -> List[Dict]:
    """
    Build heterogeneous graph data with optimized memory management
    
    Args:
        df: Process data dataframe
        node_types: Dictionary mapping node types to feature columns
                    (e.g., {'task': ['task_id', 'feat_task_id'], 'resource': ['resource_id']})
        edge_types: List of edge types to create
                    (e.g., ['task_to_task', 'task_to_resource', 'resource_to_task'])
        batch_size: Batch size for memory-efficient processing
        verbose: Whether to print progress information
        use_edge_attr: Whether to include edge attributes
        
    Returns:
        List of heterogeneous graph data objects compatible with PyG
    """
    if verbose:
        logger.info("Building heterogeneous graph data")
    start_time = time.time()
    
    # Define default node types if not provided
    if node_types is None:
        task_cols = [col for col in df.columns if col.startswith('feat_task') or col == 'task_id']
        resource_cols = [col for col in df.columns if col.startswith('feat_resource') or col == 'resource_id']
        
        node_types = {
            "task": task_cols,
            "resource": resource_cols
        }
    
    # Define default edge types if not provided
    if edge_types is None:
        edge_types = ["task_to_task", "task_to_resource", "resource_to_task"]
    
    # Get unique case IDs
    case_ids = df["case_id"].unique()
    num_cases = len(case_ids)
    
    # Function for tracking progress
    progress_tracker = _progress_tracker(num_cases, verbose, "Building heterogeneous graphs")
    
    # Process in batches
    het_graphs = []
    
    for i in range(0, num_cases, batch_size):
        batch_end = min(i + batch_size, num_cases)
        batch_case_ids = case_ids[i:batch_end]
        
        # Filter dataframe for current batch
        batch_df = df.loc[df["case_id"].isin(batch_case_ids)].copy()
        
        # Pre-sort by case_id and timestamp
        batch_df.sort_values(["case_id", "timestamp"], inplace=True)
        
        # Process each case
        for case_id, case_group in batch_df.groupby("case_id"):
            # Dictionary to store all node data
            node_data = {}
            
            # Dictionary to map node IDs to indices
            node_indices = {}
            
            # Dictionary to store edge data for each edge type
            edge_data = {}
            
            # Extract nodes and features for each node type
            for node_type, feature_cols in node_types.items():
                # Get valid feature columns
                valid_cols = [col for col in feature_cols if col in case_group.columns]
                
                if not valid_cols:
                    if verbose:
                        logger.warning(f"No valid columns found for node type {node_type}")
                    continue
                
                # Get unique node IDs
                if node_type == "task":
                    node_ids = case_group["task_id"].values
                    id_col = "task_id"
                elif node_type == "resource":
                    node_ids = case_group["resource_id"].values
                    id_col = "resource_id"
                else:
                    # For custom node types, use the first column as ID
                    id_col = valid_cols[0]
                    node_ids = case_group[id_col].values
                
                # Extract features
                if len(valid_cols) > 1:
                    # Use all feature columns
                    features = case_group[valid_cols].values
                else:
                    # Use ID column as feature
                    features = case_group[id_col].values.reshape(-1, 1)
                
                # Convert to tensors
                node_features = torch.tensor(features, dtype=torch.float32)
                
                # Store node data
                node_data[node_type] = node_features
                
                # Map node IDs to indices
                for idx, node_id in enumerate(node_ids):
                    if node_type not in node_indices:
                        node_indices[node_type] = {}
                    node_indices[node_type][node_id] = idx
            
            # Create edges for each edge type
            for edge_type in edge_types:
                if edge_type == "task_to_task":
                    # Sequential task transitions
                    n_tasks = len(case_group)
                    if n_tasks > 1:
                        sources = list(range(n_tasks - 1))
                        targets = list(range(1, n_tasks))
                        
                        # Create edge indices
                        edge_indices = torch.tensor([sources, targets], dtype=torch.long)
                        
                        # Create edge attributes if enabled
                        if use_edge_attr:
                            # Add time difference as edge attribute
                            if "timestamp" in case_group.columns:
                                timestamps = case_group["timestamp"].values
                                time_diffs = np.array([
                                    (timestamps[i+1] - timestamps[i]).total_seconds() / 3600.0  # Hours
                                    for i in range(n_tasks-1)
                                ], dtype=np.float32)
                                
                                edge_attrs = torch.tensor(time_diffs, dtype=torch.float32).view(-1, 1)
                            else:
                                # Default edge attributes
                                edge_attrs = torch.ones(len(sources), 1, dtype=torch.float32)
                        else:
                            edge_attrs = None
                        
                        # Store edge data
                        edge_data[edge_type] = {
                            "edge_index": edge_indices,
                            "edge_attr": edge_attrs
                        }
                
                elif edge_type == "task_to_resource":
                    # Connect tasks to resources
                    if "task_id" in case_group.columns and "resource_id" in case_group.columns:
                        task_indices = []
                        resource_indices = []
                        
                        for idx, row in case_group.iterrows():
                            task_id = row["task_id"]
                            resource_id = row["resource_id"]
                            
                            if "task" in node_indices and task_id in node_indices["task"] and \
                               "resource" in node_indices and resource_id in node_indices["resource"]:
                                task_idx = node_indices["task"][task_id]
                                resource_idx = node_indices["resource"][resource_id]
                                
                                task_indices.append(task_idx)
                                resource_indices.append(resource_idx)
                        
                        if task_indices and resource_indices:
                            # Create edge indices
                            edge_indices = torch.tensor([task_indices, resource_indices], dtype=torch.long)
                            
                            # Create edge attributes
                            if use_edge_attr:
                                # Default edge attributes
                                edge_attrs = torch.ones(len(task_indices), 1, dtype=torch.float32)
                            else:
                                edge_attrs = None
                            
                            # Store edge data
                            edge_data[edge_type] = {
                                "edge_index": edge_indices,
                                "edge_attr": edge_attrs
                            }
                
                elif edge_type == "resource_to_task":
                    # Connect resources to tasks (reverse of task_to_resource)
                    if "task_to_resource" in edge_data:
                        task_to_resource = edge_data["task_to_resource"]
                        
                        # Reverse source and target indices
                        edge_indices = torch.stack([
                            task_to_resource["edge_index"][1],