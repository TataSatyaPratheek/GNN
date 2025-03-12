# ProcessMine: Memory-Efficient Process Mining

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)

ProcessMine is a high-performance, memory-efficient toolkit for process mining using Graph Neural Networks, LSTMs, and Reinforcement Learning. It provides advanced analytics and predictive capabilities for business process optimization, with a focus on handling large datasets efficiently.

## Key Features

- **Memory Optimization**: Handles large process logs with minimal memory footprint through chunking, vectorization, and efficient data structures
- **Advanced Model Architectures**:
  - Graph Neural Networks with various attention mechanisms (basic, positional, diverse)
  - Sequence models with LSTM and attention for next activity prediction
  - Reinforcement Learning for process optimization
- **Comprehensive Analysis**:
  - Bottleneck detection and analysis
  - Cycle time analysis and forecasting
  - Process variant identification
  - Resource workload analysis
- **Interactive Visualizations**:
  - Process flow diagrams with bottleneck highlighting
  - Sankey diagrams of process transitions
  - Attention heatmaps for model interpretability
  - Dashboards for comprehensive analysis
- **Accelerated Training**:
  - Mixed precision training for GPUs
  - Memory-efficient batching strategies
  - CUDA-optimized implementations
- **Simple Interface**:
  - Unified model interfaces with consistent APIs
  - Command-line tools for quick analysis
  - Python API for integration with existing applications

## Installation

### Quick Installation

```bash
pip install processmine
```

### Install with Optional Dependencies

```bash
# For graph neural networks
pip install "processmine[gnn]"

# For visualizations
pip install "processmine[viz]"

# For traditional machine learning models
pip install "processmine[ml]"

# Full installation with all dependencies
pip install "processmine[all]"

# Development installation
pip install "processmine[all,dev]"
```

### Installation from Source

```bash
git clone https://github.com/erp-ai/processmine.git
cd processmine
pip install -e ".[all]"
```

## Quick Start

### Command Line Usage

```bash
# Basic process analysis
processmine analyze path/to/process_log.csv

# Train a model
processmine train path/to/process_log.csv --model enhanced_gnn

# Full pipeline (analyze, train, optimize)
processmine full path/to/process_log.csv --output-dir results/my_analysis
```

### Python API

```python
from processmine import run_analysis, create_model, load_and_preprocess_data
from processmine.visualization.viz import ProcessVisualizer

# Load and preprocess data
df, task_encoder, resource_encoder = load_and_preprocess_data("process_log.csv")

# Run analysis
analysis_results = run_analysis(df)

# Create and train a model
model = create_model(
    model_type="enhanced_gnn",
    input_dim=len([col for col in df.columns if col.startswith("feat_")]),
    hidden_dim=64,
    output_dim=len(task_encoder.classes_),
    attention_type="combined"
)

# Create visualizations
viz = ProcessVisualizer(output_dir="results/visualizations")
viz.process_flow(
    analysis_results["bottleneck_stats"],
    task_encoder,
    analysis_results["significant_bottlenecks"]
)
```

## Memory Efficiency Guidelines

ProcessMine is designed to handle large process logs efficiently. Here are some tips for maximizing performance:

1. **Use chunking for large files**: When loading large process logs, set appropriate `chunk_size` in `load_and_preprocess_data`.
2. **Enable memory-efficient mode**: Use `mem_efficient=True` for models and training to reduce memory usage at the cost of slightly slower processing.
3. **Optimize batch size**: For very large graphs, use smaller batch sizes with `build_graph_data(batch_size=100)`.
4. **Use CUDA wisely**: Clear CUDA cache when needed with `clear_memory(full_clear=True)`.
5. **Leverage sampling for huge datasets**: For exploratory analysis, sample the data first to get quick insights.

## Documentation

For full documentation, visit [docs.processmine.com](https://docs.processmine.com).

- [Tutorial](https://docs.processmine.com/tutorial)
- [API Reference](https://docs.processmine.com/api)
- [Examples](https://docs.processmine.com/examples)

## Examples

### Bottleneck Analysis and Visualization

```python
import pandas as pd
from processmine.process_mining.analysis import analyze_bottlenecks
from processmine.visualization.viz import ProcessVisualizer

# Load process data
df = pd.read_csv("process_log.csv")

# Analyze bottlenecks
bottleneck_stats, significant_bottlenecks = analyze_bottlenecks(
    df,
    freq_threshold=5,
    percentile_threshold=90.0
)

# Visualize bottlenecks
viz = ProcessVisualizer(output_dir="results")
viz.bottleneck_analysis(bottleneck_stats, significant_bottlenecks, task_encoder)
```

### Training a GNN Model

```python
from processmine import create_model, load_and_preprocess_data
from processmine.data.graph_builder import build_graph_data
from processmine.core.training import train_model, evaluate_model
import torch

# Load and preprocess data
df, task_encoder, resource_encoder = load_and_preprocess_data("process_log.csv")

# Build graph data
graphs = build_graph_data(df, enhanced=True)

# Create model
model = create_model(
    model_type="enhanced_gnn",
    input_dim=len([col for col in df.columns if col.startswith("feat_")]),
    hidden_dim=64,
    output_dim=len(task_encoder.classes_),
    attention_type="combined"
)

# Train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

model, metrics = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    epochs=20
)

# Evaluate model
eval_metrics = evaluate_model(model, test_loader, device)
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use ProcessMine in your research, please cite:

```
@software{processmine2025,
  author = {ERP.AI},
  title = {ProcessMine: Memory-Efficient Process Mining with Graph Neural Networks},
  year = {2025},
  url = {https://github.com/erp-ai/processmine}
}
```