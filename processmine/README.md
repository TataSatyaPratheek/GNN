# Process Mining Package

This package provides utilities for process mining analysis using Graph Neural Networks, LSTMs, and Reinforcement Learning.

## Features

- Data preprocessing and feature engineering for process mining
- Graph-based models (GAT, Enhanced GNN) for process analysis
- Sequence models (LSTM) for predicting next activities
- Reinforcement Learning for process optimization
- Extensive visualization tools
- Memory optimization for large datasets
- Ablation study utilities

## Installation

```bash
pip install process-mining
```

## Usage

```python
from process_mining.core.runner import run_analysis
from process_mining.models import create_model

# Run full analysis
run_analysis("path/to/data.csv", model_type="enhanced_gnn")

# Create and use a specific model
model = create_model("gat", input_dim=10, hidden_dim=64, output_dim=5)
```

## License

Apache License 2.0