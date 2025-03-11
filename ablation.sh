#!/bin/bash
set -euo pipefail

# Original parameters unchanged
DATASET="input/BPI2020_DomesticDeclarations.csv"
OUTPUT_DIR="ablation_results"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

# decision_tree random_forest xgboost 
for MODEL in mlp lstm basic_gat positional_gat diverse_gat enhanced_gnn; do
    echo "Running ablation for: $MODEL"
    
    (
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
        python main.py $DATASET \
            --run-ablation \
            --model-type $MODEL \
            --output-dir "${OUTPUT_DIR}/${MODEL}" \
            --batch-size 32 \
            --epochs 5 | tee "${LOG_DIR}/${MODEL}.log"
    )
    
    sleep 1  
done

echo "All ablation studies completed"
