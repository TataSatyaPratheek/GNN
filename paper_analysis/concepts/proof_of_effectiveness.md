# Proof of Effectiveness

## Empirical Evidence

The authors provide substantial empirical evidence to demonstrate the effectiveness of their GNN-based framework for process map optimization. The evidence is organized around several key experiments and their corresponding results.

## 1. Performance on Next-Activity Prediction

### 1.1 Accuracy and MCC Metrics

The GNN framework demonstrated superior performance in next-activity prediction compared to baseline methods:

| Model | Accuracy | MCC | Validation Loss |
|-------|----------|-----|----------------|
| GNN (MinMax) | 96.24% | 0.9433 | 0.6003 |
| GNN (L2-norm) | 57.43% | 0.5046 | 0.9236 |
| LSTM | 82.25% | - | 0.4920 |
| Random Forest* | ~85%* | ~0.83* | - |
| XGBoost* | ~88%* | ~0.86* | - |

*Values approximated from comparative analysis in the paper

The paper reports reaching up to 97.4% accuracy on some datasets, surpassing both classical machine learning baselines (Random Forest, XGBoost) and sequence-based models (LSTM).

### 1.2 Statistical Significance

The authors report a Matthews Correlation Coefficient (MCC) of up to 0.96, indicating strong statistical significance in their prediction results. This metric is particularly valuable as it provides a balanced measure even with uneven class distributions in the process data.

## 2. Cycle Time Analysis

The cycle time analysis demonstrates the framework's ability to accurately model temporal aspects of workflows:

| Metric | Value |
|--------|-------|
| MAE (hours) | 166.39 |
| R² Score | 0.0000* |

*The R² score of 0 suggests that while the model identifies patterns, there remains significant variance in cycle times not captured by simple linear regression models, highlighting the complexity of the problem.

## 3. Process Mining Statistics

The process mining analysis revealed important patterns in the workflow data:

| Metric | Value |
|--------|-------|
| Long-Running Cases (>95%) | 519 |
| Rare Transitions | 10 |
| Deviant Traces | 0/10,366 |

The framework successfully identified 519 cases exceeding the 95th percentile in execution time and 10 rare transitions that could represent potential anomalies or optimization opportunities.

## 4. Enhancement Evaluations

### 4.1 Norm-Based Representation vs. MinMax Scaling

The experimental results show a significant performance difference between norm-based and MinMax scaling approaches:

| Metric | MinMax | L2-norm |
|--------|--------|---------|
| Accuracy | 96.24% | 57.43% |
| MCC | 0.9433 | 0.5046 |

This unexpected finding indicates that MinMax scaling outperformed the authors' proposed L2-norm approach for this particular dataset, suggesting that normalization techniques must be carefully selected based on data characteristics.

### 4.2 Spectral Clustering Impact

Adding spectral clustering features to the GNN improved performance:

| Method | Accuracy | MCC | Val. Loss |
|--------|----------|-----|-----------|
| Original GAT (no cluster) | 57.43% | 0.5046 | 0.9236 |
| +Spectral Feature | 62.15% | 0.5381 | 0.9156 |

This demonstrates a 4.72% absolute improvement in accuracy when incorporating global graph structure information through spectral clustering.

### 4.3 Reinforcement Learning Integration

The RL-based optimization showed improvements over baseline scheduling:

| Policy | Final Reward Range | Max Reward |
|--------|---------------------|------------|
| Naive Baseline | -400 to -50 | -120 (typical) |
| RL Q-Learning | -315.21 to +0.52 | +0.52 |

While many episodes still resulted in negative rewards due to heavy cost/delay penalties, the RL agent discovered more efficient scheduling policies than the baseline.

## 5. Scalability Demonstrations

The authors provide evidence that their approach scales to large process maps through:

1. Successful training on synthetic workflows with up to 50,000 tasks
2. Effective mini-batching and graph sampling techniques
3. Adaptive batch sizes (16-128) based on workflow complexity

## 6. Robustness Analysis

The framework showed robustness in several dimensions:

1. **Data Quality**: Maintained performance even with noisy or incomplete task data
2. **Feature Engineering**: Demonstrated effectiveness with both raw and engineered features
3. **Hyperparameter Sensitivity**: Exhibited stability across different learning rates and batch sizes
4. **Generalization**: Performed well on both real-world ERP processes and synthetic workflows

## Critical Assessment

While the empirical results are compelling, several limitations should be noted:

1. The unexpected superiority of MinMax scaling over L2-norm suggests that the proposed normalization approach may be dataset-dependent.
2. The cycle time prediction yielded a poor R² score, indicating room for improvement in modeling temporal dynamics.
3. The RL agent, while showing improvement, still experienced many negative reward episodes, suggesting challenges in joint optimization of next-activity and resource allocation.
4. The spectral clustering revealed minimal structural diversity in the dataset (most tasks remained in one cluster), potentially limiting the generalizable insights from this technique.

Nevertheless, the comprehensive empirical evaluation across multiple metrics and experimental variations provides strong evidence for the efficacy of the GNN-based framework for process map optimization, particularly in next-activity prediction and identifying workflow patterns.