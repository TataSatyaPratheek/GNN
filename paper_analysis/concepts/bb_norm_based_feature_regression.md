# Building Block: Norm-Based Feature Representation

## Conceptual Overview

Norm-based feature representation is a key innovation in the paper, introduced to standardize embeddings and enhance the robustness of the GNN model when dealing with noisy or incomplete process data. This technique normalizes node and edge features using mathematical norms, ensuring numerical stability and improved representational capacity.

## Mathematical Foundation

### Definition of Vector Norms

For a feature vector $\mathbf{h}_v \in \mathbb{R}^d$ associated with a node $v$, the $p$-norm is defined as:

$$\|\mathbf{h}_v\|_p = \left(\sum_{i=1}^d |h_{v,i}|^p\right)^{1/p}$$

Where:
- $p \geq 1$ determines the type of norm
- $h_{v,i}$ is the $i$-th component of $\mathbf{h}_v$

Common norms used in the paper include:
- $p = 1$: L1 norm (sum of absolute values)
- $p = 2$: L2 norm (Euclidean norm)
- $p = \infty$: Max norm (maximum absolute value of components)

### Norm-Based Feature Update

The norm-based feature update for a node $v$ at layer $t+1$ is formulated as:

$$\mathbf{h}_v^{(t+1)} = \sigma\left(\frac{\mathbf{W}_t \cdot \mathbf{h}_v^{(t)}}{\|\mathbf{h}_v^{(t)}\|_p + \epsilon} + \mathbf{h}_v^{\text{agg}}\right)$$

Where:
- $\mathbf{W}_t$ is a learnable weight matrix for layer $t$
- $\|\mathbf{h}_v^{(t)}\|_p$ is the $p$-norm of the feature vector
- $\epsilon$ is a small constant added for numerical stability
- $\mathbf{h}_v^{\text{agg}}$ is the aggregated information from neighboring nodes
- $\sigma$ is a non-linear activation function

### Multi-Norm Representations

The paper also explores combining multiple norms to capture different aspects of the data:

$$\mathbf{h}_v^{(t+1)} = \|_{p=1,2,\infty} \sigma\left(\frac{\mathbf{W}_t \cdot \mathbf{h}_v^{(t)}}{\|\mathbf{h}_v^{(t)}\|_p + \epsilon}\right)$$

Where $\|$ denotes concatenation of the feature representations computed using different norms.

## Implementation Details

### Norm Computation

For practical implementation, the norms are computed efficiently:

1. **L1 Norm**:
   ```python
   def l1_norm(h):
       return np.sum(np.abs(h))
   ```

2. **L2 Norm**:
   ```python
   def l2_norm(h):
       return np.sqrt(np.sum(h**2))
   ```

3. **Max Norm**:
   ```python
   def max_norm(h):
       return np.max(np.abs(h))
   ```

### Normalization with Different Norms

The paper implements normalization using different norms to evaluate their effectiveness:

```python
# L1 normalization
h_normalized_l1 = h / (l1_norm(h) + epsilon)

# L2 normalization
h_normalized_l2 = h / (l2_norm(h) + epsilon)

# Max normalization
h_normalized_max = h / (max_norm(h) + epsilon)
```

### Application to Edge Features

The norm-based representation is also applied to edge features:

$$\mathbf{e}_{uv}^{\text{norm}} = \frac{\mathbf{e}_{uv}}{\|\mathbf{e}_{uv}\|_p + \epsilon}$$

## Theoretical Justification

The authors provide several theoretical justifications for norm-based representations:

1. **Robustness to Noise**: Normalization reduces the impact of outliers or noisy data points, making the GNN more robust.

2. **Scale Invariance**: Norm-based features are invariant to the scaling of input features, ensuring consistent learning across different scales.

3. **Improved Gradient Flow**: Normalization helps prevent vanishing or exploding gradients during training, leading to more stable optimization.

4. **Enhanced Feature Discrimination**: By normalizing features, the model can better focus on the relative importance of different features rather than their absolute magnitudes.

## Experimental Findings

Interestingly, the paper's empirical results showed that MinMax scaling outperformed the proposed L2-norm approach on their specific dataset:

| Metric | MinMax | L2-norm |
|--------|--------|---------|
| Accuracy | 96.24% | 57.43% |
| MCC | 0.9433 | 0.5046 |

This unexpected finding suggests that the effectiveness of norm-based representations may be dataset-dependent and requires careful consideration of data characteristics.

## Applications in Process Maps

The norm-based representation provides several benefits for process map analysis:

1. **Standardizing Task Features**: Normalizes diverse task attributes (duration, resource usage, priority) to a common scale.

2. **Balancing Dependency Strengths**: Prevents strong dependencies from dominating the message passing process.

3. **Handling Missing Data**: Provides stability when dealing with incomplete task or dependency information.

4. **Cross-Process Comparability**: Enables meaningful comparisons between different processes or workflows.

5. **Adaptability to Dynamic Changes**: Maintains representational stability even as task attributes change over time.

## Source Identification

This building block draws from several established areas of research:

1. **Normalization Techniques in Neural Networks**: Concepts like batch normalization, layer normalization
   - Source: Ioffe & Szegedy (2015), Ba et al. (2016)
   - Not explicitly cited but incorporated in the approach

2. **Feature Scaling in Machine Learning**: Various approaches to feature normalization
   - Citation: [9] Lee et al. (2020) for calibration in GNNs

3. **GNN Regularization**: Methods to prevent over-smoothing and over-fitting
   - Citation: [17] You et al. (2020)

4. **L2-Normalization in Embeddings**: Use of normalized embeddings in representation learning
   - Citation: [20] Zhou et al. (2020)

The paper's approach represents a novel application of these normalization concepts specifically tailored to process map optimization, with the unexpected empirical finding that traditional MinMax scaling outperformed the proposed L2-norm approach on their dataset.