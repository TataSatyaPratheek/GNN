# Building Block: Multi-head Attention

## Conceptual Overview

Multi-head attention is a sophisticated building block in the paper's GNN framework that enables the model to focus on different aspects of task dependencies simultaneously. By employing multiple attention mechanisms in parallel, the GNN can capture diverse relationship patterns within process maps, such as identifying bottlenecks, prioritizing critical tasks, and managing resource constraints.

## Mathematical Foundation

### Basic Attention Mechanism

At its core, the attention mechanism computes a weighted sum of values, where the weights (attention coefficients) are determined by a compatibility function between a query and a key:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$ is the query matrix
- $K$ is the key matrix
- $V$ is the value matrix
- $d_k$ is the dimension of the keys (scaling factor)

### Graph Attention in the Paper

For process maps, the paper adapts this to graph-structured data. For a node $v$, the attention coefficient for its neighbor $u$ is computed as:

$$\alpha_{vu} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T \cdot [\mathbf{W}_q\mathbf{h}_v||\mathbf{W}_k\mathbf{h}_u]\right)\right)}{\sum_{w\in\mathcal{N}_v}\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T \cdot [\mathbf{W}_q\mathbf{h}_v||\mathbf{W}_k\mathbf{h}_w]\right)\right)}$$

Where:
- $\mathbf{a}$ is a learnable attention vector
- $\mathbf{W}_q$ and $\mathbf{W}_k$ are learnable weight matrices for query and key transformations
- $\mathbf{h}_v$ and $\mathbf{h}_u$ are the feature vectors of nodes $v$ and $u$
- $||$ denotes vector concatenation
- LeakyReLU is the activation function with a small negative slope

### Multi-head Attention

To capture different types of dependencies, the paper employs multi-head attention, where $K$ separate attention mechanisms are computed in parallel:

For each attention head $k$ (where $k \in \{1, 2, ..., K\}$), the representation of node $v$ at layer $t+1$ is:

$$\mathbf{h}_v^{(t+1,k)} = \sigma\left(\mathbf{W}_t^k \cdot \sum_{u \in \mathcal{N}_v} \alpha_{vu}^k \cdot \mathbf{h}_u^{(t)}\right)$$

Where the attention coefficient $\alpha_{vu}^k$ for head $k$ is:

$$\alpha_{vu}^k = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}_k^T \cdot [\mathbf{W}_q^k\mathbf{h}_v^{(t)}||\mathbf{W}_k^k\mathbf{h}_u^{(t)}]\right)\right)}{\sum_{w\in\mathcal{N}_v}\exp\left(\text{LeakyReLU}\left(\mathbf{a}_k^T \cdot [\mathbf{W}_q^k\mathbf{h}_v^{(t)}||\mathbf{W}_k^k\mathbf{h}_w^{(t)}]\right)\right)}$$

The final multi-head representation combines outputs from all heads:

$$\mathbf{h}_v^{(t+1)} = \|_{k=1}^K \mathbf{h}_v^{(t+1,k)}$$

Where $\|$ denotes vector concatenation. Alternatively, averaging can be used:

$$\mathbf{h}_v^{(t+1)} = \frac{1}{K} \sum_{k=1}^K \mathbf{h}_v^{(t+1,k)}$$

### Edge-Enhanced Attention

The paper also incorporates edge features into the attention mechanism:

$$\alpha_{vu}^k = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}_k^T \cdot [\mathbf{W}_q^k\mathbf{h}_v^{(t)}||\mathbf{W}_k^k\mathbf{h}_u^{(t)}||\mathbf{W}_e^k\mathbf{e}_{vu}]\right)\right)}{\sum_{w\in\mathcal{N}_v}\exp\left(\text{LeakyReLU}\left(\mathbf{a}_k^T \cdot [\mathbf{W}_q^k\mathbf{h}_v^{(t)}||\mathbf{W}_k^k\mathbf{h}_w^{(t)}||\mathbf{W}_e^k\mathbf{e}_{vw}]\right)\right)}$$

Where $\mathbf{e}_{vu}$ is the feature vector of the edge connecting nodes $v$ and $u$, and $\mathbf{W}_e^k$ is a learnable weight matrix for edge feature transformation.

## Implementation Details

### Attention Head Configuration

The paper experiments with different numbers of attention heads $(K)$ and dimensions per head $(d_h)$:
- $K=8$ heads with $d_h=32$ dimensions each
- The total output dimension is $K \times d_h = 256$

### Attention Dropout

To prevent overfitting and promote diversity among attention heads, attention dropout is applied:

$$\hat{\alpha}_{vu}^k = \text{Dropout}(\alpha_{vu}^k, p)$$

Where $p$ is the dropout probability (typically 0.1-0.3).

### Head Diversity Regularization

To encourage different heads to focus on different aspects, a diversity loss is introduced:

$$\mathcal{L}_{\text{diversity}} = \sum_{i=1}^K \sum_{j=i+1}^K \|\mathbf{h}^{(i)} - \mathbf{h}^{(j)}\|^2$$

This loss term pushes the heads to learn distinct patterns in the process map.

## Application to Process Maps

The multi-head attention mechanism provides several key capabilities for process map analysis:

### 1. Task Prioritization

Different attention heads can focus on different aspects of task importance:
- Head 1: Tasks with tight deadlines
- Head 2: Tasks with high resource requirements
- Head 3: Tasks on the critical path
- Head 4: Tasks with high failure rates

### 2. Bottleneck Detection

Attention patterns reveal bottlenecks in the workflow:
- High attention weights to specific tasks indicate potential bottlenecks
- Consistent attention patterns across multiple process instances highlight structural bottlenecks

### 3. Resource Allocation

Attention heads can specialize in different resource considerations:
- Head 1: Personnel allocation
- Head 2: Equipment usage
- Head 3: Material requirements
- Head 4: Spatial constraints

### 4. Temporal Dependencies

Multiple heads capture different temporal aspects:
- Head 1: Short-term dependencies (immediate next tasks)
- Head 2: Medium-term dependencies (tasks within the same phase)
- Head 3: Long-term dependencies (cross-phase relationships)

### Practical Example

In a manufacturing workflow with tasks A, B, and C, where A depends on both B and C:
- Head 1 might assign higher attention to B due to its shorter delay time
- Head 2 might prioritize C based on its higher resource utilization
- The final representation for A aggregates these diverse perspectives, enabling balanced decision-making

## Challenges and Mitigations

### Attention Head Collapse

Multiple heads may converge to similar patterns, reducing the benefit of the multi-head approach. The paper addresses this through:
1. The diversity regularization loss mentioned above
2. Different initialization for each head
3. Head-specific dropout patterns

### Computational Overhead

Multi-head attention increases computation and memory requirements. The paper mitigates this through:
1. Efficient implementation using parallel tensor operations
2. Reduced dimension per head as the number of heads increases
3. Selective attention computation for large graphs (focusing on high-weight edges)

## Source Identification

This building block draws from several established areas of research:

1. **Attention Mechanisms in Neural Networks**: Original concept from sequence modeling
   - Source: Bahdanau et al. (2015), "Neural Machine Translation by Jointly Learning to Align and Translate"
   - Not explicitly cited but forms the theoretical foundation

2. **Transformer Architecture**: Multi-head attention in sequence models
   - Source: Vaswani et al. (2017), "Attention Is All You Need"
   - Not explicitly cited but referenced in the paper title ("Process Is All You Need")

3. **Graph Attention Networks**: Application of attention to graph-structured data
   - Citation: [13] Veličković et al. (2018), "Graph Attention Networks"

4. **Graph Transformers**: Combining transformer-style attention with graph structures
   - Source: Yun et al. (2019), "Graph Transformer Networks"
   - Not explicitly cited but conceptually relevant

The paper's multi-head attention mechanism represents a specialized adaptation of these general concepts to the domain of process maps, with particular emphasis on capturing diverse task dependencies, identifying bottlenecks, and optimizing resource allocation in workflows.