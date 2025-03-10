# Building Block: Custom Loss Functions

## Conceptual Overview

Custom loss functions represent a critical building block in the paper's framework, enabling the GNN to optimize for process-specific objectives rather than generic graph tasks. These specialized loss functions align the model's learning with practical goals in process optimization, such as minimizing delays, optimizing resource allocation, and improving overall workflow efficiency.

## Mathematical Foundation

### General Loss Framework

The paper proposes a comprehensive loss function that combines task-level, workflow-level, and regularization objectives:

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \mathcal{L}_{\text{workflow}} + \beta\mathcal{L}_{\text{regularization}}$$

Where $\beta$ is a hyperparameter balancing the contribution of the regularization term.

### Task-Level Losses

Task-level losses focus on optimizing individual task performance within the process map.

#### Delay Loss

The delay loss penalizes deviations from target task completion times:

$$\mathcal{L}_{\text{delay}} = \sum_{v \in V} \left(T_v - T_v^{\text{target}}\right)^2$$

Where:
- $T_v$ is the actual completion time of task $v$
- $T_v^{\text{target}}$ is the target completion time

#### Resource Utilization Loss

This loss penalizes inefficient resource allocation:

$$\mathcal{L}_{\text{resource}} = \sum_{v \in V} \left(R_v - R_v^{\text{optimal}}\right)^2$$

Where:
- $R_v$ is the actual resource usage for task $v$
- $R_v^{\text{optimal}}$ is the optimal resource allocation

#### Next-Activity Classification Loss

For predicting the next activity in a workflow, a cross-entropy loss is used:

$$\mathcal{L}_{\text{next-activity}} = -\sum_{v \in V} \sum_{j=1}^C y_{v,j} \log(\hat{y}_{v,j})$$

Where:
- $y_{v,j}$ is the ground truth (1 if the next activity is class $j$, 0 otherwise)
- $\hat{y}_{v,j}$ is the predicted probability for class $j$
- $C$ is the total number of activity classes

### Workflow-Level Losses

Workflow-level losses address overall process performance and efficiency.

#### Critical Path Loss

This loss promotes accurate embedding of tasks along the critical path:

$$\mathcal{L}_{\text{critical-path}} = \sum_{(u,v) \in P} \|\mathbf{h}_u - \mathbf{h}_v\|^2$$

Where:
- $P$ represents the set of edges along the critical path
- $\mathbf{h}_u$ and $\mathbf{h}_v$ are the embeddings of nodes $u$ and $v$

#### Cycle Time Loss

This loss focuses on optimizing the end-to-end process duration:

$$\mathcal{L}_{\text{cycle-time}} = \|\hat{T}_{\text{cycle}} - T_{\text{cycle}}^{\text{target}}\|^2$$

Where:
- $\hat{T}_{\text{cycle}}$ is the predicted cycle time
- $T_{\text{cycle}}^{\text{target}}$ is the target cycle time

#### Bottleneck Detection Loss

This loss encourages the model to identify and highlight bottlenecks:

$$\mathcal{L}_{\text{bottleneck}} = -\sum_{v \in V_{\text{bottleneck}}} \log(s_v)$$

Where:
- $V_{\text{bottleneck}}$ is the set of known bottleneck tasks
- $s_v$ is the bottleneck score assigned by the model to task $v$

### Regularization Losses

Regularization losses ensure model stability and generalization.

#### Laplacian Regularization

This promotes smoothness in the embeddings of connected nodes:

$$\mathcal{L}_{\text{regularization}} = \sum_{(u,v) \in E} w_{uv} \|\mathbf{h}_u - \mathbf{h}_v\|^2$$

Where:
- $w_{uv}$ is the weight of the edge connecting nodes $u$ and $v$
- $\mathbf{h}_u$ and $\mathbf{h}_v$ are the embeddings of the nodes

#### Head Diversity Loss

For multi-head attention, this loss encourages diversity among attention heads:

$$\mathcal{L}_{\text{diversity}} = \sum_{i=1}^K \sum_{j=i+1}^K \|\mathbf{h}^{(i)} - \mathbf{h}^{(j)}\|^2$$

Where $\mathbf{h}^{(i)}$ and $\mathbf{h}^{(j)}$ are embeddings from different attention heads.

## Implementation Details

### Loss Combination Strategies

The paper explores different strategies for combining the individual loss components:

1. **Weighted Sum**: Each loss component is assigned a weight based on its importance:
   $$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{delay}} + \lambda_2 \mathcal{L}_{\text{resource}} + \lambda_3 \mathcal{L}_{\text{critical-path}} + \ldots$$

2. **Multi-Objective Optimization**: The model alternates between optimizing different objectives in a round-robin fashion.

3. **Adaptive Weighting**: The weights of different loss components are adjusted dynamically based on their current values:
   $$\lambda_i = \frac{\exp(-\alpha \mathcal{L}_i)}{\sum_j \exp(-\alpha \mathcal{L}_j)}$$

### Practical Implementation

For practical implementation, the loss functions are computed efficiently:

```python
def compute_loss(model, batch, targets):
    # Task-level losses
    delay_loss = torch.mean((model.predict_times(batch) - targets.times) ** 2)
    next_activity_loss = F.cross_entropy(model.predict_next(batch), targets.next_activity)
    
    # Workflow-level losses
    critical_path_edges = get_critical_path(batch)
    critical_path_loss = compute_edge_embedding_loss(model, critical_path_edges)
    
    # Regularization
    reg_loss = compute_laplacian_regularization(model, batch)
    
    # Combined loss
    total_loss = delay_loss + next_activity_loss + critical_path_loss + beta * reg_loss
    
    return total_loss
```

## Application to Process Maps

The custom loss functions enable several key capabilities for process map optimization:

### 1. Process-Specific Objectives

By incorporating domain knowledge into the loss function, the model optimizes for metrics that matter in the specific process context:
- Manufacturing: Minimize production time while maintaining quality
- Logistics: Optimize delivery schedules while reducing costs
- Customer Service: Balance response time with resolution quality

### 2. Multi-Criteria Optimization

The combined loss allows optimization of multiple, potentially competing criteria:
- Minimize task delays while efficiently utilizing resources
- Reduce cycle time while maintaining process quality
- Optimize critical paths while supporting process adaptability

### 3. Bottleneck Identification and Mitigation

The bottleneck detection loss enables the model to identify and address workflow constraints:
- Highlight tasks that consistently delay the overall process
- Identify resource allocation patterns that create bottlenecks
- Detect structural inefficiencies in the process design

### Practical Example

In the paper's experiments, the custom loss functions led to concrete improvements:
- 97.4% accuracy in next-activity prediction
- Identification of high-delay transitions for immediate operational gains
- Improved resource allocation through joint optimization

## Challenges and Mitigations

### Balancing Multiple Objectives

Combining multiple loss terms can lead to conflicting gradients and training instability. The paper addresses this through:
1. Careful weight tuning based on the relative importance of each objective
2. Normalization of loss components to ensure comparable scales
3. Adaptive weighting that adjusts priorities during training

### Overfitting Prevention

Complex loss functions with many components can lead to overfitting. Mitigations include:
1. The regularization term ($\beta\mathcal{L}_{\text{regularization}}$)
2. Early stopping based on validation performance
3. Dropout and weight decay during training

## Source Identification

This building block draws from several established areas of research:

1. **Multi-Task Learning**: Frameworks for optimizing multiple objectives simultaneously
   - Source: Caruana (1997), "Multitask Learning"
   - Not explicitly cited but conceptually relevant

2. **Graph Laplacian Regularization**: Smoothing techniques for graph-structured data
   - Source: Various works on spectral graph theory
   - Citation: [7] Duong et al. (2023), implicitly referencing these concepts

3. **Process Mining Metrics**: Domain-specific objectives for process optimization
   - Citations: [6] Doe & Smith (2023), [12] Sommers & Nguyen (2021)

4. **Reinforcement Learning Rewards**: Objective functions for sequential decision-making
   - Citations: [3] Chen et al. (2021), [1] Abbasi et al. (2024)

The paper's approach represents a novel integration of these diverse loss components specifically tailored to process map optimization, aligning the GNN's learning with practical goals in workflow management and providing a flexible framework for addressing domain-specific requirements.