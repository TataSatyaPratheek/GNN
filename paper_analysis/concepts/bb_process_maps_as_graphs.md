# Building Block: Process Maps as Graph Structures

## Conceptual Overview

The foundational building block of the paper is the representation of process maps as graph structures. This representation provides the mathematical framework upon which the entire GNN-based approach is built.

## Formal Definition

A process map is modeled as a directed graph $G = (V, E)$, where:
- $V$ is the set of nodes representing tasks in the workflow
- $E \subseteq V \times V$ is the set of directed edges representing dependencies between tasks

## Node Features

Each node $v \in V$ corresponds to a task within the process map and is associated with a feature vector $\mathbf{x}_v \in \mathbb{R}^d$, where $d$ is the feature dimensionality. These features encode task-specific attributes:

$$\mathbf{x}_v = [t_v, r_v, \tau_v, s_v, \ldots]$$

Where:
- $t_v$: Task duration (estimated or actual time required for completion)
- $r_v$: Resource allocation (personnel, machines, materials)
- $\tau_v$: Categorical encoding of task type (e.g., "assembly," "quality control")
- $s_v$: Task priority score (importance or urgency)

## Edge Features

Each edge $e_{ij} \in E$ represents a dependency or transition between tasks and is associated with a feature vector $\mathbf{e}_{ij} \in \mathbb{R}^k$, where $k$ is the edge feature dimensionality:

$$\mathbf{e}_{ij} = [p_{ij}, w_{ij}, c_{ij}, \delta_{ij}, \ldots]$$

Where:
- $p_{ij}$: Transition probability (likelihood of moving from task $i$ to task $j$)
- $w_{ij}$: Dependency strength (degree of influence task $i$ exerts on task $j$)
- $c_{ij}$: Cost associated with the dependency (e.g., material transfer costs)
- $\delta_{ij}$: Sequencing constraint (temporal restrictions on task ordering)

## Adjacency Matrix Representation

The connectivity of the graph is encoded using an adjacency matrix $A \in \mathbb{R}^{|V| \times |V|}$:

$$A_{ij} = 
\begin{cases} 
1, & \text{if there exists a directed edge } (v_i, v_j) \in E \\
0, & \text{otherwise}
\end{cases}$$

For weighted graphs, the adjacency matrix can incorporate edge weights:

$$A_{ij} = w_{ij}$$

## Graph Normalization

To ensure numerical stability during training, the adjacency matrix is often normalized using symmetric normalization:

$$\hat{A} = D^{-1/2}AD^{-1/2}$$

Where $D$ is the degree matrix with entries $D_{ii} = \sum_j A_{ij}$.

## Dynamic Integration

The graph representation supports dynamic updates to capture real-time changes in the workflow:
- Node features can include real-time task progress ($\pi_v$) or resource utilization ($\rho_v$)
- Edge features can incorporate live updates on transportation delays ($\Delta_{ij}$) or bottlenecks

## Applications in the Paper

The authors leverage this graph representation for several key applications:
1. **Learning embeddings** for tasks and dependencies through message passing
2. **Identifying bottlenecks** by analyzing node connectivity and edge weights
3. **Optimizing resource allocation** by modeling resource constraints as node features
4. **Predicting next activities** based on graph structure and learned representations
5. **Detecting process anomalies** by identifying unusual patterns in the graph

## Advantages Over Traditional Representations

The graph representation offers several advantages over traditional approaches:
1. **Captures complex dependencies** beyond simple sequential models
2. **Naturally represents concurrent tasks** and their interdependencies
3. **Scales effectively** to large-scale workflows with thousands of tasks
4. **Accommodates dynamic changes** in real-time through graph updates
5. **Supports heterogeneous features** for both tasks and dependencies

## Source Identification

This building block draws from several established areas of research:

1. **Graph Theory**: The fundamental mathematical representation of process maps as directed graphs
   - Source: Graph theory principles dating back to Euler (1736)
   - Citation: [20] Zhou et al. (2020) for graph representations in modern GNNs

2. **Process Mining**: The extraction and representation of processes from event logs
   - Sources: van der Aalst's work on process mining (early 2000s)
   - Citations: [6] Doe & Smith (2023), [7] Duong et al. (2023)

3. **Business Process Modeling**: Domain-specific knowledge about workflow representation
   - Source: Business Process Model and Notation (BPMN) standards
   - Citations: Not explicitly cited but implied in domain knowledge

4. **Feature Engineering for Processes**: Techniques for extracting and representing task attributes
   - Citations: [12] Sommers & Nguyen (2021), [14] Wasi et al. (2024)

The paper builds upon these foundations to create a comprehensive graph-based representation of process maps that serves as the input to their GNN framework.