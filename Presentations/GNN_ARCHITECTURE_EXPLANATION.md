# GNN Architecture Explanation

## Overview
This model implements a **Heterogeneous Relational Graph Convolutional Network (HeteroRGCN)** for fraud detection. It's designed to work with heterogeneous graphs containing multiple node types and edge types (relations).

## Architecture Components

### 1. Graph Structure (Heterogeneous Graph)

The model operates on a **heterogeneous graph** with:
- **Multiple Node Types**: Different types of entities (e.g., `target` nodes for transactions, and other auxiliary node types)
- **Multiple Edge Types**: Different types of relationships between nodes (e.g., `target<>Card`, `target<>Address`, etc.)
- **Bidirectional Edges**: For each edge type, both forward and reverse edges are created
- **Self-loops**: Target nodes have self-connections (`self_relation`)

**Example Edge Types:**
- `(target, 'target<>Card', Card)`
- `(Card, 'Card<>target', target)`
- `(target, 'self_relation', target)`

### 2. Model Architecture: HeteroRGCN

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Layer                               │
├─────────────────────────────────────────────────────────────┤
│  Target Nodes: Use actual features (e.g., transaction data) │
│  Other Nodes: Use trainable embeddings (learned parameters)  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              HeteroRGCN Layer 1                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  For each edge type:                                  │  │
│  │    - Linear transformation (in_size → hidden_size)   │  │
│  │    - Relation-specific weight matrix                  │  │
│  │  Aggregation: Sum over all relation types             │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↓
                    LeakyReLU Activation
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              HeteroRGCN Layer 2 (Hidden Layer)              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Same structure: Relation-specific transformations    │  │
│  │  hidden_size → hidden_size                           │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↓
                    LeakyReLU Activation
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              HeteroRGCN Layer N (Additional Hidden Layers) │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              Output Layer (Linear)                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Linear(hidden_size → n_classes)                      │  │
│  │  Output: Logits for binary classification (Fraud/Not)│  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 3. Key Components Explained

#### A. Node Embeddings (`HeteroRGCN.__init__`)

**For Non-Target Nodes:**
- Uses **trainable embeddings** (learnable parameters)
- Initialized with Xavier uniform initialization
- Shape: `(num_nodes, embedding_size)` where `embedding_size=360` (default)
- These embeddings are learned during training to capture node characteristics

**For Target Nodes:**
- Uses **actual features** from the dataset (e.g., transaction features)
- Features are normalized (mean=0, std=1) before training
- Shape: `(num_target_nodes, num_features)`

#### B. HeteroRGCN Layer (`HeteroRGCNLayer`)

Each layer implements relation-specific transformations:

```python
# For each edge type (relation), create a separate Linear layer
conv_dict[etype] = Linear(in_size, out_size, bias=False)

# Use HeteroConv to aggregate across all relation types
self.conv = HeteroConv(conv_dict, aggr='sum')
```

**How it works:**
1. **Relation-Specific Transformation**: Each edge type gets its own weight matrix
   - This allows the model to learn different patterns for different relationships
   - Example: `target→Card` relationship learns different patterns than `target→Address`

2. **Message Passing**: For each node, the layer:
   - Transforms neighbor features using relation-specific weights
   - Aggregates messages from all neighbors (sum aggregation)
   - Updates node representations

3. **Heterogeneous Aggregation**: The `HeteroConv` handles:
   - Different node types
   - Different edge types
   - Proper message routing based on edge types

#### C. Forward Pass Flow

```python
def forward(self, data, features):
    # 1. Initialize node features
    x_dict = {ntype: emb for ntype, emb in self.embed.items()}  # Embeddings for non-target
    x_dict['target'] = features  # Actual features for target nodes
    
    # 2. Extract edge indices for all edge types
    edge_index_dict = {edge_type: data[edge_type].edge_index 
                       for edge_type in data.edge_types}
    
    # 3. Pass through GNN layers
    for i, layer in enumerate(self.layers[:-1]):
        if i != 0:  # Apply activation after first layer
            x_dict = {k: F.leaky_relu(h) for k, h in x_dict.items()}
        x_dict = layer(x_dict, edge_index_dict)  # Message passing
    
    # 4. Final classification (only for target nodes)
    return self.layers[-1](x_dict['target'])
```

### 4. Hyperparameters

From `estimator_fns.py`:
- **`n_hidden`**: Hidden dimension size (default: 16)
- **`n_layers`**: Number of GNN layers (default: 3)
- **`embedding_size`**: Embedding dimension for non-target nodes (default: 360)
- **`lr`**: Learning rate (default: 1e-2)
- **`weight_decay`**: L2 regularization (default: 5e-4)
- **`n_epochs`**: Training epochs (default: 700)

### 5. Why This Architecture?

#### Heterogeneous Graph:
- Real-world fraud detection involves multiple entity types (transactions, cards, addresses, devices, etc.)
- Different relationships carry different semantic meanings
- Allows modeling complex interactions between entities

#### Relation-Specific Transformations (RGCN-style):
- Each relation type learns its own transformation
- More expressive than using the same weights for all relations
- Better captures relation-specific patterns

#### Trainable Embeddings:
- Non-target nodes (like Card, Address) may not have rich features
- Learnable embeddings allow the model to discover useful representations
- Embeddings capture latent characteristics of these entities

#### Multi-Layer Architecture:
- Enables multi-hop reasoning (e.g., transaction → card → other transactions)
- Each layer aggregates information from neighbors at increasing distances
- LeakyReLU activation introduces non-linearity

### 6. Training Process

1. **Graph Construction**: Build heterogeneous graph from edge lists and node features
2. **Feature Normalization**: Normalize target node features (mean=0, std=1)
3. **Forward Pass**: Pass graph through GNN layers
4. **Loss Computation**: Cross-entropy loss for binary classification
5. **Backpropagation**: Update all parameters (GNN weights + node embeddings)
6. **Evaluation**: F1 score on test set

### 7. Key Design Choices

- **Sum Aggregation**: Simple but effective for fraud detection
- **No Dropout in GNN Layers**: Focus on learning from graph structure
- **LeakyReLU**: Prevents dying ReLU problem
- **Full Graph Training**: Uses entire graph (not mini-batching) for simplicity
- **Binary Classification**: Outputs logits for Fraud (1) vs. Not Fraud (0)

## Mathematical Formulation

For a node `v` of type `t` at layer `l+1`:

```
h_v^(l+1) = LeakyReLU(Σ_{r∈R} Σ_{u∈N_r(v)} W_r^(l) · h_u^(l))
```

Where:
- `R` = set of relation types
- `N_r(v)` = neighbors of `v` connected via relation `r`
- `W_r^(l)` = relation-specific weight matrix for relation `r` at layer `l`
- `h_u^(l)` = representation of neighbor `u` at layer `l`

This is the core message passing operation in the HeteroRGCN.

