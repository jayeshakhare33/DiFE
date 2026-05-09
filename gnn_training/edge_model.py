"""
GraphSAGE edge-classification model for transaction fraud detection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv


class GraphSAGEEdgeClassifier(nn.Module):
    """Encode user nodes with GraphSAGE and score transaction edges."""

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        aggregator_type: str = "mean",
        mlp_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggregator_type = aggregator_type
        self.mlp_hidden_dim = mlp_hidden_dim or hidden_dim

        self.input_proj = nn.Linear(node_in_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [
                SAGEConv(hidden_dim, hidden_dim, aggregator_type=aggregator_type)
                for _ in range(num_layers)
            ]
        )

        edge_repr_dim = hidden_dim * 4 + edge_in_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_repr_dim, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.mlp_hidden_dim, max(self.mlp_hidden_dim // 2, 32)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(self.mlp_hidden_dim // 2, 32), 1),
        )

    def encode_nodes(self, graph, node_features: torch.Tensor) -> torch.Tensor:
        """Run GraphSAGE over the user graph and return node embeddings."""
        h = self.input_proj(node_features)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        for conv in self.convs:
            h = conv(graph, h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        return h

    def score_edges(
        self,
        node_embeddings: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Build edge representations and return fraud logits."""
        src_h = node_embeddings[edge_src]
        dst_h = node_embeddings[edge_dst]

        edge_inputs = [src_h, dst_h, torch.abs(src_h - dst_h), src_h * dst_h]
        if edge_features.numel() > 0:
            edge_inputs.append(edge_features)

        edge_repr = torch.cat(edge_inputs, dim=1)
        return self.edge_mlp(edge_repr).squeeze(-1)

    def forward(
        self,
        graph,
        node_features: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        node_embeddings = self.encode_nodes(graph, node_features)
        return self.score_edges(node_embeddings, edge_src, edge_dst, edge_features)
