"""
GNN Training Module
Distributed GNN training using DGL and PyTorch Distributed
"""

from .distributed_trainer import DistributedTrainer
from .gnn_model import HeteroRGCN

__all__ = ['DistributedTrainer', 'HeteroRGCN']











