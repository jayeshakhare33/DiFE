"""
Distributed GNN Training
Distributed training using DGL and PyTorch Distributed
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import dgl
import numpy as np
import time
import logging
from typing import Optional, Dict, Tuple
import copy

from .gnn_model import HeteroRGCN

logger = logging.getLogger(__name__)


class DistributedTrainer:
    """Distributed GNN trainer using DGL and PyTorch Distributed"""
    
    def __init__(self, world_size: int = 4, backend: str = 'gloo'):
        """
        Initialize distributed trainer
        
        Args:
            world_size: Number of processes
            backend: Distributed backend ('gloo' or 'nccl')
        """
        self.world_size = world_size
        self.backend = backend
        logger.info(f"Initialized DistributedTrainer with world_size: {world_size}, backend: {backend}")
    
    def setup_distributed(self, rank: int, world_size: int, master_addr: str = 'localhost',
                         master_port: str = '12355'):
        """
        Setup distributed environment
        
        Args:
            rank: Process rank
            world_size: Number of processes
            master_addr: Master address
            master_port: Master port
        """
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        
        # Initialize process group
        dist.init_process_group(
            backend=self.backend,
            rank=rank,
            world_size=world_size
        )
        
        logger.info(f"Process {rank} initialized in distributed group")
    
    def cleanup_distributed(self):
        """Cleanup distributed environment"""
        dist.destroy_process_group()
        logger.info("Distributed group destroyed")
    
    def train_worker(self, rank: int, world_size: int, g: dgl.DGLHeteroGraph,
                    features: torch.Tensor, labels: torch.Tensor, train_mask: torch.Tensor,
                    test_mask: torch.Tensor, model_config: Dict, training_config: Dict):
        """
        Training worker function
        
        Args:
            rank: Process rank
            world_size: Number of processes
            g: DGL heterogeneous graph
            features: Node features
            labels: Node labels
            train_mask: Training mask
            test_mask: Test mask
            model_config: Model configuration
            training_config: Training configuration
        """
        # Setup distributed
        self.setup_distributed(rank, world_size)
        
        # Get device
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Process {rank} using device: {device}")
        
        # Partition graph for this process
        # In a real distributed setup, each process would get a partition
        # For now, we'll use the full graph on each process (not ideal, but works)
        local_g = g.to(device)
        local_features = features.to(device)
        local_labels = labels.to(device)
        local_train_mask = train_mask.to(device)
        local_test_mask = test_mask.to(device)
        
        # Create model
        ntype_dict = {ntype: g.number_of_nodes(ntype) for ntype in g.ntypes}
        model = HeteroRGCN(
            ntype_dict=ntype_dict,
            etypes=g.canonical_etypes,
            in_size=model_config.get('in_size', features.shape[1]),
            hidden_size=model_config.get('hidden_size', 16),
            out_size=model_config.get('out_size', 2),
            n_layers=model_config.get('n_layers', 3),
            embedding_size=model_config.get('embedding_size', features.shape[1])
        )
        model = model.to(device)
        
        # Wrap model with DDP
        if world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        
        # Loss and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_config.get('lr', 0.01),
            weight_decay=training_config.get('weight_decay', 5e-4)
        )
        
        # Training loop
        n_epochs = training_config.get('n_epochs', 100)
        best_loss = float('inf')
        best_model = None
        
        logger.info(f"Process {rank} starting training for {n_epochs} epochs")
        
        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            if world_size > 1:
                output = model.module(local_g, local_features)
            else:
                output = model(local_g, local_features)
            
            # Compute loss on training nodes
            train_output = output[local_train_mask.bool()]
            train_labels = local_labels[local_train_mask.bool()].long()
            loss = criterion(train_output, train_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Evaluate
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    if world_size > 1:
                        output = model.module(local_g, local_features)
                    else:
                        output = model(local_g, local_features)
                    
                    test_output = output[local_test_mask.bool()]
                    test_labels = local_labels[local_test_mask.bool()].long()
                    test_loss = criterion(test_output, test_labels)
                    
                    # Calculate accuracy
                    pred = output.argmax(dim=1)
                    test_pred = pred[local_test_mask.bool()]
                    test_acc = (test_pred == test_labels).float().mean()
                    
                    logger.info(f"Process {rank}, Epoch {epoch}, Loss: {loss.item():.4f}, "
                              f"Test Loss: {test_loss.item():.4f}, Test Acc: {test_acc.item():.4f}")
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                if world_size > 1:
                    best_model = copy.deepcopy(model.module.state_dict())
                else:
                    best_model = copy.deepcopy(model.state_dict())
        
        # Save model
        if rank == 0:  # Only save on rank 0
            save_path = training_config.get('save_path', './model/best_model.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(best_model, save_path)
            logger.info(f"Saved best model to {save_path}")
        
        # Cleanup
        self.cleanup_distributed()
    
    def train(self, g: dgl.DGLHeteroGraph, features: torch.Tensor, labels: torch.Tensor,
              train_mask: torch.Tensor, test_mask: torch.Tensor, model_config: Dict,
              training_config: Dict):
        """
        Start distributed training
        
        Args:
            g: DGL heterogeneous graph
            features: Node features
            labels: Node labels
            train_mask: Training mask
            test_mask: Test mask
            model_config: Model configuration
            training_config: Training configuration
        """
        logger.info(f"Starting distributed training with {self.world_size} processes")
        
        if self.world_size == 1:
            # Single process training
            self.train_worker(0, 1, g, features, labels, train_mask, test_mask,
                            model_config, training_config)
        else:
            # Multi-process training
            mp.spawn(
                self.train_worker,
                args=(self.world_size, g, features, labels, train_mask, test_mask,
                      model_config, training_config),
                nprocs=self.world_size,
                join=True
            )
        
        logger.info("Distributed training completed")










