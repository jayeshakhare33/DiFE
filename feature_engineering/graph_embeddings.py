"""
Graph Embedding Extraction
Extracts node embeddings from trained GNN models
"""

import torch as th
import numpy as np
import pandas as pd
import dgl
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class GraphEmbeddings:
    """Extracts and manages graph embeddings"""
    
    def __init__(self):
        """Initialize graph embeddings extractor"""
        logger.info("Initialized GraphEmbeddings")
    
    def extract_embeddings(self, model: th.nn.Module, g: dgl.DGLHeteroGraph, 
                          features: th.Tensor, node_type: str = 'target') -> np.ndarray:
        """
        Extract node embeddings from trained model
        
        Args:
            model: Trained GNN model
            g: DGL heterogeneous graph
            features: Node features tensor
            node_type: Node type to extract embeddings for
            
        Returns:
            Node embeddings array
        """
        logger.info(f"Extracting embeddings for node type: {node_type}")
        
        model.eval()
        with th.no_grad():
            # Get embeddings from model
            if hasattr(model, 'embed'):
                # Get embeddings from embedding layer
                if node_type in model.embed:
                    embeddings = model.embed[node_type].detach().cpu().numpy()
                else:
                    # Extract from intermediate layers
                    h_dict = {ntype: emb for ntype, emb in model.embed.items()}
                    h_dict['target'] = features
                    
                    # Pass through layers to get embeddings
                    for i, layer in enumerate(model.layers[:-1]):
                        h_dict = layer(g, h_dict)
                    
                    embeddings = h_dict[node_type].detach().cpu().numpy()
            else:
                # Extract from forward pass
                output = model(g, features)
                # Use last hidden layer
                embeddings = output.detach().cpu().numpy()
        
        logger.info(f"Extracted embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def extract_all_embeddings(self, model: th.nn.Module, g: dgl.DGLHeteroGraph,
                              features: th.Tensor) -> Dict[str, np.ndarray]:
        """
        Extract embeddings for all node types
        
        Args:
            model: Trained GNN model
            g: DGL heterogeneous graph
            features: Node features tensor
            
        Returns:
            Dictionary of embeddings by node type
        """
        logger.info("Extracting embeddings for all node types")
        
        embeddings = {}
        
        if hasattr(model, 'embed'):
            for ntype in model.embed.keys():
                embeddings[ntype] = model.embed[ntype].detach().cpu().numpy()
        
        # Extract target embeddings
        model.eval()
        with th.no_grad():
            h_dict = {ntype: emb for ntype, emb in model.embed.items()} if hasattr(model, 'embed') else {}
            h_dict['target'] = features
            
            # Pass through layers
            for i, layer in enumerate(model.layers[:-1]):
                h_dict = layer(g, h_dict)
            
            embeddings['target'] = h_dict['target'].detach().cpu().numpy()
        
        logger.info(f"Extracted embeddings for {len(embeddings)} node types")
        return embeddings
    
    def save_embeddings(self, embeddings: Dict[str, np.ndarray], output_dir: str):
        """
        Save embeddings to files
        
        Args:
            embeddings: Dictionary of embeddings by node type
            output_dir: Output directory
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for ntype, emb in embeddings.items():
            output_path = os.path.join(output_dir, f'{ntype}_embeddings.npy')
            np.save(output_path, emb)
            logger.info(f"Saved {ntype} embeddings to {output_path}")
    
    def load_embeddings(self, input_dir: str, node_type: str) -> np.ndarray:
        """
        Load embeddings from file
        
        Args:
            input_dir: Input directory
            node_type: Node type
            
        Returns:
            Embeddings array
        """
        import os
        input_path = os.path.join(input_dir, f'{node_type}_embeddings.npy')
        embeddings = np.load(input_path)
        logger.info(f"Loaded {node_type} embeddings with shape: {embeddings.shape}")
        return embeddings














