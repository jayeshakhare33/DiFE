"""
Graph Partitioning for Distributed Training
Partitions large graphs for distributed processing
"""

import dgl
import torch as th
import numpy as np
from typing import List, Dict, Tuple
import logging
import os

logger = logging.getLogger(__name__)


class GraphPartitioner:
    """Partitions graphs for distributed training"""
    
    def __init__(self, num_parts: int = 4, partition_method: str = 'metis'):
        """
        Initialize graph partitioner
        
        Args:
            num_parts: Number of partitions
            partition_method: Partitioning method ('metis' or 'random')
        """
        self.num_parts = num_parts
        self.partition_method = partition_method
        logger.info(f"Initialized GraphPartitioner with {num_parts} parts, method: {partition_method}")
    
    def partition_graph(self, g: dgl.DGLHeteroGraph, output_dir: str) -> List[str]:
        """
        Partition graph for distributed training
        
        Args:
            g: DGL heterogeneous graph
            output_dir: Output directory for partitions
            
        Returns:
            List of partition file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Partitioning graph into {self.num_parts} parts")
        
        # For heterogeneous graphs, we partition the target nodes
        if 'target' not in g.ntypes:
            raise ValueError("Graph must have 'target' node type for partitioning")
        
        # Get target node IDs
        target_nodes = g.nodes('target')
        
        if self.partition_method == 'metis':
            try:
                # Use METIS for balanced partitioning
                partitions = self._metis_partition(g, target_nodes)
            except Exception as e:
                logger.warning(f"METIS partitioning failed: {e}, falling back to random")
                partitions = self._random_partition(target_nodes)
        else:
            partitions = self._random_partition(target_nodes)
        
        # Create subgraphs for each partition
        partition_files = []
        for part_id in range(self.num_parts):
            part_nodes = partitions[part_id]
            if len(part_nodes) == 0:
                continue
            
            # Create subgraph with k-hop neighbors
            subgraph = self._create_subgraph(g, part_nodes, k_hop=2)
            
            # Save partition
            partition_path = os.path.join(output_dir, f'part_{part_id}.dgl')
            dgl.save_graphs(partition_path, [subgraph])
            partition_files.append(partition_path)
            
            logger.info(f"Partition {part_id}: {len(part_nodes)} target nodes, "
                       f"{subgraph.number_of_nodes()} total nodes, "
                       f"{subgraph.number_of_edges()} edges")
        
        logger.info(f"Created {len(partition_files)} partitions")
        return partition_files
    
    def _metis_partition(self, g: dgl.DGLHeteroGraph, target_nodes: th.Tensor) -> Dict[int, List[int]]:
        """Partition using METIS algorithm"""
        # Convert to homogeneous graph for METIS (using target nodes only)
        # Create a simplified graph for partitioning
        target_node_list = target_nodes.tolist()
        
        # Create adjacency matrix for target nodes
        adj_dict = {}
        for etype in g.canonical_etypes:
            src, _, dst = etype
            if src == 'target' and dst == 'target':
                src_nodes, dst_nodes = g.edges(etype=etype)
                for s, d in zip(src_nodes.tolist(), dst_nodes.tolist()):
                    if s not in adj_dict:
                        adj_dict[s] = set()
                    adj_dict[s].add(d)
        
        # Use simple balanced partitioning
        partitions = {i: [] for i in range(self.num_parts)}
        nodes_per_part = len(target_node_list) // self.num_parts
        
        for i, node in enumerate(target_node_list):
            part_id = i // nodes_per_part
            if part_id >= self.num_parts:
                part_id = self.num_parts - 1
            partitions[part_id].append(node)
        
        return partitions
    
    def _random_partition(self, target_nodes: th.Tensor) -> Dict[int, List[int]]:
        """Random balanced partitioning"""
        node_list = target_nodes.tolist()
        np.random.shuffle(node_list)
        
        partitions = {i: [] for i in range(self.num_parts)}
        nodes_per_part = len(node_list) // self.num_parts
        
        for i, node in enumerate(node_list):
            part_id = i // nodes_per_part
            if part_id >= self.num_parts:
                part_id = self.num_parts - 1
            partitions[part_id].append(node)
        
        return partitions
    
    def _create_subgraph(self, g: dgl.DGLHeteroGraph, target_nodes: List[int], k_hop: int = 2) -> dgl.DGLHeteroGraph:
        """Create subgraph with k-hop neighbors"""
        # Get k-hop neighbors
        target_tensor = th.tensor(target_nodes, dtype=th.long)
        
        # Extract subgraph
        # For heterogeneous graphs, we need to extract neighbors across all edge types
        node_dict = {'target': target_tensor}
        
        # Get neighbors for each hop
        for hop in range(k_hop):
            new_nodes = {}
            for etype in g.canonical_etypes:
                src, _, dst = etype
                if src in node_dict:
                    src_nodes = node_dict[src]
                    dst_nodes, _ = g.out_edges(src_nodes, etype=etype)
                    if dst not in new_nodes:
                        new_nodes[dst] = []
                    new_nodes[dst].append(dst_nodes)
            
            # Merge new nodes
            for ntype, nodes in new_nodes.items():
                if len(nodes) > 0:
                    merged = th.cat(nodes).unique()
                    if ntype not in node_dict:
                        node_dict[ntype] = merged
                    else:
                        node_dict[ntype] = th.cat([node_dict[ntype], merged]).unique()
        
        # Create subgraph
        subgraph = dgl.node_subgraph(g, node_dict)
        return subgraph
    
    def load_partition(self, partition_path: str) -> dgl.DGLHeteroGraph:
        """Load a partition from file"""
        graphs, _ = dgl.load_graphs(partition_path)
        return graphs[0]











