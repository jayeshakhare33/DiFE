import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, Linear


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # Create a HeteroConv with relation-specific transformations
        # Each edge type gets its own linear transformation (RGCN-style)
        conv_dict = {}
        for etype in etypes:
            # Use Linear layer for each relation type
            # The Linear layer will be applied to source node features
            conv_dict[etype] = Linear(in_size, out_size, bias=False)
        self.conv = HeteroConv(conv_dict, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        # Apply relation-specific transformations and aggregate
        # HeteroConv expects x_dict (dict of node_type -> features) and edge_index_dict (dict of edge_type -> edge_index)
        out_dict = self.conv(x_dict, edge_index_dict)
        return out_dict


class HeteroRGCN(nn.Module):
    def __init__(self, ntype_dict, etypes, in_size, hidden_size, out_size, n_layers, embedding_size):
        super(HeteroRGCN, self).__init__()
        # Use trainable node embeddings as featureless inputs.
        embed_dict = {ntype: nn.Parameter(torch.Tensor(num_nodes, in_size))
                      for ntype, num_nodes in ntype_dict.items() if ntype != 'target'}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)
        
        # create layers
        self.layers = nn.ModuleList()
        self.layers.append(HeteroRGCNLayer(embedding_size, hidden_size, etypes))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(HeteroRGCNLayer(hidden_size, hidden_size, etypes))

        # output layer
        self.layers.append(nn.Linear(hidden_size, out_size))

    def forward(self, data, features):
        # data is a HeteroData object from PyG
        # get embeddings for all node types. for target node type, use passed in features
        x_dict = {ntype: emb for ntype, emb in self.embed.items()}
        x_dict['target'] = features

        # Get edge_index_dict from HeteroData
        edge_index_dict = {}
        for edge_type in data.edge_types:
            edge_index_dict[edge_type] = data[edge_type].edge_index

        # pass through all layers
        for i, layer in enumerate(self.layers[:-1]):
            if i != 0:
                x_dict = {k: F.leaky_relu(h) for k, h in x_dict.items()}
            x_dict = layer(x_dict, edge_index_dict)

        # get target node logits
        return self.layers[-1](x_dict['target'])