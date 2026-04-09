import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)

import torch as th
import numpy as np
import pandas as pd
import time
import pickle
import copy

try:
    import torch_geometric
    print('PyTorch Geometric version: {}'.format(torch_geometric.__version__))
except ImportError:
    print("Warning: torch_geometric not found. Please install with: pip install torch-geometric")

from sklearn.metrics import confusion_matrix
from gnn.estimator_fns import *
from gnn.graph_utils import *
from gnn.data import *
from gnn.utils import *
from gnn.pytorch_model import *

# Import distributed feature engineering
try:
    from gnn.distributed_feature_pipeline import DistributedFeaturePipeline
    from gnn.feature_engineering import initialize_default_registry
    from gnn.advanced_features import (
        GraphCentralityExtractor, PatternMatchingExtractor, CrossFeatureExtractor
    )
    DISTRIBUTED_FEATURES_AVAILABLE = True
except ImportError:
    DISTRIBUTED_FEATURES_AVAILABLE = False
    print("Warning: Distributed feature engineering not available")

def initial_record():
    if os.path.exists('./output/results.txt'):
        os.remove('./output/results.txt')
    with open('./output/results.txt','w') as f:    
        f.write("Epoch,Time(s),Loss,F1\n")   


def normalize(feature_matrix):
    mean = th.mean(feature_matrix, axis=0)
    stdev = th.sqrt(th.sum((feature_matrix - mean)**2, axis=0)/feature_matrix.shape[0])
    return mean, stdev, (feature_matrix - mean) / stdev


def train_fg(model, optim, loss, features, labels, train_g, test_g, test_mask,
             device, n_epochs, thresh, compute_metrics=True):
    """
    A full graph version of RGCN training
    """

    # Move graph to device
    train_g = train_g.to(device)
    test_g = test_g.to(device)

    duration = []
    best_loss = 1
    for epoch in range(n_epochs):
        tic = time.time()
        loss_val = 0.

        pred = model(train_g, features.to(device))

        l = loss(pred, labels)

        optim.zero_grad()
        l.backward()
        optim.step()

        loss_val += l

        duration.append(time.time() - tic)
        metric = evaluate(model, train_g, features, labels, device)
        print("Epoch {:05d}, Time(s) {:.4f}, Loss {:.4f}, F1 {:.4f} ".format(
                epoch, np.mean(duration), loss_val, metric))
        
        epoch_result = "{:05d},{:.4f},{:.4f},{:.4f}\n".format(epoch, np.mean(duration), loss_val, metric)
        with open('./output/results.txt','a+') as f:    
            f.write(epoch_result)  

        if loss_val < best_loss:
            best_loss = loss_val
            best_model = copy.deepcopy(model)


    class_preds, pred_proba = get_model_class_predictions(best_model,
                                                          test_g,
                                                          features,
                                                          labels,
                                                          device,
                                                          threshold=thresh)

    if compute_metrics:
        acc, f1, p, r, roc, pr, ap, cm = get_metrics(class_preds, pred_proba, labels.numpy(), test_mask.numpy(), './output/')
        print("Metrics")
        print("""Confusion Matrix:
                                {}
                                f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, acc: {:.4f}, roc: {:.4f}, pr: {:.4f}, ap: {:.4f}
                             """.format(cm, f1, p, r, acc, roc, pr, ap))

    return best_model, class_preds, pred_proba


def get_f1_score(y_true, y_pred):
    """
    Only works for binary case.
    Attention!
    tn, fp, fn, tp = cf_m[0,0],cf_m[0,1],cf_m[1,0],cf_m[1,1]

    :param y_true: A list of labels in 0 or 1: 1 * N
    :param y_pred: A list of labels in 0 or 1: 1 * N
    :return:
    """
    # print(y_true, y_pred)

    cf_m = confusion_matrix(y_true, y_pred)
    # print(cf_m)

    precision = cf_m[1,1] / (cf_m[1,1] + cf_m[0,1] + 10e-5)
    recall = cf_m[1,1] / (cf_m[1,1] + cf_m[1,0])
    f1 = 2 * (precision * recall) / (precision + recall + 10e-5)

    return precision, recall, f1


def evaluate(model, g, features, labels, device):
    "Compute the F1 value in a binary classification case"

    model.eval()
    with th.no_grad():
        preds = model(g, features.to(device))
        preds = th.argmax(preds, axis=1).cpu().numpy()
        precision, recall, f1 = get_f1_score(labels.cpu().numpy(), preds)
    model.train()

    return f1


def get_model_class_predictions(model, g, features, labels, device, threshold=None):
    model.eval()
    with th.no_grad():
        unnormalized_preds = model(g, features.to(device))
        pred_proba = th.softmax(unnormalized_preds, dim=-1)
        if not threshold:
            return unnormalized_preds.argmax(axis=1).cpu().detach().numpy(), pred_proba[:,1].cpu().detach().numpy()
        return np.where(pred_proba.cpu().detach().numpy() > threshold, 1, 0), pred_proba[:,1].cpu().detach().numpy()


def save_model(g, model, model_dir, id_to_node, mean, stdev):

    # Save Pytorch model's parameters to model.pth
    th.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))

    # Save graph's structure information to metadata.pkl for inference codes to initialize RGCN model.
    etype_list = list(g.edge_types)
    ntype_cnt = {}
    for ntype in g.node_types:
        if hasattr(g[ntype], 'x') and g[ntype].x is not None:
            ntype_cnt[ntype] = g[ntype].x.shape[0]
        else:
            # Count from edge indices
            max_node = 0
            for etype in g.edge_types:
                if hasattr(g[etype], 'edge_index'):
                    edge_idx = g[etype].edge_index
                    if edge_idx.numel() > 0:
                        if etype[0] == ntype:
                            max_node = max(max_node, edge_idx[0].max().item() + 1)
                        if etype[2] == ntype:
                            max_node = max(max_node, edge_idx[1].max().item() + 1)
            ntype_cnt[ntype] = max_node if max_node > 0 else 1
    
    with open(os.path.join(model_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump({'etypes': etype_list,
                     'ntype_cnt': ntype_cnt,
                     'feat_mean': mean,
                     'feat_std': stdev}, f)

    # Save original IDs to Node_ids, and trained embedding for non-target node type
    # Covert id_to_node into pandas dataframes
    for ntype, mapping in id_to_node.items():

        # ignore target node
        if ntype == 'target':
            continue

        # retrieve old and node id list
        old_id_list, node_id_list = [], []
        for old_id, node_id in mapping.items():
            old_id_list.append(old_id)
            node_id_list.append(node_id)

        # retrieve embeddings of a node type
        node_feats = model.embed[ntype].detach().numpy()

        # get the number of nodes and the dimension of features
        num_nodes = node_feats.shape[0]
        num_feats = node_feats.shape[1]

        # create id dataframe
        node_ids_df = pd.DataFrame({'~label': [ntype] * num_nodes})
        node_ids_df['~id_tmp'] = old_id_list
        node_ids_df['~id'] = node_ids_df['~label'] + '-' + node_ids_df['~id_tmp']
        node_ids_df['node_id'] = node_id_list

        # create feature dataframe columns
        cols = {'val' + str(i + 1) + ':Double': node_feats[:, i] for i in range(num_feats)}
        node_feats_df = pd.DataFrame(cols)

        # merge id with feature, where feature_df use index
        node_id_feats_df = node_ids_df.merge(node_feats_df, left_on='node_id', right_on=node_feats_df.index)
        # drop the id_tmp and node_id columns to follow the Grelim format requirements
        node_id_feats_df = node_id_feats_df.drop(['~id_tmp', 'node_id'], axis=1)

        # dump the embeddings to files
        node_id_feats_df.to_csv(os.path.join(model_dir, ntype + '.csv'),
                                index=False, header=True, encoding='utf-8')


def enhance_features_with_distributed_engineering(features, graph, args, target_id_to_node):
    """
    Enhance features using distributed feature engineering
    
    Args:
        features: Original feature matrix
        graph: HeteroData graph object
        args: Training arguments
        target_id_to_node: Mapping of target node IDs
    
    Returns:
        Enhanced feature matrix or None if not available
    """
    try:
        # Check if pre-extracted features exist
        feature_file = os.path.join(args.feature_dir, 'extracted_features.npy')
        if os.path.exists(feature_file):
            print(f"Loading pre-extracted features from {feature_file}")
            enhanced = np.load(feature_file)
            # Align with current features
            if enhanced.shape[0] == features.shape[0]:
                return enhanced
            else:
                print(f"Warning: Feature shape mismatch. Expected {features.shape[0]}, got {enhanced.shape[0]}")
        
        # If enhance_features flag is set, extract on-the-fly
        if args.enhance_features:
            print("Extracting advanced features on-the-fly...")
            from gnn.distributed_feature_pipeline import DistributedFeaturePipeline
            from gnn.feature_engineering import initialize_default_registry
            from gnn.advanced_features import (
                GraphCentralityExtractor, PatternMatchingExtractor, CrossFeatureExtractor
            )
            
            # Initialize registry with advanced extractors
            registry = initialize_default_registry()
            registry.register(GraphCentralityExtractor())
            registry.register(PatternMatchingExtractor())
            registry.register(CrossFeatureExtractor())
            
            # Initialize pipeline
            pipeline = DistributedFeaturePipeline(
                registry=registry,
                n_workers=args.n_feature_workers,
                output_dir=args.feature_dir
            )
            
            # Create dummy transaction dataframe (in practice, you'd load the actual data)
            # For now, return None as we need the actual transaction data
            print("Warning: On-the-fly feature extraction requires transaction data. Use pre-extracted features instead.")
            return None
            
    except Exception as e:
        print(f"Error in distributed feature engineering: {e}")
        return None
    
    return None


def get_model(ntype_dict, etypes, hyperparams, in_feats, n_classes, device):

    model = HeteroRGCN(ntype_dict, etypes, in_feats, hyperparams['n_hidden'], n_classes, hyperparams['n_layers'], in_feats)
    model = model.to(device)

    return model


if __name__ == '__main__':
    # logging = get_logger(__name__)

    try:
        import torch_geometric
        pyg_version = torch_geometric.__version__
    except:
        pyg_version = "Not installed"
    
    print('numpy version:{} PyTorch version:{} PyG version:{}'.format(np.__version__,
                                                                      th.__version__,
                                                                      pyg_version))

    args = parse_args()
    print(args)

    args.edges = get_edgelists('relation*', args.training_dir)

    g, features, target_id_to_node, id_to_node = construct_graph(args.training_dir,
                                                                 args.edges,
                                                                 args.nodes,
                                                                 args.target_ntype)

    # Enhance features with distributed feature engineering if enabled
    if args.use_distributed_features and DISTRIBUTED_FEATURES_AVAILABLE:
        print("Enhancing features with distributed feature engineering...")
        enhanced_features = enhance_features_with_distributed_engineering(
            features, g, args, target_id_to_node
        )
        if enhanced_features is not None and enhanced_features.size > 0:
            # Concatenate original and enhanced features
            features = np.hstack([features, enhanced_features])
            print(f"Enhanced features shape: {features.shape}")
    
    mean, stdev, features = normalize(th.from_numpy(features))

    print('feature mean shape:{}, std shape:{}'.format(mean.shape, stdev.shape))

    # Update target node features in HeteroData
    g['target'].x = features

    print("Getting labels")
    n_nodes = g['target'].x.shape[0]

    labels, _, test_mask = get_labels(target_id_to_node,
                                               n_nodes,
                                               args.target_ntype,
                                               os.path.join(args.training_dir, args.labels),
                                               os.path.join(args.training_dir, args.new_accounts))
    print("Got labels")

    labels = th.from_numpy(labels).float()
    test_mask = th.from_numpy(test_mask).float()

    # Count nodes and edges in HeteroData
    n_nodes = sum([g[ntype].x.shape[0] if hasattr(g[ntype], 'x') and g[ntype].x is not None 
                   else len(g[ntype].edge_index[0].unique()) if hasattr(g[ntype], 'edge_index') 
                   else 0 for ntype in g.node_types])
    n_edges = sum([g[etype].edge_index.shape[1] if hasattr(g[etype], 'edge_index') else 0 
                   for etype in g.edge_types])

    print("""----Data statistics------'
                #Nodes: {}
                #Edges: {}
                #Features Shape: {}
                #Labeled Test samples: {}""".format(n_nodes,
                                                      n_edges,
                                                      features.shape,
                                                      test_mask.sum()))

    if args.num_gpus:
        cuda = True
        device = th.device('cuda:0')
    else:
        cuda = False
        device = th.device('cpu')

    print("Initializing Model")
    in_feats = features.shape[1]
    n_classes = 2

    # Get node type counts from HeteroData
    ntype_dict = {}
    for ntype in g.node_types:
        if hasattr(g[ntype], 'x') and g[ntype].x is not None:
            ntype_dict[ntype] = g[ntype].x.shape[0]
        else:
            # For nodes without features, count unique nodes in edge indices
            max_node = 0
            for etype in g.edge_types:
                if hasattr(g[etype], 'edge_index'):
                    edge_idx = g[etype].edge_index
                    if edge_idx.numel() > 0:
                        # Check which node types this edge connects
                        if etype[0] == ntype:
                            max_node = max(max_node, edge_idx[0].max().item() + 1)
                        if etype[2] == ntype:
                            max_node = max(max_node, edge_idx[1].max().item() + 1)
            ntype_dict[ntype] = max_node if max_node > 0 else 1

    # Get edge types from HeteroData
    etypes = list(g.edge_types)

    model = get_model(ntype_dict, etypes, vars(args), in_feats, n_classes, device)
    print("Initialized Model")

    features = features.to(device)

    labels = labels.long().to(device)
    test_mask = test_mask.to(device)

    loss = th.nn.CrossEntropyLoss()

    # print(model)
    optim = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("Starting Model training")
    initial_record()
    model, class_preds, pred_proba = train_fg(model, optim, loss, features, labels, g, g,
                                              test_mask, device, args.n_epochs,
                                              args.threshold,  args.compute_metrics)
    print("Finished Model training")

    print("Saving model")
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    save_model(g, model, args.model_dir, id_to_node, mean, stdev)
    print("Model and metadata saved")