import os
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch as th
from data_prop import sparse_mx_to_torch_sparse_tensor


def load_data(dataset_name, num_labels=None, *args):
    """
    Load graph data for multilabel classification from precomputed files with separate label columns.

    Parameters:
        dataset_name (str): Name of the dataset directory under '../new_data'.
        num_labels (int, optional): Number of label columns. If None, inferred from header.
        *args: Ignored (e.g., split indices or percentages).

    Returns:
        adj1 (torch.sparse.FloatTensor): Normalized adjacency matrix with self-loops on GPU.
        features (torch.FloatTensor): Node features on GPU.
        labels (torch.FloatTensor): Multi-hot label matrix (nodes x labels) on GPU.
    """
    # Setup file paths
    base_path = os.path.join('../new_data', dataset_name)
    edge_path = os.path.join(base_path, 'out1_graph_edges.txt')
    node_path = os.path.join(base_path, 'out1_node_feature_label.txt')

    # Read header to determine label columns
    with open(node_path) as f:
        header = f.readline().rstrip().split('\t')
    label_cols = header[2:]
    inferred_labels = len(label_cols)
    if num_labels is None:
        num_labels = inferred_labels
    elif num_labels != inferred_labels:
        # Mismatch between provided and actual columns; will proceed with actual count
        num_labels = inferred_labels

    # Read node features and label vectors
    node_feats = {}
    node_labels = {}
    with open(node_path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.rstrip().split('\t')
            idx = int(parts[0])
            feat = np.array(parts[1].split(','), dtype=np.float16)
            lbl_vals = [int(x) for x in parts[2:2 + num_labels]] #cls
            #lbl_vals = [float(x) for x in parts[2:2 + num_labels]]
            node_feats[idx] = feat
            node_labels[idx] = lbl_vals
                # ─── DEBUG PRINTS ──────────────────────────────────────────────────────────────
        print(f"[load_data] total nodes read: {len(node_labels)}")
        print("[load_data] sample node→labels:")
        for nid in sorted(node_labels)[:10]:
            print(f"   node {nid} → {node_labels[nid]}")
        print(f"[load_data] label matrix shape will be: ({len(node_labels)}, {num_labels})")
    # ────────────────────────────────────────────────────────────────────────────────


    # Build directed graph
    G = nx.DiGraph()
    with open(edge_path) as f:
        f.readline()  # skip header
        for line in f:
            src, dst = map(int, line.rstrip().split('\t'))
            if src not in G:
                G.add_node(src, features=node_feats[src], label=node_labels[src])
            if dst not in G:
                G.add_node(dst, features=node_feats[dst], label=node_labels[dst])
            G.add_edge(src, dst)

    # Create adjacency matrix
    adj = nx.adjacency_matrix(G, nodelist=sorted(G.nodes()))

    # Stack features and labels in sorted order
    features = np.array([data['features'] for _, data in sorted(G.nodes(data=True))], dtype=np.float32)
    labels = np.array([data['label'] for _, data in sorted(G.nodes(data=True))], dtype=np.float32)

    # Symmetrize and add self-loops
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_self = adj + sp.eye(adj.shape[0])

    # Normalize adjacency
    adj_norm = normalize_adj(adj_self)
    adj1 = sparse_mx_to_torch_sparse_tensor(adj_norm)

    # Convert to torch tensors
    features = th.FloatTensor(features)
    labels = th.FloatTensor(labels)

    # Move to GPU
    features = features.cuda()
    adj1 = adj1.cuda()
    labels = labels.cuda()

    return adj1, features, labels


def normalize_adj(mx):
    """Symmetrically normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)).flatten()
    inv_sqrt = np.power(rowsum, -0.5)
    inv_sqrt[np.isinf(inv_sqrt)] = 0.0
    mat_inv_sqrt = sp.diags(inv_sqrt)
    return mx.dot(mat_inv_sqrt).transpose().dot(mat_inv_sqrt)