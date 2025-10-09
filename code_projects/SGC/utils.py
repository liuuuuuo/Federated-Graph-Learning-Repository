import numpy as np
import scipy.sparse as sp
import torch

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj_.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sgc_precompute(features, adj, degree):
    """K阶邻接归一化特征"""
    adj_normalized = normalize_adj(adj)
    result = features
    for _ in range(degree):
        result = adj_normalized.dot(result)
    return result

def to_tensor(x):
    if sp.issparse(x):
        return torch.FloatTensor(np.array(x.todense()))
    else:
        return torch.FloatTensor(np.array(x))
