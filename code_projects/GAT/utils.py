import numpy as np
import scipy.sparse as sp
import torch

def to_tensor(x):
    if sp.issparse(x):
        return torch.FloatTensor(np.array(x.todense()))
    else:
        return torch.FloatTensor(np.array(x))

def adj_to_dense_tensor(adj):
    if sp.issparse(adj):
        return torch.FloatTensor(adj.todense())
    else:
        return torch.FloatTensor(adj)
