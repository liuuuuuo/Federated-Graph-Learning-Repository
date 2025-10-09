import os
import numpy as np
import pickle as pkl
import scipy.sparse as sp

# 修正为相对于当前脚本的路径
PLANETOID_DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/Planetoid')

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset_str):
    names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph']
    objects = []
    for name in names:
        with open(os.path.join(PLANETOID_DATA_DIR, f"ind.{dataset_str}.{name}"), 'rb') as f:
            if name == 'graph':
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f, encoding='latin1'))
    x, tx, allx, y, ty, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(PLANETOID_DATA_DIR, f"ind.{dataset_str}.test.index"))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    adj = sp.coo_matrix(sp.csr_matrix((len(graph), len(graph))))
    adj = sp.lil_matrix((len(graph), len(graph)))
    for i, neighbors in graph.items():
        adj[i, neighbors] = 1
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y) + 500))

    return adj, features, labels, idx_train, idx_val, idx_test

# 用法示例
if __name__ == "__main__":
    adj, features, labels, idx_train, idx_val, idx_test = load_data('cora')
    print("adj shape:", adj.shape)
    print("features shape:", features.shape)
    print("labels shape:", labels.shape)
