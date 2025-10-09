import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append('../GCN')
from data_loader import load_data
from gat import GAT
from utils import to_tensor, adj_to_dense_tensor

# 加载数据
adj, features, labels, idx_train, idx_val, idx_test = load_data('cora')

features = to_tensor(features)
labels = torch.LongTensor(np.where(labels)[1])
adj = adj_to_dense_tensor(adj)

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

# 模型与优化器
model = GAT(nfeat=features.shape[1], nhid=8, nclass=labels.max().item()+1, dropout=0.6, alpha=0.2, nheads=8)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

def train(epoch):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = (output[idx_train].argmax(1) == labels[idx_train]).float().mean()
    loss_train.backward()
    optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss_train.item():.4f}, Train Acc: {acc_train.item():.4f}')

def test():
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
        acc_test = (output[idx_test].argmax(1) == labels[idx_test]).float().mean()
        print(f"Test set results: Loss = {loss_test.item():.4f}, Accuracy = {acc_test.item():.4f}")

for epoch in range(1, 201):
    train(epoch)

test()
