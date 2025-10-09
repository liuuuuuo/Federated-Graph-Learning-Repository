import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from data_loader import load_data
from gcn import GCN
from utils import normalize_adj, sparse_mx_to_torch_sparse_tensor

# 加载数据
adj, features, labels, idx_train, idx_val, idx_test = load_data('cora')

# 数据预处理
features = torch.FloatTensor(np.array(features.todense()))
labels = torch.LongTensor(np.where(labels)[1])
adj_norm = normalize_adj(adj)
adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

# 模型与优化器#1#模型定义↓
model = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max().item()+1, dropout=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
#2#优化器选择↑
def train(epoch):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj_norm)#3#前向传播
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])#4#计算损失
    acc_train = (output[idx_train].argmax(1) == labels[idx_train]).float().mean()
    loss_train.backward()#5#反向传播
    optimizer.step()#6#优化器参数更新
    print(f'Epoch {epoch}, Loss: {loss_train.item():.4f}, Train Acc: {acc_train.item():.4f}')

def test():
    model.eval()
    with torch.no_grad():
        output = model(features, adj_norm)
        loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
        acc_test = (output[idx_test].argmax(1) == labels[idx_test]).float().mean()
        print(f"Test set results: Loss = {loss_test.item():.4f}, Accuracy = {acc_test.item():.4f}")

for epoch in range(1, 201):
    train(epoch)

test()
