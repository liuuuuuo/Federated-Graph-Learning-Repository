import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append('../GCN')
from data_loader import load_data
from sgc import SGC
from utils import sgc_precompute, to_tensor

# 加载数据
adj, features, labels, idx_train, idx_val, idx_test = load_data('cora')

# SGC特征预处理
degree = 2  # SGC论文推荐K=2
features_sgc = sgc_precompute(features, adj, degree)
features_sgc = to_tensor(features_sgc)
labels = torch.LongTensor(np.where(labels)[1])
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

# 模型与优化器
model = SGC(nfeat=features_sgc.shape[1], nclass=labels.max().item()+1)
optimizer = optim.Adam(model.parameters(), lr=0.2, weight_decay=5e-6)

def train():
    model.train()
    optimizer.zero_grad()
    output = model(features_sgc)
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = (output[idx_train].argmax(1) == labels[idx_train]).float().mean()
    loss_train.backward()
    optimizer.step()
    print(f'Loss: {loss_train.item():.4f}, Train Acc: {acc_train.item():.4f}')

def test():
    model.eval()
    with torch.no_grad():
        output = model(features_sgc)
        loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
        acc_test = (output[idx_test].argmax(1) == labels[idx_test]).float().mean()
        print(f"Test set results: Loss = {loss_test.item():.4f}, Accuracy = {acc_test.item():.4f}")

for epoch in range(1, 101):
    train()

test()
