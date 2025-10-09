import torch
import torch.nn as nn

class SGC(nn.Module):
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()
        self.fc = nn.Linear(nfeat, nclass)

    def forward(self, x):
        return self.fc(x)
