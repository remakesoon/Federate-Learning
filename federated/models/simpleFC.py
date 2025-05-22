import torch
import torch.nn as nn

class SimpleFC(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleFC, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # 输入尺寸为28x28，隐藏层128个神经元
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.8)  # 防止过拟合

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 将空间维度压成一维
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
