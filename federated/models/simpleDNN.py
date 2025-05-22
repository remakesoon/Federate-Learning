import torch
import torch.nn as nn

class SimpleDNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleDNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1)
        self.fc = nn.Linear(32*28*28, num_classes)  # 7x7x32

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        return self.fc(x.view(x.size(0), -1))
