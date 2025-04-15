from torch import nn
import torch.nn.functional as F


class CNNEmnist(nn.Module):
    def __init__(self, num_classes):
        super(CNNEmnist, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x