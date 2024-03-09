import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet101, resnet50, ResNet18_Weights


class ResNetClassifier(nn.Module):
    def __init__(self):
        super(ResNetClassifier, self).__init__()
        self.resnet = resnet18(weights=None)
        num_features = self.resnet.fc.in_features
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = torch.squeeze(x, dim=1)
        x = self.sigmoid(x)
        return x
