import torch.nn as nn
import torchvision.models as models
from functions import ReverseLayerF

class DA_Model(nn.Module):
    def __init__(self):
        super(DA_Model, self).__init__()

        # Load pretrained ResNet-18 model
        resnet = models.resnet18(weights=None)
        num_features = resnet.fc.in_features
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the last fully connected layer of ResNet-18
        self.feature = nn.Sequential(*list(resnet.children())[:-1])

        self.class_classifier = nn.Linear(num_features, 2)

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(512, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100, track_running_stats=False))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 3))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        # input_data = input_data.expand(input_data.data.shape[0], 1, 102, 224)

        feature = self.feature(input_data)
        feature = feature.view(feature.size(0), -1)

        reverse_feature = ReverseLayerF.apply(feature, alpha)

        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output
