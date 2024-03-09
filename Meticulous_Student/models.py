import torch
from torchvision.models import resnet18, resnet101, resnet50, ResNet18_Weights
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Function


## 3 experts
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

class ResNetClassifier1(nn.Module):
    def __init__(self):
        super(ResNetClassifier1, self).__init__()
        self.resnet = resnet18(weights=None)
        num_features = self.resnet.fc.in_features
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc = nn.Linear(512, 1)  # FC layer for classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = torch.squeeze(x)
        x = self.fc(x)  # Pass the features through the FC layer
        x = self.sigmoid(x)
        return x


# DA expert

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


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

    def forward(self, input_data):
        input_data = input_data.expand(input_data.data.shape[0], 1, 102, 224)

        feature = self.feature(input_data)
        feature = feature.view(feature.size(0), -1)

        class_output = self.class_classifier(feature)

        return class_output, feature


# =================================================================
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.attend = nn.MultiheadAttention(dim, heads, dropout=dropout)

    def forward(self, x):
        # print(x.shape)
        q = x.permute((1, 0, 2))
        k = x.permute((1, 0, 2))
        v = x.permute((1, 0, 2))
        out, _ = self.attend(q, k, v)
        out = out.permute((1, 0, 2))
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class fa_selector(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0., out_dim=512, pool='mean'):
        super(fa_selector, self).__init__()
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout=dropout)
        self.pool = pool
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )

    def get_feat(self, x):
        x = self.transformer(x)
        if self.pool == 'mean':
            x = x.mean(dim=1)
        elif self.pool == 'max':
            x = x.max(dim=1)
        else:
            raise NotImplementedError
        return x

    def forward(self, x):
        x = self.transformer(x)
        if self.pool == 'mean':
            x = x.mean(dim=1)
        elif self.pool == 'max':
            x = x.max(dim=1)
        else:
            raise NotImplementedError

        x = self.mlp(x)

        return x


class fa_selector1(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0., out_dim=512, pool='mean'):
        super(fa_selector1, self).__init__()
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout=dropout)
        self.pool = pool
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )
        self.fc = nn.Linear(512, 1)  # FC layer for classification
        self.sigmoid = nn.Sigmoid()

    def get_feat(self, x):
        x = self.transformer(x)
        if self.pool == 'mean':
            x = x.mean(dim=1)
        elif self.pool == 'max':
            x = x.max(dim=1)
        else:
            raise NotImplementedError
        return x

    def forward(self, x):
        x = self.transformer(x)
        if self.pool == 'mean':
            x = x.mean(dim=1)
        elif self.pool == 'max':
            x = x.max(dim=1)
        else:
            raise NotImplementedError

        x = self.mlp(x)

        x = self.fc(x)  # Pass the features through the FC layer
        x = self.sigmoid(x)
        return x
