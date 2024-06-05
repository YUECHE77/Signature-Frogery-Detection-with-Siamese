import torch
import torch.nn as nn

from torch.hub import load_state_dict_from_url


class MyVgg16(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(MyVgg16, self).__init__()

        self.features = features  # the feature layer -> it is a block

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # Adaptively process input size, and output a set-size feature map

        self.classifier = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 512, out_features=4096),
            nn.ReLU(True),  # inplace = True -> overwrite the original data
            nn.Dropout(p=0.3),  # default p = 0.5, I set 0.3 here

            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(True),
            nn.Dropout(p=0.3),

            nn.Linear(in_features=4096, out_features=num_classes)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)

        x = torch.flatten(x, start_dim=1)  # flatten before pass through fully connected layer
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():  # self.modules()
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.constant_(m.bias, val=0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)


# (224, 224, 3) -> (224, 224, 64) -> (224, 224, 64) -> Pooling: (112, 112, 128)
# (112, 112, 128) -> (112, 112, 128) -> (112, 112, 128) -> Pooling: (56, 56, 256)
# (56, 56, 256) -> (56, 56, 256) -> (56, 56, 256) -> (56, 56, 256) -> Pooling: (28, 28, 512)
# (28, 28, 512) -> (28, 28, 512) -> (28, 28, 512) -> (28, 28, 512) -> Pooling: (14, 14, 512)
# (14, 14, 512) -> (14, 14, 512) -> (14, 14, 512) -> (14, 14, 512) -> Pooling: (7, 7, 512)


# Constructing the backbone of VGG16
def make_layers(cfg, batch_norm=False, in_channels=3):
    layers = []

    for param in cfg:
        if param == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=param, kernel_size=3, padding=1)

            # VGG16 or VGG16_bn
            if batch_norm:
                bn = nn.BatchNorm2d(num_features=param)  # number of input channel = out_channels of conv
                layers += [conv, bn, nn.ReLU(inplace=True)]
            else:
                layers += [conv, nn.ReLU(inplace=True)]

            in_channels = param

    return nn.Sequential(*layers)


cfgs = {
    'Default': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}


def construct_vgg16(pretrain=False, in_channels=3, **kwargs):
    features = make_layers(cfg=cfgs['Default'], batch_norm=False, in_channels=in_channels)
    vgg16 = MyVgg16(features=features, **kwargs)

    if pretrain:
        state_dict = load_state_dict_from_url(url="https://download.pytorch.org/models/vgg16-397923af.pth",
                                              model_dir='./model_data')
        vgg16.load_state_dict(state_dict=state_dict)

    return vgg16
