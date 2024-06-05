import torch
import torch.nn as nn
import torch.nn.functional as F

from .VGG16 import construct_vgg16


def get_output_size(width, height):
    # Number of output channels is set -> 512
    # The output width and height is determined by pooling
    def compute_size(input_shape):
        kernel_size = [2, 2, 2, 2, 2]
        stride = [2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0]

        for i in range(len(kernel_size)):
            # Just follow the formula of computing dimension
            input_shape = (input_shape - kernel_size[i] + 2 * padding[i]) // stride[i] + 1

        return input_shape

    return compute_size(width) * compute_size(height)


class Siamese(nn.Module):
    def __init__(self, input_shape=224, pretrain=False, gray=True):
        super(Siamese, self).__init__()

        if gray:
            self.vgg = construct_vgg16(pretrain=False, in_channels=1).features
            flat_shape = 512 * 7 * 7  # should be 25088
        else:
            self.vgg = construct_vgg16(pretrain=pretrain, in_channels=3).features
            flat_shape = get_output_size(width=input_shape, height=input_shape)

        self.fully_connect1 = nn.Linear(in_features=flat_shape, out_features=512)
        self.fully_connect2 = nn.Linear(in_features=512, out_features=1)

    def forward(self, img1, img2):
        x1 = self.vgg(img1)
        x2 = self.vgg(img2)

        # Then flatten them
        x1 = torch.flatten(x1, start_dim=1)
        x2 = torch.flatten(x2, start_dim=1)
        x = torch.abs(x1 - x2)

        # Pass through to the Fully connected layer
        x = self.fully_connect1(x)
        x = F.relu(x)  # don't know if it is necessary

        x = self.fully_connect2(x)

        return x
