import torch.nn as nn

from .model import register


def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


@register("convnet4")
class ProtoNet(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        # mini-80/84: 64 * 5 * 5
        # cifar_fs-32: 64 * 2 * 2
        self.out_dim = 64 * 2 * 2

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)