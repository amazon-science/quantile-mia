"""ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet10(**kwargs):
    return ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)


def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


# MICO network
class CNN(nn.Module):
    def __init__(self, out_features: int = 10):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=8, stride=2, padding=3),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.Tanh(),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=6400, out_features=out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape of x is [B, 3, 32, 32] for CIFAR10
        logits = self.cnn(x)
        return logits


# Wide ResNets in the style of https://objax.readthedocs.io/en/latest/_modules/objax/zoo/wide_resnet.html#WRNBlock

BN_MOM = 0.9
BN_EPS = 1e-5


class WRNBlock(nn.Module):
    """WideResNet block."""

    def __init__(
        self, nin: int, nout: int, stride: int = 1, bn: Callable = nn.BatchNorm2d
    ):
        """Creates WRNBlock instance.

        Args:
            nin: number of input filters.
            nout: number of output filters.
            stride: stride for convolution and projection convolution in this block.
            bn: module which used as batch norm function.
        """
        super().__init__()
        if nin != nout or stride > 1:
            # self.proj_conv = objax.nn.Conv2D(nin, nout, 1, strides=stride, **conv_args(1, nout))
            self.proj_conv = nn.Conv2d(
                nin, nout, kernel_size=1, stride=stride, bias=False
            )
        else:
            self.proj_conv = None

        self.norm_1 = bn(nin, eps=BN_EPS, momentum=BN_MOM)
        # self.conv_1 = objax.nn.Conv2D(nin, nout, 3, strides=stride, **conv_args(3, nout))
        self.conv_1 = nn.Conv2d(
            nin, nout, kernel_size=3, stride=stride, bias=False, padding=1
        )
        self.norm_2 = bn(nout, eps=BN_EPS, momentum=BN_MOM)
        # self.conv_2 = objax.nn.Conv2D(nout, nout, 3, strides=1, **conv_args(3, nout))
        self.conv_2 = nn.Conv2d(
            nout, nout, kernel_size=3, stride=1, bias=False, padding=1
        )

    def forward(self, x):
        o1 = F.relu(self.norm_1(x))
        y = self.conv_1(o1)
        o2 = F.relu(self.norm_2(y))
        z = self.conv_2(o2)
        return z + self.proj_conv(o1) if self.proj_conv else z + x


class WideResNetGeneral(nn.Module):
    """Base WideResNet implementation."""

    def __init__(
        self,
        nin: int,
        nclass: int,
        blocks_per_group: List[int],
        width: int,
        bn: Callable = nn.BatchNorm2d,
    ):
        """Creates WideResNetGeneral instance.

        Args:
            nin: number of channels in the input image.
            nclass: number of output classes.
            blocks_per_group: number of blocks in each block group.
            width: multiplier to the number of convolution filters.
            bn: module which used as batch norm function.
        """
        super().__init__()
        widths = [
            int(v * width)
            for v in [16 * (2**i) for i in range(len(blocks_per_group))]
        ]

        n = 16
        # ops = [objax.nn.Conv2D(nin, n, 3, **conv_args(3, n))]
        ops = [nn.Conv2d(nin, n, kernel_size=3, bias=False, padding=1)]
        for i, (block, width) in enumerate(zip(blocks_per_group, widths)):
            stride = 2 if i > 0 else 1
            ops.append(WRNBlock(n, width, stride, bn))
            for b in range(1, block):
                ops.append(WRNBlock(width, width, 1, bn))
            n = width
        ops += [
            bn(n, eps=BN_EPS, momentum=BN_MOM),
            # objax.functional.relu,
            nn.ReLU(),
            # self.mean_reduce,
            torch.nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # objax.nn.Linear(n, nclass, w_init=objax.nn.init.xavier_truncated_normal)
            nn.Linear(n, nclass),
        ]
        self.model = nn.Sequential(*ops)

    def forward(self, x):
        return self.model(x)


class WideResNet(WideResNetGeneral):
    """WideResNet implementation with 3 groups.

    Reference:
        http://arxiv.org/abs/1605.07146
        https://github.com/szagoruyko/wide-residual-networks
    """

    def __init__(
        self,
        num_classes: int = 10,
        nin: int = 3,
        depth: int = 28,
        width: int = 2,
        # bn: Callable = functools.partial(objax.nn.BatchNorm2D, momentum=BN_MOM, eps=BN_EPS)):
        bn: Callable = nn.BatchNorm2d,
    ):
        """Creates WideResNet instance.

        Args:
            nin: number of channels in the input image.
            nclass: number of output classes.
            depth: number of convolution layers. (depth-4) should be divisible by 6
            width: multiplier to the number of convolution filters.
            bn: module which used as batch norm function.
        """
        assert (depth - 4) % 6 == 0, "depth should be 6n+4"
        n = (depth - 4) // 6
        blocks_per_group = [n] * 3
        super().__init__(nin, num_classes, blocks_per_group, width, bn)
