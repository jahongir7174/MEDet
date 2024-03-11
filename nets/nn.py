import math

import numpy
import torch


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.norm(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p):
        super().__init__()

        assert k == 3
        assert p == 1
        self.in_channels = in_ch

        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Identity()

        self.conv1 = Conv(in_ch, out_ch, k=k, s=s, p=p)
        self.conv2 = Conv(in_ch, out_ch, k=1, s=s, p=p - k // 2)
        self.identity = torch.nn.BatchNorm2d(in_ch) if in_ch == out_ch and s == 1 else None

    @staticmethod
    def __pad(k):
        if k is None:
            return 0
        else:
            return torch.nn.functional.pad(k, [1, 1, 1, 1])

    def __fuse_norm(self, m):
        if m is None:
            return 0, 0
        if isinstance(m, Conv):
            kernel = m.conv.weight
            running_mean = m.norm.running_mean
            running_var = m.norm.running_var
            gamma = m.norm.weight
            beta = m.norm.bias
            eps = m.norm.eps
        else:
            assert isinstance(m, torch.nn.BatchNorm2d)
            if not hasattr(self, 'norm'):
                k = numpy.zeros((self.in_channels, self.in_channels, 3, 3), dtype=numpy.float32)
                for i in range(self.in_channels):
                    k[i, i % self.in_channels, 1, 1] = 1
                self.norm = torch.from_numpy(k).to(m.weight.device)
            kernel = self.norm
            running_mean = m.running_mean
            running_var = m.running_var
            gamma = m.weight
            beta = m.bias
            eps = m.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def forward(self, x):
        if self.identity is None:
            return self.relu(self.conv1(x) + self.conv2(x))
        else:
            return self.relu(self.conv1(x) + self.conv2(x) + self.identity(x))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))

    def fuse(self):
        k1, b1 = self.__fuse_norm(self.conv1)
        k2, b2 = self.__fuse_norm(self.conv2)
        k3, b3 = self.__fuse_norm(self.identity)

        self.conv = torch.nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                                    out_channels=self.conv1.conv.out_channels,
                                    kernel_size=self.conv1.conv.kernel_size,
                                    stride=self.conv1.conv.stride,
                                    padding=self.conv1.conv.padding,
                                    dilation=self.conv1.conv.dilation,
                                    groups=self.conv1.conv.groups, bias=True)
        self.conv.weight.data = k1 + self.__pad(k2) + k3
        self.conv.bias.data = b1 + b2 + b3

        if hasattr(self, 'conv1'):
            self.__delattr__('conv1')
        if hasattr(self, 'conv2'):
            self.__delattr__('conv2')
        if hasattr(self, 'identity'):
            self.__delattr__('identity')
        if hasattr(self, 'norm'):
            self.__delattr__('norm')

        self.forward = self.fuse_forward


class RepVGG(torch.nn.Module):
    def __init__(self, width, depth, num_classes=1000):
        super().__init__()

        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1
        self.p1.append(Residual(width[0], width[1], k=3, s=2, p=1))
        # p2
        for i in range(depth[0]):
            if i == 0:
                self.p2.append(Residual(width[1], width[2], k=3, s=2, p=1))
            else:
                self.p2.append(Residual(width[2], width[2], k=3, s=1, p=1))
        # p3
        for i in range(depth[1]):
            if i == 0:
                self.p3.append(Residual(width[2], width[3], k=3, s=2, p=1))
            else:
                self.p3.append(Residual(width[3], width[3], k=3, s=1, p=1))
        # p4
        for i in range(depth[2]):
            if i == 0:
                self.p4.append(Residual(width[3], width[4], k=3, s=2, p=1))
            else:
                self.p4.append(Residual(width[4], width[4], k=3, s=1, p=1))
        # p5
        for i in range(depth[3]):
            if i == 0:
                self.p5.append(Residual(width[4], width[5], k=3, s=2, p=1))
            else:
                self.p5.append(Residual(width[5], width[5], k=3, s=1, p=1))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)
        self.fc = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                      torch.nn.Flatten(),
                                      torch.nn.Dropout(0.2),
                                      torch.nn.Linear(width[5], num_classes))

    def forward(self, x):
        x = self.p1(x)
        x = self.p2(x)
        x = self.p3(x)
        x = self.p4(x)
        x = self.p5(x)
        return self.fc(x)

    def fuse(self):
        for m in self.modules():
            if type(m) is Residual:
                m.fuse()
        return self


def rep_vgg_a0():
    return RepVGG(width=[3, 48, 48, 96, 192, 1280], depth=[2, 4, 14, 1])


def rep_vgg_a1():
    return RepVGG(width=[3, 64, 64, 128, 256, 1280], depth=[2, 4, 14, 1])


def rep_vgg_a2():
    return RepVGG(width=[3, 64, 96, 192, 384, 1408], depth=[2, 4, 14, 1])


def rep_vgg_b0():
    return RepVGG(width=[3, 64, 64, 128, 256, 1280], depth=[4, 6, 16, 1])


def rep_vgg_b1():
    return RepVGG(width=[3, 64, 128, 256, 512, 2048], depth=[4, 6, 16, 1])


def rep_vgg_b2():
    return RepVGG(width=[3, 64, 160, 320, 640, 2560], depth=[4, 6, 16, 1])


class CosineLR:
    def __init__(self, args, params):
        min_lr = params['min_lr']
        max_lr = params['max_lr']

        epochs = int(args.epochs)
        warmup_epochs = int(params['warmup_epochs'])

        cosine_lr = []
        linear_lr = numpy.linspace(min_lr, max_lr, warmup_epochs, endpoint=False)

        for epoch in range(epochs - warmup_epochs + 1):
            alpha = math.cos(math.pi * epoch / args.epochs)
            cosine_lr.append(min_lr + 0.5 * (max_lr - min_lr) * (1 + alpha))

        self.learning_rates = numpy.concatenate((linear_lr, cosine_lr))

    def step(self, epoch, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.learning_rates[epoch]


class CrossEntropyLoss(torch.nn.Module):
    """
    NLL Loss with label smoothing.
    """

    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, outputs, targets):
        prob = self.softmax(outputs)
        mean = torch.mean(prob, dim=-1)

        index = torch.unsqueeze(targets, dim=1)

        nll_loss = torch.gather(prob, -1, index)
        nll_loss = torch.squeeze(nll_loss, dim=1)

        return ((self.epsilon - 1) * nll_loss - self.epsilon * mean).mean()
