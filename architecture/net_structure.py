import torch
from torch import nn
from torch.nn import functional as F


class Alex(nn.Module):
    def __init__(self):
        channels_conv = [3, 96, 256, 384, 384, 256]
        super(Alex, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels_conv[0],
                               out_channels=channels_conv[1],
                               kernel_size=11,
                               stride=2,
                               padding=0,
                               bias=True)
        self.batch1 = nn.BatchNorm2d(num_features=channels_conv[1],
                                     affine=True,
                                     track_running_stats=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3,
                                  stride=2,
                                  padding=0)
        self.conv2 = nn.Conv2d(in_channels=channels_conv[1],
                               out_channels=channels_conv[2],
                               kernel_size=5,
                               stride=1,
                               padding=0,
                               groups=2,
                               bias=True)
        self.batch2 = nn.BatchNorm2d(num_features=channels_conv[2],
                                     affine=True,
                                     track_running_stats=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3,
                                  stride=2,
                                  padding=0)
        self.conv3 = nn.Conv2d(in_channels=channels_conv[2],
                               out_channels=channels_conv[3],
                               kernel_size=3,
                               stride=1,
                               padding=0,
                               bias=True)
        self.batch3 = nn.BatchNorm2d(num_features=channels_conv[3],
                                     affine=True,
                                     track_running_stats=True)
        self.conv4 = nn.Conv2d(in_channels=channels_conv[3],
                               out_channels=channels_conv[4],
                               kernel_size=3,
                               stride=1,
                               padding=0,
                               groups=2,
                               bias=True)
        self.batch4 = nn.BatchNorm2d(num_features=channels_conv[4],
                                     affine=True,
                                     track_running_stats=True)
        self.conv5 = nn.Conv2d(in_channels=channels_conv[4],
                               out_channels=channels_conv[5],
                               kernel_size=3,
                               stride=1,
                               padding=0,
                               groups=2,
                               bias=True)
        # self.batch5 = nn.BatchNorm2d(num_features=channels_conv[5],
        #                              affine=True,
        #                              track_running_stats=True)  # todo

    def forward(self, x):
        x = self.pool1(F.relu(self.batch1(self.conv1(x))))
        x = self.pool2(F.relu(self.batch2(self.conv2(x))))
        x = F.relu(self.batch3(self.conv3(x)))
        x = F.relu(self.batch4(self.conv4(x)))
        # feature = self.batch5(self.conv5(x))  # todo
        feature = self.conv5(x)
        return feature


class Correlation(nn.Module):
    def __init__(self):
        super(Correlation, self).__init__()

    def forward(self, z, x):
        assert z.size()[1] == x.size()[1]
        assert z.size()[2] <= x.size()[2]
        assert z.size()[3] <= x.size()[3]
        x_stack = x.view(1, -1, x.size()[2], x.size()[3])
        score_stack = 1e-3 * F.conv2d(input=x_stack, weight=z, stride=1, padding=0, groups=x.size()[0])
        score = score_stack.view(x.size()[0], 1, score_stack.size()[2], score_stack.size()[3])
        return score


class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.cnn = Alex()
        self.corr = Correlation()
        self.z_feat = torch.Tensor()

    def set_target(self, z):
        self.z_feat = self.cnn(z)

    def forward(self, x, z=None):
        if z is not None:
            self.set_target(z)
        score = self.corr(self.z_feat, self.cnn(x))
        return score


if __name__ == '__main__':
    net = Siamese()
    print(net)
