import torch.nn as nn
import torch
import math
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)  # 有无bias对bn没多大影响
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.downsample = downsample

    def forward(self, x):
        identity = x  # 记录上一个残差模块输出的结果
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, input_layer):  # layers=参数列表 block选择不同的类
        self.inplanes = 32
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_layer, 32, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))  # 每个blocks的第一个residual结构保存在layers列表中。
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))  # 该部分是将每个blocks的剩下residual 结构保存在layers列表中，这样就完成了一个blocks的构造。

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class RL_model(nn.Module):
    def __init__(self, line_board, hidden_dim, out_layer, resnet18):
        super(RL_model, self).__init__()
        self.out_dim = 4
        self.out_layer = out_layer

        self.lines = line_board
        self.resnet = resnet18

        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.tanh = nn.Tanh()

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=out_layer, kernel_size=1, stride=1)

        # 计算棋盘落子概率的线形层
        self.tran_layer1 = nn.Linear(out_layer * self.out_dim * self.out_dim, hidden_dim)
        self.tran_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.tran_layer3 = nn.Linear(hidden_dim, line_board * line_board)

    def forward(self, x):
        x = self.resnet(x)
        # 计算每个点的概率
        pro = self.conv2(x).view(-1, self.out_layer * self.out_dim * self.out_dim)
        pro = self.leakyrelu(self.tran_layer1(pro))
        pro = self.leakyrelu(self.tran_layer2(pro))
        pro = self.leakyrelu(self.tran_layer3(pro))

        return pro


def make_model(line_board=15, hidden_dim=100, input_layer=17, out_layer=15):

    resnet = ResNet(BasicBlock, [2, 2, 2, 2], input_layer=input_layer)

    model = RL_model(line_board=line_board, hidden_dim=hidden_dim, out_layer=out_layer, resnet18=resnet)

    return model


"""
data = torch.randn(1, 17, 15, 15)

model = make_model(line_board=15, hidden_dim=100, input_layer=17, out_layer=15)

for i in range(8000):
    out = model(data)
"""