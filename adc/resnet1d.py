import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


def deconv3x3(in_planes, out_planes, stride=1):
    if stride == 1:
        return nn.Conv1d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
    else:
        return nn.ConvTranspose1d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            output_padding=1,
            bias=False,
        )


def deconv1x1(in_planes, out_planes, stride=1):
    if stride == 1:
        return nn.Conv1d(
            in_planes, out_planes, kernel_size=1, stride=stride, bias=False
        )
    else:
        return nn.ConvTranspose1d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=stride,
            bias=False,
            output_padding=1,
        )


class BasicDeconvBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super().__init__()
        self.deconv1 = deconv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.deconv2 = deconv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.deconv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.deconv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def make_basic_block(inplanes, planes, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = nn.Sequential(
            conv1x1(inplanes, planes, stride), nn.BatchNorm1d(planes),
        )
    return BasicBlock(inplanes, planes, stride, downsample)


class ResNetBlock(nn.Module):
    def __init__(self, block, in_channels=1):
        super().__init__()

        self.inplanes = 32
        self.conv1 = nn.Conv1d(
            in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(32)

        self.layer1 = self._make_layer(block, planes=32, blocks=1, stride=2)
        self.layer2 = self._make_layer(block, planes=64, blocks=1, stride=2)
        self.layer3 = self._make_layer(block, planes=128, blocks=1, stride=2)
        self.layer4 = self._make_layer(block, planes=128, blocks=1, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.tanh(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResNetDecoder(nn.Module):
    def __init__(self, block, in_channels, out_channels):
        super().__init__()
        self.dense_block = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=128),
            nn.ReLU(inplace=True),
        )
        self.inplanes = 64

        self.layer1 = self._make_layer(
            block, planes=self.inplanes, blocks=1, stride=2
        )
        self.layer2 = self._make_layer(block, planes=32, blocks=1, stride=2)
        self.layer3 = self._make_layer(block, planes=16, blocks=1, stride=2)
        self.layer4 = self._make_layer(block, planes=8, blocks=1, stride=2)

        self.deconv_out = nn.ConvTranspose1d(
            8, out_channels, kernel_size=3, stride=2, padding=1, bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                deconv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.dense_block(x)
        x = x.view(batch_size, 64, 2)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_out(x)
        # output only 52 spectrum values from the middle
        x = x[:, :, 5:57]
        return x
