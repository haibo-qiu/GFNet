import torch
from torch import nn
import torch.nn.functional as F
from libs.utils.tools import mp_logger

BatchNorm = nn.BatchNorm2d

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=(1, 0), bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = F.pad(x, (1, 1, 0, 0), mode='circular')
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = F.pad(out, (1, 1, 0, 0), mode='circular')
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=(1, 0), bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = F.pad(out, (1, 1, 0, 0), mode='circular')
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=(3, 0), bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.pad(x, (3, 3, 0, 0), mode='circular')
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        model_path = 'pretrained/resnet18-5c106cde.pth'
        mp_logger('loading model from: {}'.format(model_path))
        model.load_state_dict(torch.load(model_path), strict=False)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        model_path = 'pretrained/resnet34-333f7ec4.pth'
        mp_logger('loading model from: {}'.format(model_path))
        model.load_state_dict(torch.load(model_path), strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        model_path = 'pretrained/resnet50-19c8e357.pth'
        mp_logger('loading model from: {}'.format(model_path))
        model.load_state_dict(torch.load(model_path), strict=False)
    return model

class UnetCircular(nn.Module):
    def __init__(self, n_class, n_height, layers=18, BatchNorm=nn.BatchNorm2d, pretrained=True):
        super(UnetCircular, self).__init__()
        assert n_class > 1
        if layers == 18:
            resnet = resnet18(pretrained=pretrained)
            block = BasicBlock
            layers = [2, 2, 2, 2]
        elif layers == 34:
            resnet = resnet34(pretrained=pretrained)
            block = BasicBlock
            layers = [3, 4, 6, 3]
        elif layers == 50:
            resnet = resnet50(pretrained=pretrained)
            block = Bottleneck
            layers = [3, 4, 6, 3]
        self.layer0 = inconv(n_height, 64)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        # Decoder
        self.up4 = nn.Sequential(
            nn.Conv2d(512 * block.expansion, 256 * block.expansion, kernel_size=3, stride=1, padding=(1, 0)),
            BatchNorm(256 * block.expansion), nn.ReLU())
        resnet.inplanes = 256 * block.expansion + 256 * block.expansion
        self.delayer4 = resnet._make_layer(block, 256, layers[-1])

        self.up3 = nn.Sequential(
            nn.Conv2d(256 * block.expansion, 128 * block.expansion, kernel_size=3, stride=1, padding=(1, 0)),
            BatchNorm(128 * block.expansion), nn.ReLU())
        resnet.inplanes = 128 * block.expansion + 128 * block.expansion
        self.delayer3 = resnet._make_layer(block, 128, layers[-2])

        self.up2 = nn.Sequential(
            nn.Conv2d(128 * block.expansion, 64 * block.expansion, kernel_size=3, stride=1, padding=(1, 0)),
            BatchNorm(64 * block.expansion), nn.ReLU())
        resnet.inplanes = 64 * block.expansion + 64 * block.expansion
        self.delayer2 = resnet._make_layer(block, 64, layers[-3])

        self.extend = nn.Sequential(
            nn.Conv2d(64 * block.expansion, 64 * block.expansion * (n_height//4), kernel_size=3, padding=(1, 0), bias=False),
            BatchNorm(64 * block.expansion * (n_height//4)),
            nn.ReLU(inplace=True),
        )

        self.cls = nn.Sequential(
            nn.Conv2d(64 * block.expansion, 256, kernel_size=3, padding=(1, 0), bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_class, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(nn.Conv2d(256 * block.expansion, 256, kernel_size=1, bias=False),
                                     BatchNorm(256), nn.ReLU(inplace=True),
                                     nn.Conv2d(256, n_class, kernel_size=1))

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.layer0(x)  # 1/4
        x2 = self.layer1(x)  # 1/4
        x3 = self.layer2(x2)  # 1/8
        x4 = self.layer3(x3)  # 1/16
        x5 = self.layer4(x4)  # 1/32

        out = F.interpolate(x5, x4.shape[-2:], mode='bilinear', align_corners=True)
        out = F.pad(out, (1, 1, 0, 0), mode='circular')
        p4 = self.up4(out)  # 1/16
        p4 = torch.cat([p4, x4], dim=1)
        p4 = self.delayer4(p4)

        out = F.interpolate(p4, x3.shape[-2:], mode='bilinear', align_corners=True)
        out = F.pad(out, (1, 1, 0, 0), mode='circular')
        p3 = self.up3(out)  # 1/8
        p3 = torch.cat([p3, x3], dim=1)
        p3 = self.delayer3(p3)

        out = F.interpolate(p3, x2.shape[-2:], mode='bilinear', align_corners=True)
        out = F.pad(out, (1, 1, 0, 0), mode='circular')
        p2 = self.up2(out)  # 1/4
        p2 = torch.cat([p2, x2], dim=1)
        p2 = self.delayer2(p2)

        p2 = F.pad(p2, (1, 1, 0, 0), mode='circular')
        x = self.cls(p2)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)  # 1/1
        if self.training:
            aux = self.aux(x4)
            aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            return x, aux
        else:
            return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=(1, 0), bias=False)
        self.bn = BatchNorm(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=(1, 0))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        x = self.maxpool(x)
        return x


class BEV_Unet(nn.Module):
    def __init__(self, n_class, n_height, layers=18, BatchNorm=nn.BatchNorm2d, pretrained=True):
        super(BEV_Unet, self).__init__()
        self.n_class = n_class
        self.n_height = n_height
        self.network = UnetCircular(n_class * n_height, n_height, layers, BatchNorm, pretrained)

    def forward(self, x):
        x = self.network(x)
        x = x.permute(0, 2, 3, 1)
        new_shape = list(x.size())[:3] + [self.n_height, self.n_class]
        x = x.view(new_shape)
        x = x.permute(0, 4, 1, 2, 3)
        return x
