import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, ds=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes,
                               kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=True)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=True)
        for i in self.bn2.parameters():
            i.requires_grad = False

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=True)
        for i in self.bn3.parameters():
            i.requires_grad = False

        self.relu = nn.ReLU(inplace=True)
        self.ds = ds
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.ds is not None:
            residual = self.ds(x)

        out += residual
        out = self.relu(out)

        return out


class Residual_Covolution(nn.Module):
    def __init__(self, icol, ocol, num_classes):
        super(Residual_Covolution, self).__init__()
        self.conv1 = nn.Conv2d(icol, ocol, kernel_size=3, stride=1, padding=12, dilation=12, bias=True)
        self.conv2 = nn.Conv2d(ocol, num_classes, kernel_size=3, stride=1, padding=12, dilation=12, bias=True)
        self.conv3 = nn.Conv2d(num_classes, ocol, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.conv4 = nn.Conv2d(ocol, icol, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        dow1 = self.conv1(x)
        dow1 = self.relu(dow1)
        seg = self.conv2(dow1)
        inc1 = self.conv3(seg)
        add1 = dow1 + self.relu(inc1)
        inc2 = self.conv4(add1)
        out = x + self.relu(inc2)
        return out, seg


class Residual_Refinement_Module(nn.Module):

    def __init__(self, num_classes):
        super(Residual_Refinement_Module, self).__init__()
        self.RC1 = Residual_Covolution(2048, 512, num_classes)
        self.RC2 = Residual_Covolution(2048, 512, num_classes)

    def forward(self, x):
        x, seg1 = self.RC1(x)
        _, seg2 = self.RC2(x)
        #return [seg1, seg1+seg2]
        return seg1 + seg2


class CFM(nn.Module):
    def __init__(self, dilation_series, padding_series, num_classes):
        super(CFM, self).__init__()

        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(2048, num_classes, kernel_size=3, stride=1,
                          padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=True)
        for i in self.bn1.parameters():
            i.requires_grad = False

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,
                                    padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                       dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilation=4)
        self.layer5 = self._make_pred_layer(CFM,
                                            [6, 12, 18, 24], [6, 12, 18, 24],
                                            num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        ds = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            ds = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=True))

        for i in ds._modules['1'].parameters():
            i.requires_grad = False

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, ds=ds))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series, num_classes):
        return block(dilation_series, padding_series, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x


class ResNet_Refine(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet_Refine, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=True)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = Residual_Refinement_Module(num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        ds = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            ds = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=True))
        for i in ds._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, ds=ds))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x


def Deeplab(num_classes=2):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    return model


def Deeplabv3(num_classes=2):
    model = ResNet_Refine(Bottleneck, [3, 4, 23, 3], num_classes)
    return model
