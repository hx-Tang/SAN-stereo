import torch
import torch.nn as nn

from nets.sa.modules import Subtraction, Subtraction2, Aggregation


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc

# pure SAN
class GAM0(nn.Module):
    def __init__(self, in_planes, rel_planes, out_planes, kernel_size=3, stride=1, dilation=1):
        super(GAM0, self).__init__()
        self.kernel_size, self.stride = kernel_size, stride
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_w1 = nn.Sequential(nn.BatchNorm2d(rel_planes + 2), nn.ReLU(inplace=True),
                                     nn.Conv2d(rel_planes + 2, rel_planes, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(rel_planes), nn.ReLU(inplace=True),
                                     nn.Conv2d(rel_planes, 1, kernel_size=1))
        self.conv_p = nn.Conv2d(2, 2, kernel_size=1)
        self.subtraction = Subtraction(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation,pad_mode=1)
        self.subtraction2 = Subtraction2(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation,pad_mode=1)
        self.aggregation = Aggregation(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation,pad_mode=1)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        x1, x2, x = self.conv1(x), self.conv2(x), self.conv3(x)
        p = self.conv_p(position(x.shape[2], x.shape[3], x.is_cuda))
        w1 = self.softmax(self.conv_w1(torch.cat([self.subtraction2(x1, x2), self.subtraction(p).repeat(x.shape[0], 1, 1, 1)], 1)))
        x = self.aggregation(x, w1)
        return x

# ga_type = 1 SAN; else GANv1
class GAM(nn.Module):
    def __init__(self, ga_type, in_planes, rel_planes, out_planes, kernel_size=3, stride=1, dilation=1):
        super(GAM, self).__init__()
        self.kernel_size, self.stride, self.ga_type= kernel_size, stride, ga_type
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_w1 = nn.Sequential(nn.BatchNorm2d(rel_planes + 2), nn.ReLU(inplace=True),
                                     nn.Conv2d(rel_planes + 2, rel_planes, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(rel_planes), nn.ReLU(inplace=True),
                                     nn.Conv2d(rel_planes, 1, kernel_size=1))
        self.conv_p = nn.Conv2d(2, 2, kernel_size=1)
        self.subtraction = Subtraction(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation,pad_mode=1)
        self.subtraction2 = Subtraction2(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation,pad_mode=1)
        self.aggregation = Aggregation(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation,pad_mode=1)
        if self.ga_type == 0:
            self.convg1 = nn.Conv2d(in_planes*2, rel_planes, kernel_size=1)
            self.convg2 = nn.Conv2d(in_planes*2, rel_planes, kernel_size=1)
            self.conv_w2 = nn.Sequential(nn.BatchNorm2d(rel_planes + 2), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes + 2, rel_planes, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(rel_planes), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes, 1, kernel_size=1))
            self.conv_pg = nn.Conv2d(2, 2, kernel_size=1)
            self.subtractiong = Subtraction(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation,pad_mode=1)
            self.subtractiong2 = Subtraction2(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation,pad_mode=1)
            self.aggregationg = Aggregation(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation,pad_mode=1)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x, g):
        x1, x2, x = self.conv1(x), self.conv2(x), self.conv3(x)
        p = self.conv_p(position(x.shape[2], x.shape[3], x.is_cuda))
        w1 = self.softmax(self.conv_w1(torch.cat([self.subtraction2(x1, x2), self.subtraction(p).repeat(x.shape[0], 1, 1, 1)], 1)))
        x = self.aggregation(x, w1)
        if self.ga_type == 0:
            g1, g2 = self.convg1(g), self.convg2(g)
            pg = self.conv_pg(position(g.shape[2], g.shape[3], g.is_cuda))
            w2 = self.softmax(self.conv_w2(torch.cat([self.subtractiong2(g1, g2), self.subtractiong(pg).repeat(g.shape[0], 1, 1, 1)], 1)))
            x = self.aggregationg(x, w2)
        return x

# GAN v2
class GAM2(nn.Module):
    def __init__(self, in_planes, rel_planes, out_planes, kernel_size=3, stride=1, dilation=1):
        super(GAM2, self).__init__()
        self.kernel_size, self.stride = kernel_size, stride
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.convg1 = nn.Conv2d(128, rel_planes, kernel_size=1)
        self.convg2 = nn.Conv2d(128, rel_planes, kernel_size=1)
        self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes*2 + 2), nn.ReLU(inplace=True),
                                     nn.Conv2d(rel_planes*2 + 2, rel_planes, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(rel_planes), nn.ReLU(inplace=True),
                                     nn.Conv2d(rel_planes, 1, kernel_size=1))
        self.conv_p = nn.Conv2d(2, 2, kernel_size=1)
        self.subtraction = Subtraction(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation,pad_mode=1)
        self.subtraction2 = Subtraction2(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation,pad_mode=1)
        self.aggregation = Aggregation(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation,pad_mode=1)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x, g):
        x1, x2, x = self.conv1(x), self.conv2(x), self.conv3(x)
        g1,g2 = self.convg1(g), self.convg2(g)
        p = self.conv_p(position(x.shape[2], x.shape[3], x.is_cuda))
        w = self.softmax(self.conv_w(torch.cat([self.subtraction2(x1, x2), self.subtraction2(g1, g2), self.subtraction(p).repeat(x.shape[0], 1, 1, 1)], 1)))
        x = self.aggregation(x, w)
        return x

# GAN v2.2
class GAM3(nn.Module):
    def __init__(self, in_planes, rel_planes, out_planes, kernel_size=3, stride=1, dilation=1):
        super(GAM3, self).__init__()
        self.kernel_size, self.stride = kernel_size, stride
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.convg = nn.Conv2d(128, rel_planes, kernel_size=1)
        self.conv_w1 = nn.Sequential(nn.BatchNorm2d(rel_planes + 2), nn.ReLU(inplace=True),
                                     nn.Conv2d(rel_planes + 2, rel_planes, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(rel_planes), nn.ReLU(inplace=True),
                                     nn.Conv2d(rel_planes, 1, kernel_size=1))
        self.conv_w2 = nn.Sequential(nn.BatchNorm2d(rel_planes), nn.ReLU(inplace=True),
                                    nn.Conv2d(rel_planes, 1, kernel_size=1))
        self.conv_p = nn.Conv2d(2, 2, kernel_size=1)
        self.subtraction = Subtraction(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation,pad_mode=1)
        self.subtraction2 = Subtraction2(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation,pad_mode=1)
        self.aggregation = Aggregation(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation,pad_mode=1)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x, g):
        x1, x2, x = self.conv1(x), self.conv2(x), self.conv3(x)
        g = self.convg(g)
        p = self.conv_p(position(x.shape[2], x.shape[3], x.is_cuda))
        w1 = self.softmax(self.conv_w1(torch.cat([self.subtraction2(x1, x2), self.subtraction(p).repeat(x.shape[0], 1, 1, 1)], 1)))
        w2 = self.softmax(self.conv_w2(self.subtraction(g)))
        x = self.aggregation(x, w1)
        x = self.aggregation(x, w2)
        return x

# GAN v2.1
class GAM4(nn.Module):
    def __init__(self, in_planes, rel_planes, out_planes, kernel_size=3, stride=1, dilation=1):
        super(GAM4, self).__init__()
        self.kernel_size, self.stride = kernel_size, stride
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.convg = nn.Conv2d(128, rel_planes, kernel_size=1)
        self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes*2 + 2), nn.ReLU(inplace=True),
                                     nn.Conv2d(rel_planes*2 + 2, rel_planes, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(rel_planes), nn.ReLU(inplace=True),
                                     nn.Conv2d(rel_planes, 1, kernel_size=1))
        self.conv_p = nn.Conv2d(2, 2, kernel_size=1)
        self.subtraction = Subtraction(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation,pad_mode=1)
        self.subtraction2 = Subtraction2(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation,pad_mode=1)
        self.aggregation = Aggregation(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation,pad_mode=1)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x, g):
        x1, x2, x = self.conv1(x), self.conv2(x), self.conv3(x)
        g = self.convg(g)
        p = self.conv_p(position(x.shape[2], x.shape[3], x.is_cuda))
        w = self.softmax(self.conv_w(torch.cat([self.subtraction2(x1, x2), self.subtraction(p).repeat(x.shape[0], 1, 1, 1), self.subtraction(g)], 1)))
        x = self.aggregation(x, w)
        return x

# GAN block
class GAABottleneck(nn.Module):
    def __init__(self, ga_type, in_planes=64, rel_planes=4, out_planes=64, kernel_size=7, stride=1):
        super(GAABottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_planes)
        if ga_type == 0:
            self.sam = GAM2(in_planes, rel_planes, out_planes, kernel_size, stride)
        else:
            self.sam = GAM(0, in_planes, rel_planes, out_planes, kernel_size, stride)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, g):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.sam(out, g)))
        out = self.bn3(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

# SAN block
class GAABottleneck0(nn.Module):
    def __init__(self, in_planes=64, rel_planes=4, out_planes=64, kernel_size=3, stride=1):
        super(GAABottleneck0, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.sam = GAM0(in_planes, rel_planes, out_planes, kernel_size, stride)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.sam(out)))
        out = self.bn3(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

# if __name__ == '__main__':
    # net = sam(sa_type=0, layers=(3, 4, 6, 8, 3), kernels=[3, 7, 7, 7, 7], num_classes=1000).cuda().eval()
    # # print(net)
    # y = net(torch.randn(4, 3, 224, 224).cuda())
    # print(y.size())
