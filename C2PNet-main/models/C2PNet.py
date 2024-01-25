import torch
import torch.nn as nn


#定义了一个标准的二维卷积层，它是构建更复杂网络模块的基本单元。
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


#实现了一个通道注意力层（Channel Attention Layer），用于增强网络的特征表示能力。它通过关注不同通道的重要性来提升模型的性能。
class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

#定义了一个物理解耦单元（Physical Decoupling Unit），用于处理图像中的物理特性，如光照和反射。这是模型特有的设计，用于提高去雾任务的性能。
class PDU(nn.Module):  # physical block
    def __init__(self, channel):
        super(PDU, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ka = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.td = nn.Sequential(
            default_conv(channel, channel, 3),
            default_conv(channel, channel // 8, 3),
            nn.ReLU(inplace=True),
            default_conv(channel // 8, channel, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        a = self.avg_pool(x)
        a = self.ka(a)
        t = self.td(x)
        j = torch.mul((1 - t), a) + torch.mul(t, x)
        return j

#定义了模型中的基本块，它包括卷积层、激活函数、通道注意力层和物理解耦单元。
class Block(nn.Module):  # origin
    def __init__(self, conv, dim, kernel_size, ):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.pdu = PDU(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.pdu(res)
        res += x
        return res

#将多个 Block 组合在一起，并在最后添加一个卷积层。这样的设计使得模型可以学习到更深层次的特征。
class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res

#这是模型的主体，它包含预处理层、多个 Group 层、通道注意力和物理解耦单元，以及后处理层。这个架构设计旨在通过多阶段处理来有效地去除图像中的雾霾。
class C2PNet(nn.Module):
    def __init__(self, gps, blocks, conv=default_conv):
        super(C2PNet, self).__init__()
        #定义了网络的预处理部分，使用一个卷积层将输入通道从3（彩色图像的RGB通道）转换到 self.dim 指定的通道数。
        self.gps = gps
        self.dim = 64
        kernel_size = 3
        #定义了网络的预处理部分，使用一个卷积层将输入通道从3（彩色图像的RGB通道）转换到 self.dim 指定的通道数。
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps == 3
        #这些行创建了三个 Group 实例，每个 Group 包含多个 Block 块，用于处理图像的不同方面。每个 Group 可以看作是网络中的一个子模块。
        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g3 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])
        self.pdu = PDU(self.dim)

        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1):
        x = self.pre(x1)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)
        w = self.ca(torch.cat([res1, res2, res3], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
        out = w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3
        out = self.pdu(out)
        x = self.post(out)
        return x + x1


if __name__ == "__main__":
    net = C2PNet(gps=3, blocks=19)
    print(net)
