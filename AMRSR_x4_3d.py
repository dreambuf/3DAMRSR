import torch
import torch.nn as nn
from CBAM_3D import *
from torchsummary import summary


###3D AMRSR model structure
def default_conv(in_channelss, out_channels, kernel_size, bias=True):
    return nn.Conv3d(
        in_channelss, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class ResBlockb(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlockb, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CBAM_Block3d(channel=64))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class AMRSR(nn.Module):
    def __init__(self, conv=default_conv):
        super(AMRSR, self).__init__()
        n_feats = 64
        kernel_size = 3
        n_resblock = 16
        act = nn.ReLU(True)
        res_scale = 1
        scale = 4

        self.head = nn.Sequential(conv(1, n_feats, kernel_size))

        self.bs1 = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.bs2 = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.bs3 = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.bs4 = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.bs5 = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.bs6 = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.bs7 = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.bs8 = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.bs9 = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.bs10 = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.bs11 = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.bs12 = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.bs13 = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.bs14 = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.bs15 = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.bs16 = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.tbs = nn.Sequential(conv(1, n_feats, kernel_size))
        self.tbs1 = ResBlockb(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.tbs2 = ResBlockb(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.tbs3 = ResBlockb(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.tbs4 = ResBlockb(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.tbs5 = ResBlockb(conv, n_feats, kernel_size, act=act, res_scale=res_scale)

        self.tbs10 = nn.Sequential(conv(n_feats, 1, kernel_size))


        modules_tail = [
            nn.Upsample(scale_factor=scale, mode='trilinear'),
            conv(n_feats, 1, kernel_size)
            ]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = x.contiguous()
        x = self.head(x)
        res = self.bs1(x)
        res = self.bs2(res)
        res = self.bs3(res)
        res = self.bs4(res)
        res = self.bs5(res)
        res = self.bs6(res)
        res = self.bs7(res)
        res = self.bs8(res)
        res = self.bs9(res)
        res = self.bs10(res)
        res = self.bs11(res)
        res = self.bs12(res)
        res = self.bs13(res)
        res = self.bs14(res)
        res = self.bs15(res)
        res = self.bs16(res)

        res += x

        x = self.tail(res)
        t = self.tbs(x)
        tbs = self.tbs1(t)
        tbs = self.tbs2(tbs)
        tbs = self.tbs3(tbs)
        c = t + tbs
        x = self.tbs10(c)
        return x
