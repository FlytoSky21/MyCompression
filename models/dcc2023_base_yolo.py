# -*- coding:utf-8 -*-
# @Time: 2023/11/19 18:02
# @Author: TaoFei
# @FileName: dcc2023.py
# @Software: PyCharm

from compressai.models import CompressionModel
import math
import warnings

import torch
import torch.nn as nn
import re
import torch.nn.functional as F

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
# from models.yolov3_models import load_model
from pytorchyolo import detect, my_models


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


class DCC2023Model(CompressionModel):
    def __init__(self, Cs=256, N=192, M1=64, M2=128, M=192):
        super(DCC2023Model, self).__init__(entropy_bottleneck_channels=N)
        yolov3 = my_models.load_model("/home/adminroot/taofei/DCC2023fuxian/config/yolov3.cfg",
                                      "/home/adminroot/taofei/DCC2023fuxian/config/yolov3.weights")
        self.yolov3_front = my_models.load_front_model("/home/adminroot/taofei/DCC2023fuxian/config/yolov3_front.cfg")
        self.yolov3_back = my_models.load_back_model("/home/adminroot/taofei/DCC2023fuxian/config/yolov3_back.cfg")

        yolov3_dict = yolov3.state_dict()
        yolov3_front_dict = self.yolov3_front.state_dict()
        yolov3_front_pretrained = {k: v for k, v in yolov3_dict.items() if k in yolov3_front_dict}
        yolov3_front_dict.update(yolov3_front_pretrained)
        self.yolov3_front.load_state_dict(yolov3_front_dict)

        yolov3_back_dict = self.yolov3_back.state_dict()
        new_yolov3_back_dict = {}
        for k, v in yolov3_dict.items():
            # 将键中的数字-13
            k = re.sub(r'(\d+)', lambda x: str(int(x.group(1)) - 13), k)
            if k in yolov3_back_dict:
                new_yolov3_back_dict[k] = v

        yolov3_back_dict.update(new_yolov3_back_dict)
        self.yolov3_back.load_state_dict(yolov3_back_dict)

        # self.yolov3_front = torch.nn.Sequential(*list(self.yolov3.module_list)[:13])
        # self.yolov3_back = torch.nn.Sequential(*list(self.yolov3.module_list)[13:])

        # Freeze the parameters of yolo_front and yolo_back
        for param in self.yolov3_front.parameters():
            param.requires_grad = False
        for param in self.yolov3_back.parameters():
            param.requires_grad = False

        self.gs_a = nn.Sequential(
            conv(Cs + 3, N, 5, 1),
            GDN(N),
            conv(N, N, 5, 1),
            GDN(N),
            conv(N, M1, 5, 2),
        )
        self.gs_s = nn.Sequential(
            deconv(M1, N, 5, 1),
            GDN(N, inverse=True),
            deconv(N, N, 5, 1),
            GDN(N, inverse=True),
            deconv(N, Cs, 5, 2),
        )
        self.hs_a = nn.Sequential(
            conv(M1, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.hs_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M1, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.gx_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )
        self.gx_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M2),
        )
        self.hx_a = nn.Sequential(
            conv(M2, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.hx_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M2, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.gaussian_conditional = GaussianConditional(None)
        self.Cs = Cs
        self.N = N
        self.M1 = M1
        self.M2 = M2
        self.M = M

    def forward(self, x, x_lr):
        # baselayer
        img_size = x.size(2)
        s = self.yolov3_front(x)
        # f = torch.cat([s, x_lr], dim=1)
        # y1 = self.gs_a(f)
        # z1 = self.hs_a(torch.abs(y1))
        # z1_hat, z1_likelihoods = self.entropy_bottleneck(z1)
        # scales_hat_z1 = self.hs_s(z1_hat)
        # y1_hat, y1_likelihoods = self.gaussian_conditional(y1, scales_hat_z1)
        # s_hat = self.gs_s(y1_hat)
        t_hat = self.yolov3_back(s, img_size)

        # enhance layer
        # y2 = self.gx_a(x)
        # z2 = self.hx_a(torch.abs(y2))
        # z2_hat, z2_likelihoods = self.entropy_bottleneck(z2)
        # scales_hat_z2 = self.hx_s(z2_hat)
        # y2_hat, y2_likelihoods = self.gaussian_conditional(y2, scales_hat_z2)
        # x_hat = self.gx_s(torch.cat([y2_hat, y1_hat], dim=1))

        return {
            # "x_hat": x_hat,
            # "base_likelihoods": {"y1": y1_likelihoods, "z1": z1_likelihoods},
            # # "enhance_likelihoods": {"y2": y2_likelihoods, "z2": z2_likelihoods},
            # "s": s,
            # "s_hat": s_hat,
            "t_hat": t_hat,
        }
