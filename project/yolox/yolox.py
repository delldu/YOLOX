#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .darknet import CSPDarknet, BaseConv, CSPLayer
import pdb
import todos


def torch_nn_arange(x):
    if x.dim() == 2:
        B, C = x.size()
        a = torch.arange(x.nelement()) / x.nelement()
        a = a.to(x.device)
        return a.view(B, C)

    if x.dim() == 3:
        B, C, HW = x.size()
        a = torch.arange(x.nelement()) / x.nelement()
        a = a.to(x.device)
        return a.view(B, C, HW)

    B, C, H, W = x.size()
    a = torch.arange(x.nelement()) / x.nelement()
    a = a.to(x.device)
    return a.view(B, C, H, W)


class YOLOX(nn.Module):
    def __init__(self):
        super().__init__()
        self.MAX_H = 640
        self.MAX_W = 640
        self.MAX_TIMES = 1
        self.backbone = YOLOPAFPN()
        self.head = YOLOXHead(80)

        self.load_weights()
        # from ggml_engine import create_network
        # create_network(self)
        # print(self)
        # torch.save(self.state_dict(), "/tmp/yolox_l.pth")

    def forward(self, x):
        B, C, H, W = x.size()
        x = F.interpolate(x, size=(self.MAX_H, self.MAX_W), mode="nearest")
        x = x * 255.0
        # tensor [x] size: [1, 3, 640, 640], min: 0.0, max: 255.0, mean: 128.074554

        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)
        # fpn_outs is tuple: len = 3
        #     tensor [item] size: [1, 256, 80, 80], min: -0.278465, max: 8.218046, mean: 0.038865
        #     tensor [item] size: [1, 512, 40, 40], min: -0.278465, max: 11.195083, mean: 0.11869
        #     tensor [item] size: [1, 1024, 20, 20], min: -0.278465, max: 11.274212, mean: 0.179487

        detect_result = self.head(fpn_outs)

        # Scale x1y1, x2y2
        sh = H / self.MAX_H
        sw = W / self.MAX_W

        detect_result[:, :, 0:1] = detect_result[:, :, 0:1] * sw  # x1
        detect_result[:, :, 1:2] = detect_result[:, :, 1:2] * sh  # y1
        detect_result[:, :, 2:3] = detect_result[:, :, 2:3] * sw  # x2
        detect_result[:, :, 3:4] = detect_result[:, :, 3:4] * sh  # y2

        # (x1, y1, x2, y2, obj_score*cls_score, cls_id)
        # tensor [detect_result] size: [1, 8400, 6], min: -51.966671, max: 866.434814, mean: 194.278854

        return detect_result

    def load_weights(self, model_path="models/yolox_l.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        self.load_state_dict(torch.load(checkpoint))


class YOLOXHead(nn.Module):
    def __init__(self,
        num_classes=80,
        in_channels=[256, 512, 1024],
    ):
        super().__init__()

        self.stems = nn.ModuleList()
        self.num_classes = num_classes
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        for i in range(len(in_channels)):  # 3
            self.stems.append(
                BaseConv(
                    in_channels=in_channels[i],
                    out_channels=256,
                    ksize=1,
                    stride=1,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        BaseConv(in_channels=256, out_channels=256, ksize=3, stride=1),
                        BaseConv(in_channels=256, out_channels=256, ksize=3, stride=1),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        BaseConv(in_channels=256, out_channels=256, ksize=3, stride=1),
                        BaseConv(in_channels=256, out_channels=256, ksize=3, stride=1),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(in_channels=256, out_channels=self.num_classes, kernel_size=1, stride=1, padding=0)
            )
            self.reg_preds.append(
                nn.Conv2d(in_channels=256, out_channels=4, kernel_size=1, stride=1, padding=0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
            )

        g, s = self.init_grids_strides()
        self.register_buffer('grids', g)
        self.register_buffer('strides', s)

    def forward(self, xin):
        # xin is tuple: len = 3
        #     tensor [item] size: [1, 256, 80, 80], min: -0.278465, max: 8.218053, mean: 0.038865
        #     tensor [item] size: [1, 512, 40, 40], min: -0.278465, max: 11.195078, mean: 0.11869
        #     tensor [item] size: [1, 1024, 20, 20], min: -0.278465, max: 11.274218, mean: 0.179487

        outputs = []
        for k, (cls_conv, reg_conv, x) in enumerate(zip(self.cls_convs, self.reg_convs, xin)):
            x = self.stems[k](x)
            cls_feat = cls_conv(x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
            output = output.flatten(start_dim=2)

            outputs.append(output)

        # outputs is list: len = 3
        #     tensor [item] size: [1, 85, 80*80], min: -1.856502, max: 3.459077, mean: 0.054403
        #     tensor [item] size: [1, 85, 40*40], min: -2.040908, max: 3.398808, mean: 0.060484
        #     tensor [item] size: [1, 85, 20*20], min: -1.647478, max: 3.015006, mean: 0.062889

        # [batch, n_anchors_all, 85]
        # outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
        outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)

        # tensor [outputs] size: [1, 8400, 85], min: -2.040908, max: 3.459077, mean: 0.055966
        return self.decode_outputs(outputs)

    def init_grids_strides(self):
        grids = []
        strides = []
        # strides=[8, 16, 32],
        for stride in [8, 16, 32]:
            hsize = 640//stride
            wsize = 640//stride

            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)], indexing="ij")
            # yv.size() -- [80, 80], xv.size() -- [80, 80]
            # (Pdb) yv
            # tensor([[ 0,  0,  0,  ...,  0,  0,  0],
            #         [ 1,  1,  1,  ...,  1,  1,  1],
            #         [ 2,  2,  2,  ...,  2,  2,  2],
            #         ...,
            #         [77, 77, 77,  ..., 77, 77, 77],
            #         [78, 78, 78,  ..., 78, 78, 78],
            #         [79, 79, 79,  ..., 79, 79, 79]])
            # torch.stack((xv, yv), 2) -- [80, 80, 2]
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            # grid.size() -- [1, 6400, 2]
            grids.append(grid)
            shape = grid.shape[:2] # [1, 6400]
            strides.append(torch.full((*shape, 1), stride))

        # grids is list: len = 3
        #     tensor [item] size: [1, 6400, 2], min: 0.0, max: 79.0, mean: 39.5
        #     tensor [item] size: [1, 1600, 2], min: 0.0, max: 39.0, mean: 19.5
        #     tensor [item] size: [1, 400, 2], min: 0.0, max: 19.0, mean: 9.5
        # strides is list: len = 3
        #     tensor [item] size: [1, 6400, 1], min: 8.0, max: 8.0, mean: 8.0
        #     tensor [item] size: [1, 1600, 1], min: 16.0, max: 16.0, mean: 16.0
        #     tensor [item] size: [1, 400, 1], min: 32.0, max: 32.0, mean: 32.0
        grids = torch.cat(grids, dim=1)
        strides = torch.cat(strides, dim=1)
        # tensor [grids] size: [1, 8400, 2], min: 0.0, max: 79.0, mean: 34.261906
        # tensor [strides] size: [1, 8400, 1], min: 8.0, max: 32.0, mean: 10.666667
        return grids, strides


    def decode_outputs(self, outputs):
        # tensor [outputs] size: [1, 8400, 85], min: -2.040908, max: 3.459077, mean: 0.055966

        # (x1, y1, w2, h2, obj_score, class_scores 80 ...)
        # ==> (x1, y1, x2, y2, obj_score * cls_score, cls_id)
        x1y1 = (outputs[:, :, 0:2] + self.grids) * self.strides
        w2h2 = torch.exp(outputs[..., 2:4]) * self.strides
        obj_score = outputs[:, :, 4:5] # 4
        cls_score = outputs[:, :, 5:] # 5 ...

        # Convert to boxes ...
        x2y2 = x1y1 + w2h2/2.0
        x1y1 = x1y1 - w2h2/2.0

        # Find max class id
        cls_score, cls_id = torch.max(cls_score, 2, keepdim=True)
        outputs = torch.cat([x1y1, x2y2, obj_score * cls_score, cls_id.float()], dim=2)

        return outputs # [1, 8400, 6]


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(self, in_channels=[256, 512, 1024]):
        super().__init__()
        self.backbone = CSPDarknet()
        self.in_features = ("dark3", "dark4", "dark5")

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(in_channels[2], in_channels[1], 1, 1)
        self.C3_p4 = CSPLayer(2 * in_channels[1], in_channels[1], 3, shortcut=False)  # cat
        self.reduce_conv1 = BaseConv(in_channels[1], in_channels[0], 1, 1)
        self.C3_p3 = CSPLayer(2 * in_channels[0], in_channels[0], 3, shortcut=False)

        # bottom-up conv
        self.bu_conv2 = BaseConv(in_channels[0], in_channels[0], 3, 2)
        self.C3_n3 = CSPLayer(2 * in_channels[0], in_channels[1], 3, shortcut=False)

        # bottom-up conv
        self.bu_conv1 = BaseConv(in_channels[1], in_channels[1], 3, 2)
        self.C3_n4 = CSPLayer(2 * in_channels[1], in_channels[2], 3, shortcut=False)

    def forward(self, input):
        # tensor [input] size: [1, 3, 640, 640], min: 0.0, max: 255.0, mean: 124.46978
        dark3, dark4, dark5 = self.backbone(input)
        # tensor [dark3] size: [1, 256, 80, 80], min: -0.278465, max: 8.673933, mean: -0.011218
        # tensor [dark4] size: [1, 512, 40, 40], min: -0.278465, max: 11.121074, mean: 0.053395
        # tensor [dark5] size: [1, 1024, 20, 20], min: -0.278465, max: 10.520874, mean: 0.174783

        dark5_fpn_out = self.lateral_conv0(dark5)  # 1024->512/32
        f_out0 = self.upsample(dark5_fpn_out)  # 512/16
        f_out0 = torch.cat([f_out0, dark4], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, dark3], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, dark5_fpn_out], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        # outputs is tuple: len = 3
        #     tensor [item] size: [1, 256, 80, 80], min: -0.278465, max: 8.218046, mean: 0.038865
        #     tensor [item] size: [1, 512, 40, 40], min: -0.278465, max: 11.195083, mean: 0.11869
        #     tensor [item] size: [1, 1024, 20, 20], min: -0.278465, max: 11.274212, mean: 0.179487

        return outputs
