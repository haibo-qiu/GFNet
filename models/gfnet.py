import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.range_net import ResUnet as RangeUnet
from models.polar_net import polar_net as PolarUnet
from models.kpconv.blocks import KPConv
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch_scatter

ALIGN=False
BatchNorm = nn.BatchNorm2d

def get_range_model(**kwargs):
    model = RangeUnet(**kwargs)
    return model


def get_polar_model(**kwargs):
    model = PolarUnet(**kwargs)
    return model

def resample_grid(predictions, pxpy):
    resampled = F.grid_sample(predictions, pxpy)

    return resampled


class KPClassifier(nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        super(KPClassifier, self).__init__()
        self.kpconv = KPConv(
            kernel_size=15,
            p_dim=3,
            in_channels=in_channels,
            out_channels=out_channels,
            KP_extent=1.2,
            radius=0.60,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, pxyz, pknn):
        # x should be tne concated feature Nx2C
        res = []
        for i in range(x.shape[0]):
            points = pxyz[i, ...]
            feats = x[i, ...].transpose(0, 2).squeeze()
            feats = self.kpconv(points, points, pknn[i, ...], feats)
            res.append(feats.unsqueeze(2).transpose(0, 2).unsqueeze(2))
        res = torch.cat(res, axis=0)
        res = self.relu(self.bn(res))
        return res

class GFNet(nn.Module):

    def __init__(self, ARCH, layers=34, kernal_size=1, n_class=19, flow=False, data_type=torch.float32):
        super(GFNet, self).__init__()
        range_model = get_range_model(in_channel=5, layers=layers, classes=n_class)
        polar_model = get_polar_model(grid_size=ARCH['polar']['grid_size'],
                                      n_height=ARCH['polar']['grid_size'][-1],
                                      n_class=n_class,
                                      layers=layers,
                                      kernal_size=kernal_size,
                                      fea_compre=ARCH['polar']['grid_size'][-1])
        # channels distribution
        self.channels = {18: [64, 128, 256, 512],
                         34: [64, 128, 256, 512],
                         50: [256, 512, 1024, 2048]}[layers]
        self.n_class = n_class
        self.data_type = data_type
        self.n_height = ARCH['polar']['grid_size'][-1]

        # for range net
        self.range_layer0 = range_model.layer0
        self.range_layer1 = range_model.layer1
        self.range_layer2 = range_model.layer2
        self.range_layer3 = range_model.layer3
        self.range_layer4 = range_model.layer4
        self.range_up4 = range_model.up4
        self.range_delayer4 = range_model.delayer4
        self.range_up3 = range_model.up3
        self.range_delayer3 = range_model.delayer3
        self.range_up2 = range_model.up2
        self.range_delayer2 = range_model.delayer2
        self.range_cls = range_model.cls

        # for polar net
        self.grid_size = ARCH['polar']['grid_size']
        # for method
        self.polar_preprocess = polar_model.preprocess
        self.polar_reformat_data = polar_model.reformat_data
        # for module
        self.polar_PPmodel = polar_model.PPmodel
        self.polar_compress = polar_model.fea_compression
        if kernal_size != 1:
            self.local_pool_op = torch.nn.MaxPool2d(kernal_size, stride=1, padding=(kernal_size - 1) // 2, dilation=1)
        else:
            self.local_pool_op = None

        self.polar_layer0 = polar_model.unet_model.network.layer0
        self.polar_layer1 = polar_model.unet_model.network.layer1
        self.polar_layer2 = polar_model.unet_model.network.layer2
        self.polar_layer3 = polar_model.unet_model.network.layer3
        self.polar_layer4 = polar_model.unet_model.network.layer4
        self.polar_up4 = polar_model.unet_model.network.up4
        self.polar_delayer4 = polar_model.unet_model.network.delayer4
        self.polar_up3 = polar_model.unet_model.network.up3
        self.polar_delayer3 = polar_model.unet_model.network.delayer3
        self.polar_up2 = polar_model.unet_model.network.up2
        self.polar_delayer2 = polar_model.unet_model.network.delayer2
        self.polar_cls = polar_model.unet_model.network.cls

        # flow
        self.flow = flow
        self.flow_l2_r2p = R2B_flow(self.channels[0], self.data_type)
        self.flow_l3_r2p = R2B_flow(self.channels[1], self.data_type)
        self.flow_l4_r2p = R2B_flow(self.channels[2], self.data_type)
        self.flow_l5_r2p = R2B_flow(self.channels[3], self.data_type)

        self.flow_l2_p2r = B2R_flow(self.channels[0], self.data_type)
        self.flow_l3_p2r = B2R_flow(self.channels[1], self.data_type)
        self.flow_l4_p2r = B2R_flow(self.channels[2], self.data_type)
        self.flow_l5_p2r = B2R_flow(self.channels[3], self.data_type)

        # aspp
        self.range_aspp = ASPP(in_channels=self.channels[3],
                               out_channels=256,
                               atrous_rates=(16, 32, 48))     

        self.polar_aspp = ASPP(in_channels=self.channels[3],
                               out_channels=256,
                               atrous_rates=(3, 6, 9))       

        self.kpconv_range = KPClassifier(n_class)
        self.kpconv_polar = KPClassifier(n_class)

        self.kpconv = KPClassifier(256)
        self.kpconv_cls_range = nn.Conv2d(128, n_class, kernel_size=1)
        self.kpconv_cls_polar = nn.Conv2d(128, n_class, kernel_size=1)
        self.kpconv_cls_fusion = nn.Conv2d(128, n_class, kernel_size=1)

    def forward(self, x, pt_fea, xy_ind, num_pt_each, r2p_matrix, p2r_matrix, pxpy_range, pxpypz_polar, pxyz, knns):
        # range view feature extract
        _, _, range_h, range_w = x.shape
        range_x = self.range_layer0(x)  # 1/4
        range_x2 = self.range_layer1(range_x)  # 1/4
        range_x3 = self.range_layer2(range_x2)  # 1/8
        range_x4 = self.range_layer3(range_x3)  # 1/16
        range_x5 = self.range_layer4(range_x4)  # 1/32

        # polar view preprocess
        cat_pt_fea, unq, unq_inv, batch_size, cur_dev = self.polar_preprocess(pt_fea, xy_ind, num_pt_each)
        processed_cat_pt_fea = self.polar_PPmodel(cat_pt_fea)
        # torch scatter does not support float16
        pooled_data, pooled_idx = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)
        processed_pooled_data = self.polar_compress(pooled_data)

        out_data = self.polar_reformat_data(processed_pooled_data, unq,
                                            batch_size, cur_dev, self.data_type)
        _, _, polar_h, polar_w = out_data.shape

        # feature extract
        polar_x = self.polar_layer0(out_data)  # 1/4
        polar_x2 = self.polar_layer1(polar_x)  # 1/4
        polar_x3 = self.polar_layer2(polar_x2)  # 1/8
        polar_x4 = self.polar_layer3(polar_x3)  # 1/16
        polar_x5 = self.polar_layer4(polar_x4)  # 1/32

        range_x5 =self.range_aspp(range_x5)
        polar_x5 =self.polar_aspp(polar_x5)

        self.polar_h = polar_h
        self.range_w = range_w
        self.p2r_matrix = p2r_matrix
        self.r2p_matrix = r2p_matrix

        # feature flow
        range_p4, polar_p4 = self.feature_flow(5, range_x5, polar_x5, range_x4, polar_x4,
                                               self.flow_l5_p2r, self.flow_l5_r2p,
                                               self.range_up4, self.range_delayer4,
                                               self.polar_up4, self.polar_delayer4)

        range_p3, polar_p3 = self.feature_flow(4, range_p4, polar_p4, range_x3, polar_x3,
                                               self.flow_l4_p2r, self.flow_l4_r2p,
                                               self.range_up3, self.range_delayer3,
                                               self.polar_up3, self.polar_delayer3)

        range_p2, polar_p2 = self.feature_flow(3, range_p3, polar_p3, range_x2, polar_x2,
                                               self.flow_l3_p2r, self.flow_l3_r2p,
                                               self.range_up2, self.range_delayer2,
                                               self.polar_up2, self.polar_delayer2)

        if self.flow:
            range_p2 = self.flow_l2_p2r(self.p2r_matrix.clone(),
                                        range_p2, polar_p2)
            polar_p2 = self.flow_l2_r2p(self.r2p_matrix.clone(),
                                        polar_p2, range_p2)

        range_x = self.range_cls(range_p2)
        range_x = F.interpolate(range_x, size=(range_h, range_w), mode='bilinear', align_corners=ALIGN)

        polar_x = self.polar_cls(F.pad(polar_p2, (1, 1, 0, 0), mode='circular'))
        polar_x = F.interpolate(polar_x, size=(polar_h, polar_w), mode='bilinear', align_corners=ALIGN)

        # reformat polar feature
        polar_x = polar_x.permute(0, 2, 3, 1).contiguous()
        new_shape = list(polar_x.size())[:3] + [self.n_height, self.n_class]
        polar_x = polar_x.view(new_shape)
        polar_x = polar_x.permute(0, 4, 1, 2, 3).contiguous()

        range_pred = F.grid_sample(range_x, pxpy_range, align_corners=ALIGN)
        polar_pred = F.grid_sample(polar_x, pxpypz_polar, align_corners=ALIGN).squeeze(2)

        range_fea = self.kpconv_range(range_pred, pxyz, knns)
        polar_fea = self.kpconv_polar(polar_pred, pxyz, knns)

        fusion_fea = torch.cat([range_fea, polar_fea], dim=1)
        fusion_fea = self.kpconv(fusion_fea, pxyz, knns)

        range_pred_kpconv = self.kpconv_cls_range(range_fea)
        polar_pred_kpconv = self.kpconv_cls_polar(polar_fea)
        fusion_pred_kpconv = self.kpconv_cls_fusion(fusion_fea)

        return fusion_pred_kpconv, range_pred_kpconv, polar_pred_kpconv, range_x, polar_x

    def feature_flow(self, level,
                     range_p_pre, polar_p_pre,
                     range_x, polar_x,
                     flow_p2r, flow_r2p,
                     range_up, range_delayer,
                     polar_up, polar_delayer):
        # flow on level
        if self.flow:
            fused_range_p_pre = flow_p2r(self.p2r_matrix.clone(),
                                         range_p_pre, polar_p_pre)
            # factor changed to be same or [1, factor] for horizontal conv
            fused_polar_p_pre = flow_r2p(self.r2p_matrix.clone(),
                                         polar_p_pre, range_p_pre)
        else:
            fused_range_p_pre, fused_polar_p_pre = range_p_pre, polar_p_pre

        # for range
        range_p = range_up(F.interpolate(fused_range_p_pre, range_x.shape[-2:], mode='bilinear', align_corners=ALIGN))
        range_p = torch.cat([range_p, range_x], dim=1)
        range_p = range_delayer(range_p)

        # for polar
        polar_p = F.interpolate(fused_polar_p_pre, polar_x.shape[-2:], mode='bilinear', align_corners=ALIGN)
        polar_p = F.pad(polar_p, (1, 1, 0, 0), mode='circular')
        polar_p = polar_up(polar_p)
        polar_p = torch.cat([polar_p, polar_x], dim=1)
        polar_p = polar_delayer(polar_p)

        return range_p, polar_p

class B2R_flow(nn.Module):
    def __init__(self, fea_dim, data_type):
        super(B2R_flow, self).__init__()
        self.fea_dim = fea_dim
        self.data_type = data_type

        self.fusion = nn.Sequential(
            nn.Conv2d(fea_dim * 2, fea_dim, kernel_size=3, padding=1, bias=False),
            BatchNorm(fea_dim),
            nn.ReLU(inplace=True)
        )
        self.attention = nn.Sequential(
            nn.Conv2d(fea_dim, fea_dim, kernel_size=3, padding=1, bias=False),
            BatchNorm(fea_dim),
            nn.Softmax(dim=1)
        )
    def forward(self, flow_matrix, range_fea, polar_fea):
        """
        range_fea: [N, C1, 64, 2048]
        polar_fea: [N, C2, 480, 360]
        flow_matrix: [N, 64, 2048, 2], need to be [-1, 1] for grid sample
        """
        # rescale the flow matrix
        _, _, H, W = range_fea.shape
        N, C, _, _ = polar_fea.shape
        # because for F.grid_sample, i,j,k index w,h,d (i.e., reverse order)
        flow_matrix = torch.flip(flow_matrix, dims=[-1])
        flow_matrix_scaled = F.interpolate(flow_matrix.permute(0, 3, 1, 2).contiguous().float(),
                                           (H, W), mode='nearest')  # N*2*H*W
        flow_matrix_scaled = flow_matrix_scaled.permute(0, 2, 3, 1).contiguous() # N*H*W*2
        flow_fea = F.grid_sample(polar_fea, flow_matrix_scaled, padding_mode='zeros', align_corners=ALIGN) # N*C*H*W

        fea = torch.cat((range_fea, flow_fea), dim=1)
        res = self.fusion(fea)
        res = res * self.attention(res)
        fea  = range_fea + res

        return fea

class R2B_flow(nn.Module):

    def __init__(self, fea_dim, data_type):
        super(R2B_flow, self).__init__()
        self.fea_dim = fea_dim
        self.data_type=data_type

        self.fusion = nn.Sequential(
            nn.Conv2d(fea_dim * 2, fea_dim, kernel_size=3, padding=(1, 0), bias=False),
            BatchNorm(fea_dim),
            nn.ReLU(inplace=True)
        )
        self.attention = nn.Sequential(
            nn.Conv2d(fea_dim, fea_dim, kernel_size=3, padding=(1, 0), bias=False),
            BatchNorm(fea_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, flow_matrix, polar_fea, range_fea):
        """
        range_fea: [N, C1, 64, 2048]
        polar_fea: [N, C2, 480, 360]
        flow_matrix: [N, 480, 360, 32, 2]
        """
        range_fea_5d = range_fea.unsqueeze(2)

        _, _, H, W = polar_fea.shape
        N, C, _, _ = range_fea.shape

        flow_matrix = torch.flip(flow_matrix, dims=[-1])
        N, h, w, K, c = flow_matrix.shape
        flow_matrix = flow_matrix.view(N, h, w, K * c).permute(0, 3, 1, 2).contiguous()
        flow_matrix_scaled = F.interpolate(flow_matrix.float(), (H, W), mode='nearest')
        flow_matrix_scaled = flow_matrix_scaled.permute(0, 2, 3, 1).view(N, H, W, K, c)
        flow_matrix_scaled = F.pad(flow_matrix_scaled, pad=(0, 1), mode='constant', value=0.0)
        flow_fea = F.grid_sample(range_fea_5d, flow_matrix_scaled, padding_mode='zeros', align_corners=ALIGN) # N*C*H*W*K

        flow_fea = torch.max(flow_fea, dim=-1)[0] # N*C*H*W

        fea = torch.cat((polar_fea, flow_fea), dim=1)
        fea = F.pad(fea, (1, 1, 0, 0), mode='circular')
        res = self.fusion(fea)
        res = res * self.attention(F.pad(res, (1, 1, 0, 0), mode='circular'))
        fea  = polar_fea + res

        return fea

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=ALIGN)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, atrous_rates=(12, 24, 36)):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, in_channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
