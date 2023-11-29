# ----------------------------------------------------
# Modified SegFormer decoder with Atrous convolutions
# https://arxiv.org/pdf/2105.15203v3
# ----------------------------------------------------


import torch
import torch.nn as nn
from math import sqrt

from models.registry import decoder_entrypoints
from utils import utils


class ConvActNorm(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dialation=1,
                    groups=1, bias=True, act=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dialation,
                        groups, bias)
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = act() if act is not None else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.norm(x)
        return x


class ASSPP(nn.Module):
    """ Depthwise Separable Atrous Spatial Pyramid Pooling with residual"""

    def __init__(self, in_features, out_features=None, dw_size=3, act=nn.Hardswish):
        super().__init__()
        dilation_rates = [1,2,4,8]
        out_features = out_features if out_features else in_features
        self.fc1 = nn.Conv2d(in_features, out_features, 1, 1, 0)
        self.conv = nn.ModuleList(
            ConvActNorm(out_features, out_features, dw_size, 1, rate, rate,
                    groups=out_features, bias=False, act=None)
            for rate in dilation_rates
        )
        self.fc2 = nn.Conv2d(int(out_features*len(dilation_rates)), out_features, 1, 1, 0, bias=False)
        self.act = act()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.cat([conv(x) for conv in self.conv], dim=1)
        x = self.fc2(self.act(x))
        return x


class AtrousDecoder(nn.Module):
    """ Modified SegFormer Decoder
    Args:
        img_size (int): Input resolution.
        num_classes (int): Number of classes for classification head.
        in_features (int): Embedding dimension of the final encoder layer.
        embed_dim (int): Decoder embedding dimension.
        depth (int): Number of encoder layers.
        fuse_op (str): Fusion operation (add|cat).
    """

    def __init__(self, img_size, num_classes, in_features, embed_dim, depth, fuse_op):
        super().__init__()
        assert fuse_op in ('add','cat'), "Invalid fusion operation!"
        self.fuse_op = fuse_op
        self.img_size = utils.to_2tuple(img_size)
        self.aux = nn.ModuleList([
            nn.Sequential(
                ASSPP(in_features//2**(depth-i-1), embed_dim),
                nn.Upsample(scale_factor=2**i, mode='bilinear', align_corners=True)
            )
            for i in range(depth-2,0,-1)
        ])
        self.merge = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_features//2**(depth-i-1), embed_dim, 1, 1, 0),
                nn.Upsample(scale_factor=2**i, mode='bilinear', align_corners=True)
            )
            for i in range(depth-1,-1,-1)
        ])
        self.fuse = nn.Conv2d(embed_dim, embed_dim, 1, 1, 0) if fuse_op == 'add' else \
                        nn.Conv2d(depth*embed_dim, embed_dim, 1, 1, 0)
        self.head = nn.Conv2d((depth-2+1)*embed_dim, num_classes, 1, 1, 0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            utils.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, x_trace):
        x_trace.insert(0, x)
        merged = []
        aux = []
        i, n = 0, len(x_trace)
        for blk, x in zip(self.merge, x_trace):
            B, L, C = x.shape
            H = W = int(sqrt(L))
            x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
            merged.append(blk(x))
            if 0 < i < n-1:
                aux.append(self.aux[i-1](x))
            i += 1
        x = torch.cat(merged, dim=1) if self.fuse_op == 'cat' else \
                torch.stack(merged).sum(dim=0)
        x = self.fuse(x)
        x = torch.cat([x, *aux], dim=1)
        x = self.head(x)
        x = nn.functional.interpolate(x, size=self.img_size, mode='bilinear')
        return x


@decoder_entrypoints.register('atrous')
def build_decoder(config):
    return AtrousDecoder(
            img_size=config.DATA.IMAGE_SIZE,
            num_classes=config.DATA.NUM_CLASSES,
            in_features=config.MODEL.DECODER.IN_FEATURES,
            embed_dim=config.MODEL.DECODER.EMBED_DIM,
            depth=config.MODEL.DECODER.DEPTH,
            fuse_op=config.MODEL.DECODER.FUSE_OP
        )
