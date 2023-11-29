# ---------------------------------------------------
# Modified SegFormer decoder with Ghost convolutions
# https://arxiv.org/pdf/2105.15203
# ---------------------------------------------------


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


class GhostConv(nn.Module):
    """Ghost module as defined in:
    GhostNet: More Features from Cheap Operations, Han et al., 2020
    https://arxiv.org/abs/1911.11907
    Adapted from:
    https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnet_pytorch/ghostnet.py
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.primary_conv = ConvActNorm(in_channels, out_channels, 1, 1, 0,
                                bias=False, act=nn.Hardswish)
        self.cheap_op = ConvActNorm(out_channels, out_channels, 3, 1, 1, 
                                groups=out_channels, bias=False, act=nn.Hardswish)
        self.secondary_conv = ConvActNorm(2*out_channels, out_channels, 1, 1, 0, 
                                    bias=False, act=nn.Hardswish)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_op(x1)
        x = torch.cat([x1,x2], dim=1)
        x = self.secondary_conv(x)
        return x


class ConvDecoder(nn.Module):
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
        self.merge = nn.ModuleList([
            nn.Sequential(
                GhostConv(in_features//2**(depth-i-1), embed_dim),
                nn.Upsample(scale_factor=2**i, mode='bilinear', align_corners=True)
            )
            for i in range(depth-1,-1,-1)
        ])
        self.fuse = nn.Conv2d(embed_dim, embed_dim, 1, 1, 0) if fuse_op == 'add' else \
                        nn.Conv2d(depth*embed_dim, embed_dim, 1, 1, 0)
        self.head = nn.Conv2d(embed_dim, num_classes, 1, 1, 0)
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
        for blk, x in zip(self.merge, x_trace):
            B, L, C = x.shape
            H = W = int(sqrt(L))
            x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
            merged.append(blk(x))
        x = torch.cat(merged, dim=1) if self.fuse_op == 'cat' else \
                torch.stack(merged).sum(dim=0)
        x = self.head(self.fuse(x))
        x = nn.functional.interpolate(x, size=self.img_size, mode='bilinear')
        return x


@decoder_entrypoints.register('conv')
def build_decoder(config):
    return ConvDecoder(
            img_size=config.DATA.IMAGE_SIZE,
            num_classes=config.DATA.NUM_CLASSES,
            in_features=config.MODEL.DECODER.IN_FEATURES,
            embed_dim=config.MODEL.DECODER.EMBED_DIM,
            depth=config.MODEL.DECODER.DEPTH,
            fuse_op=config.MODEL.DECODER.FUSE_OP
        )
