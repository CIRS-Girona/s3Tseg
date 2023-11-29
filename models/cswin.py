# -----------------------------------------------
# Modified by Hayat Rajani (hayat.rajani@udg.edu)
# CSWin Transformer
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Xiaoyi Dong
# -----------------------------------------------


import torch
import torch.nn as nn

from math import sqrt
from utils import utils

from models.registry import encoder_entrypoints
from models.registry import decoder_entrypoints


def channel_shuffle(x, groups):
    B, C, H, W = x.shape
    # reshape
    x = x.view(B, groups, C // groups, H, W)
    # transpose
    x = x.transpose(1, 2).contiguous()
    # flatten
    x = x.view(B, -1, H, W)
    return x


'''
Adapted from:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
'''
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks """

    def __init__(self, in_features, mlp_ratio=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = round(in_features * mlp_ratio) if mlp_ratio else in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvActNorm(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1,
                    groups=1, bias=True, act=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation,
                        groups, bias)
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = act() if act is not None else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.norm(x)
        return x


class MultiSepConv(nn.Module):
    """ Multi-scale Separable Convolution Block """    

    def __init__(self, in_features, out_features=None, paths=4, dw_size=3, stride=1, act=nn.Hardswish):
        super().__init__()
        assert paths > 0 and paths <= 4, "Paths should be in the interval of [1,4]"
        out_features = out_features if out_features else in_features
        self.res = nn.Conv2d(in_features, out_features, 1, 1, 0)
        self.conv = nn.ModuleList(
            nn.Sequential(*[
                ConvActNorm(in_features, in_features, dw_size, 1, dw_size//2,
                        groups=in_features, bias=False, act=None)
                for _ in range(i+1)
            ])
            for i in range(paths)
        )
        self.fc = nn.Conv2d(int(in_features*paths), out_features, 1, 1, 0)
        self.pool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.act = act()

    def forward(self, x):
        res = self.pool(self.res(x))
        x = self.act(torch.cat([conv(x) for conv in self.conv], dim=1))
        x = self.fc(self.pool(x))
        return x + res


class MultiGhostConv(nn.Module):
    """ Multi-scale Ghost Convolution
    GhostNet: More Features from Cheap Operations, Han et al., 2020
    https://arxiv.org/abs/1911.11907
    """

    def __init__(self, in_features, out_features=None, paths=2, dw_size=3, stride=1, act=nn.Hardswish):
        super().__init__()
        assert paths > 0 and paths <= 4, "Paths should be in the interval of [1,4]"
        hidden_features = out_features//paths if out_features else in_features
        self.pw_conv = ConvActNorm(in_features, hidden_features, 1, 1, 0, bias=False, act=None)
        self.dw_conv = nn.ModuleList(
            nn.Sequential(*[
                ConvActNorm(hidden_features, hidden_features, dw_size, stride if i==0 else 1, dw_size//2,
                            groups=hidden_features, bias=False, act=None)
                for i in range(p+1)
            ])
            for p in range(paths)
        )
        self.act = act()

    def forward(self, x):
        x = self.pw_conv(x)
        x = torch.cat([conv(x) for conv in self.dw_conv], dim=1)
        x = self.act(x)
        return x


class GhostFFN(nn.Module):
    """FFN module using Multi-scale Ghost Convolutions.
        Replaces MLP module.
    """

    def __init__(self, in_features, mlp_ratio=None, act_layer=nn.Hardswish, drop=0.):
        super().__init__()
        self.conv = MultiGhostConv(in_features, paths=mlp_ratio, act=act_layer)
        self.fc = nn.Conv2d(int(in_features*mlp_ratio), in_features, 1, 1, 0)
        self.drop = nn.Dropout2d(drop)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(sqrt(L))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        x = self.drop(x)
        x = self.fc(x)
        x = x.reshape(B, C, L).transpose(-2, -1).contiguous()
        return x


class PatchMerge(nn.Module):
    """ Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        dim_out (int): Number of output channels.
        use_multi_merge (bool, optional): Use multiscale convolutions for patch merging. Default: False
    """

    def __init__(self, dim, dim_out, use_multi_merge=False, act_layer=nn.Hardswish):
        super().__init__()
        self.merge = nn.Conv2d(dim, dim_out, 3, 2, 1) if not use_multi_merge else \
                        MultiSepConv(dim, dim_out, stride=2, paths=4, act=act_layer)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(sqrt(L))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.merge(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        return x


class PatchExpand(nn.Module):
    """ Patch Expansion Layer.
    Args:
        dim (int): Number of input channels.
        dim_out (int): Number of output channels.
        scale (int): Expansion scale.
    """

    def __init__(self, dim, dim_out, scale):
        super().__init__()
        assert dim//dim_out==scale or dim==dim_out, "dimension mismatch"
        dim_interm = dim*scale**2 if dim==dim_out else dim*scale
        self.expand = nn.Linear(dim, dim_interm, bias=False)
        self.shuffle = nn.PixelShuffle(scale)
    
    def forward(self, x):
        x = self.expand(x)
        B, L, C = x.shape
        H = W = int(sqrt(L))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.shuffle(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class LePEAttention(nn.Module):
    """ Cross-shaped window based multi-head self attention (W-MSA) module with
    locally-enhanced positional encoding (LePE).
    Args:
        dim (int): Number of input channels.
        res (int): Resolution of input feature.
        idx (-1 | 0 | 1): If 0, horizontal striped windows,; if 1, vertical striped windows; else global attention.
        split_size (int): The size of the window strip.
        dim_out (int): Number of output channels.
        num_heads (int): Number of attention heads.
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, res, idx, split_size=8, dim_out=None, num_heads=8, attn_drop=0., qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = res
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            H_sp, W_sp = self.split_size,  self.resolution
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, L, C = x.shape
        H = W = int(sqrt(L))

        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C //
                      self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, L, C = x.shape
        H = W = int(sqrt(L))
        
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = x.view(B, C, H // self.H_sp, self.H_sp, W // self.W_sp, self.W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, self.H_sp, self.W_sp)

        lepe = func(x)  # B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads,
                            self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads,
                        self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Img2Window
        B, L, C = q.shape
        H = W = self.resolution
        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        
        attn_out = attn
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe  # B head N N @ B head N C
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)

        # Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x, attn_out


class CSWinBlock(nn.Module):
    """ CSwin Transformer Block.
    Args:
        dim (int): Number of input channels.
        res (int): Input resolution.
        num_heads (int): Number of attention heads.
        split_size (int): Split size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        use_ghost_ffn (bool, optional): If True, use the modified MLP block. Default: False
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, res, num_heads, split_size=8, mlp_ratio=4., use_ghost_ffn=False,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = res
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        last_stage = self.patches_resolution == split_size
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, res=self.patches_resolution, idx=-1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop)
                for _ in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim // 2, res=self.patches_resolution, idx=i,
                    split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                    qk_scale=qk_scale, attn_drop=attn_drop)
                for i in range(self.branch_num)])

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        mlp_layer = GhostFFN if use_ghost_ffn else Mlp
        self.mlp = mlp_layer(in_features=dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x):
        """
        x: B, H*W, C
        """

        B, L, C = x.shape
        H = W = self.patches_resolution
        assert L == H * W, "flatten img_tokens has wrong size"

        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1, attn1 = self.attns[0](qkv[:, :, :, :C // 2])
            x2, attn2 = self.attns[1](qkv[:, :, :, C // 2:])
            attened_x = torch.cat([x1, x2], dim=2)
            attn = torch.cat([attn1, attn2], dim=1)
        else:
            attened_x, attn = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)

        shortcut = x
        x = self.mlp(self.norm2(x))
        x = shortcut + self.drop_path(x)

        return x, attn


class BasicEncoderLayer(nn.Module):
    """ A basic CSwin Transformer layer for one encoder stage.
    Args:
        dim (int): Number of input channels.
        patches_resolution (int): Input patch resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        split_size (int): Split size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        use_ghost_ffn (bool, optional): If True, use the modified MLP block. Default: False
        use_multi_merge (bool, optional): Use multiscale depth separable convolutions while patch merging. Default: False
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the beginning of the layer. Default: None
    """

    def __init__(self, dim, patches_resolution, depth, num_heads, split_size, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 use_ghost_ffn=False, use_multi_merge=False, norm_layer=nn.LayerNorm,
                 downsample=PatchMerge):

        super().__init__()
        self.blocks = nn.ModuleList([
            CSWinBlock(dim=dim, res=patches_resolution,
                       num_heads=num_heads, split_size=split_size,
                       mlp_ratio=mlp_ratio,
                       qkv_bias=qkv_bias, qk_scale=qk_scale,
                       drop=drop, attn_drop=attn_drop,
                       drop_path=drop_path[i] if isinstance(
                           drop_path, list) else drop_path,
                       use_ghost_ffn=use_ghost_ffn,
                       norm_layer=norm_layer)
            for i in range(depth)])
        self.downsample = downsample(dim=dim//2, dim_out=dim, use_multi_merge=use_multi_merge)
        self.norm = norm_layer(dim)

    def forward(self, x):
        x = self.downsample(x)
        for blk in self.blocks:
            x, _ = blk(x)
        x = self.norm(x)
        x_trace = x.detach().clone()
        return x, x_trace

    def forward_with_attn(self, x):
        attns = []
        x = self.downsample(x)
        for blk in self.blocks:
            x, attn = blk(x)
            attns.append(attn)
        x = self.norm(x)
        return x, attns


class BasicDecoderLayer(nn.Module):
    """ A basic CSwin Transformer layer for one decoder stage.
    Args:
        dim (int): Number of input channels.
        patches_resolution (int): Input patch resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        split_size (int): Split size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        use_ghost_ffn (bool, optional): If True, use the modified MLP block. Default: False
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): Upsample layer at the end of the layer. Default: None
        skip_type ('add' | 'concat', optional): Type of operation for skip connections. Default: 'concat'
    """

    def __init__(self, dim, patches_resolution, depth, num_heads, split_size, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 use_ghost_ffn=False, skip_type='concat', norm_layer=nn.LayerNorm,
                 upsample=PatchExpand):

        super().__init__()

        assert skip_type in ('add','concat'), "Invalid skip type!"
        self.skip_type = skip_type
        
        if skip_type == 'concat':
            # dim reduction layer
            # reduces channel dimension after skip connections
            self.reduce = nn.Linear(dim*2, dim)

        # patch expanding layer
        self.upsample = upsample(dim=dim*2, dim_out=dim, scale=2)

        # build blocks
        self.blocks = nn.ModuleList([
            CSWinBlock(dim=dim, res=patches_resolution,
                       num_heads=num_heads, split_size=split_size,
                       mlp_ratio=mlp_ratio,
                       qkv_bias=qkv_bias, qk_scale=qk_scale,
                       drop=drop, attn_drop=attn_drop,
                       drop_path=drop_path[i] if isinstance(
                           drop_path, list) else drop_path,
                       use_ghost_ffn=use_ghost_ffn,
                       norm_layer=norm_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(dim)

    def forward(self, x, x_trace):
        x = self.upsample(x)
        if self.skip_type == 'concat':
            x = torch.cat([x,x_trace],-1)
            x = self.reduce(x)
        else:
            x += x_trace
        for blk in self.blocks:
            x, _ = blk(x)
        x = self.norm(x)
        return x

    def forward_with_attn(self, x, x_trace):
        attns = []
        x = self.upsample(x)
        if self.skip_type == 'concat':
            x = torch.cat([x,x_trace],-1)
            x = self.reduce(x)
        else:
            x += x_trace
        for blk in self.blocks:
            x, attn = blk(x)
            attns.append(attn)
        x = self.norm(x)
        return x, attns


class CSWinEncoder(nn.Module):
    """ CSwin Transformer Encoder
        A PyTorch impl of : `CSwin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows`
          https://arxiv.org/pdf/2107.00652
    Args:
        img_size (int | tuple(int)): Input image size. Default 256
        in_chans (int): Number of input image channels. Default: 1
        num_classes (int): Number of classes for classification head. Default: 0
        patch_size (int): Patch size. Default: 4
        embed_dim (int): Patch embedding dimension. Default: 64
        depths (tuple(int)): Depth of each CSwin Transformer layer.
        split_size (tuple(int)): Split size for different layers.
        num_heads (tuple(int)): Number of attention heads in different layers.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        attn_drop_rate (float): Attention dropout rate. Default: 0
        use_multi_merge (bool, optional): Use multiscale depth separable convolutions while patch merging. Default: False
        use_ghost_ffn (bool, optional): If True, use the modified MLP block. Default: False
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, img_size=256, in_chans=1, num_classes=0, patch_size=4, embed_dim=64,
                 depths=[1, 2, 21, 1], split_size=[1, 2, 8, 8], num_heads=[2, 4, 8, 16], mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 use_ghost_ffn=False, use_multi_merge=False, norm_layer=nn.LayerNorm):

        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        patches_resolution = img_size // patch_size

        self.stem = ConvActNorm(in_chans, embed_dim//2, 7, 2, 3, act=nn.Hardswish)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build encoder layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicEncoderLayer(dim=int(embed_dim * 2 ** i_layer),
                               patches_resolution=patches_resolution // (2 ** i_layer),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               split_size=split_size[i_layer],
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               use_ghost_ffn=use_ghost_ffn,
                               use_multi_merge=use_multi_merge,
                            )
            self.layers.append(layer)

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

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        x = self.stem(x).flatten(2).transpose(1, 2)  # B ph*pw C
        x_trace = []
        for i, layer in enumerate(self.layers):
            x, x_ = layer(x)
            if i != self.num_layers-1:
                x_trace.append(x_)
        return x, x_trace
    
    def forward_attention(self, x):
        x = self.stem(x).flatten(2).transpose(1, 2)  # B ph*pw C
        attention = []
        for layer in self.layers:
            x, attns = layer.forward_with_attn(x)
            attention.append(attns)
        return attention


class CSWinDecoder(nn.Module):
    """ CSwin Transformer Decoder
        A PyTorch impl of : `CSwin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows`
          https://arxiv.org/pdf/2107.00652
    Args:
        img_size (int | tuple(int)): Input image size. Default 256
        in_chans (int): Number of input image channels. Default: 1
        num_classes (int): Number of classes for classification head. Default: 0
        patch_size (int): Patch size. Default: 4
        embed_dim (int): Patch embedding dimension. Default: 64
        depths (tuple(int)): Depth of each CSwin Transformer layer.
        split_size (tuple(int)): Split size for different layers.
        num_heads (tuple(int)): Number of attention heads in different layers.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        attn_drop_rate (float): Attention dropout rate. Default: 0
        use_ghost_ffn (bool, optional): If True, use the modified MLP block. Default: False
        skip_type ('add' | 'concat'): Type of operation for skip connections. Default: 'concat'
        expand_first (bool): If True, upsample before prediction head. Default: False
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, img_size=256, in_chans=1, num_classes=0, patch_size=4, embed_dim=64,
                 depths=[21, 2, 1], split_size=[8, 2, 1], num_heads=[8, 4, 2], mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 use_ghost_ffn=False, skip_type='concat', expand_first=False,
                 norm_layer=nn.LayerNorm):

        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        
        self.expand_first = expand_first
        self.img_size = img_size
        
        patches_resolution = img_size // patch_size

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # build decoder layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            idx = self.num_layers - i_layer - 1
            layer = BasicDecoderLayer(dim=int(embed_dim * 2 ** idx),
                               patches_resolution=patches_resolution // (2 ** idx),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               split_size=split_size[i_layer],
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               skip_type=skip_type,
                               use_ghost_ffn=use_ghost_ffn
                            )
            self.layers.append(layer)
        
        self.patch_expand = PatchExpand(self.embed_dim, self.embed_dim, patch_size) \
                                if expand_first else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes)

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

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, x_trace):
        for i, layer in enumerate(self.layers):
            x = layer(x, x_trace[i])
        x = self.patch_expand(x)
        x = self.head(x)

        B, L, C = x.shape
        H = W = int(sqrt(L))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        if not self.expand_first:
            x = nn.functional.interpolate(x, size=self.img_size, mode='bilinear')

        return x

    def forward_attention(self, x, x_trace):
        attention = []
        for i, layer in enumerate(self.layers):
            x, attns = layer.forward_with_attn(x, x_trace[i])
            attention.append(attns)
        return attention


@encoder_entrypoints.register('cswin')
def build_encoder(config):
    encoder_config = config.MODEL.ENCODER
    return CSWinEncoder(
            config.DATA.IMAGE_SIZE, config.DATA.IN_CHANS, 0,
            patch_size=encoder_config['PATCH_SIZE'],
            split_size=encoder_config['SPLIT_SIZE'],
            embed_dim=encoder_config['EMBED_DIM'],
            num_heads=encoder_config['NUM_HEADS'],
            depths=encoder_config['DEPTHS'],
            mlp_ratio=encoder_config['MLP_RATIO'],
            qkv_bias=encoder_config['QKV_BIAS'],
            qk_scale=encoder_config['QK_SCALE'],
            drop_rate=encoder_config['DROP_RATE'],
            drop_path_rate=encoder_config['DROP_PATH_RATE'],
            attn_drop_rate=encoder_config['ATTN_DROP_RATE'],
            use_ghost_ffn=encoder_config['USE_GHOST_FFN'],
            use_multi_merge=encoder_config['USE_MULTI_MERGE']
        )


@decoder_entrypoints.register('cswin')
def build_decoder(config):
    encoder_config = config.MODEL.ENCODER
    return CSWinDecoder(
            config.DATA.IMAGE_SIZE, config.DATA.IN_CHANS, config.DATA.NUM_CLASSES,
            patch_size=encoder_config['PATCH_SIZE'],
            split_size=encoder_config['SPLIT_SIZE'][-2::-1],
            embed_dim=encoder_config['EMBED_DIM'],
            num_heads=encoder_config['NUM_HEADS'][-2::-1],
            depths=encoder_config['DEPTHS'][-2::-1],
            mlp_ratio=encoder_config['MLP_RATIO'],
            qkv_bias=encoder_config['QKV_BIAS'],
            qk_scale=encoder_config['QK_SCALE'],
            drop_rate=encoder_config['DROP_RATE'],
            drop_path_rate=encoder_config['DROP_PATH_RATE'],
            attn_drop_rate=encoder_config['ATTN_DROP_RATE'],
            use_ghost_ffn=encoder_config['USE_GHOST_FFN'],
            skip_type=config.MODEL.DECODER.SKIP_TYPE,
            expand_first=config.MODEL.DECODER.EXPAND_FIRST
        )
