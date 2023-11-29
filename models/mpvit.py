# -----------------------------------------------------------
# Implementation of the architecture proposed in
# MPViT: Multi-Path Vision Transformer for Dense Prediction
# https://arxiv.org/pdf/2112.11010.pdf
# -----------------------------------------------------------


import torch
import torch.nn as nn
from math import sqrt

from models.registry import encoder_entrypoints
from utils import utils


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
        x = self.norm(x)
        x = self.act(x)
        return x


class DepthSepConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                            groups=in_channels)
        self.dw_norm = nn.GroupNorm(1, in_channels)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.pw_norm = nn.GroupNorm(1, out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.dw_norm(self.dw_conv(x))
        x = self.relu(x)
        x = self.pw_norm(self.pw_conv(x))
        return x


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
        x = self.ffn(x)
        x = self.drop(x)
        x = self.fc(x)
        x = x.reshape(B, C, L).transpose(-2, -1).contiguous()
        return x


class Attention(nn.Module):
    """ SimA module, adapted from:
    SimA: Simple Softmax-free Attention for Vision Transformers
    https://arxiv.org/pdf/2206.08898.pdf

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, proj_drop=0.):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}
    
    def get_lepe(self, x, func):
        B, N, L, C_ = x.shape
        H = W = int(sqrt(L))

        x = x.transpose(-2, -1).contiguous().view(B, -1, H, W)
        lepe = func(x)  # B_, C, H, W
        
        lepe = lepe.reshape(B, N, C_, L).permute(0, 1, 3, 2).contiguous()
        x = x.reshape(B, N, C_, L).permute(0, 1, 3, 2).contiguous()
        
        return x, lepe
    
    def forward(self, x):
        """
        Args:
            x: input features with shape (B, H*W, C)
        """
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                        # B, N, L, C_

        q = nn.functional.normalize(q, dim=-2)
        k = nn.functional.normalize(k, dim=-2)
        v, lepe = self.get_lepe(v, self.get_v)                  # B, N, L, C_

        attn = (k.transpose(-2, -1) @ v) * self.temperature     # B, N, C_, C_

        x = (q @ attn) + lepe                                   # B, N, L, C_
        x = x.transpose(1, 2).reshape(B, L, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn


class SimABlock(nn.Module):
    """ SimA Block.
    Args:
        dim (int): Number of input channels.
        patches_resolution (int): Input resolution.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        use_ghost_ffn (bool, optional): If True, use the modified MLP block. Default: False
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, patches_resolution, num_heads, mlp_ratio=4., qkv_bias=True,
                    drop=0., drop_path=0., use_ghost_ffn=False, norm_layer=nn.LayerNorm):
        super().__init__()

        self.dim = dim
        self.patches_resolution = patches_resolution

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        mlp_layer = GhostFFN if use_ghost_ffn else Mlp
        self.mlp = mlp_layer(in_features=dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x):
        H = W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size %d, %d, %d" % (L, H, W)
        
        # multi-head self-attention
        shortcut = x
        x, attn = self.attn(self.norm1(x))
        x = shortcut + self.drop_path(x)
        
        # FFN
        shortcut = x
        x = self.mlp(self.norm2(x))
        x = shortcut + self.drop_path(x)

        return x, attn


class EncoderBlock(nn.Module):
    """ A basic MPViT layer for one encoder stage.
    Args:
        dim (int): Number of input channels.
        patches_resolution (int): Input patch resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        use_ghost_ffn (bool, optional): If True, use the modified MLP block. Default: False
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, patches_resolution, depth, num_heads, mlp_ratio=4., qkv_bias=True,
                    drop=0., drop_path=0., use_ghost_ffn=False, norm_layer=nn.LayerNorm):

        super().__init__()
        self.blocks = nn.ModuleList([
            SimABlock(dim=dim, patches_resolution=patches_resolution,
                        num_heads=num_heads, mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, drop=drop,
                        drop_path=drop_path[i] if isinstance(
                                    drop_path, list) else drop_path,
                        use_ghost_ffn=use_ghost_ffn,
                        norm_layer=norm_layer)
                    for i in range(depth)])
        self.norm = norm_layer(dim)

    def forward(self, x):
        for blk in self.blocks:
            x, _ = blk(x)
        x = self.norm(x)
        return x

    def forward_with_attn(self, x):
        attns = []
        for blk in self.blocks:
            x, attn = blk(x)
            attns.append(attn)
        x = self.norm(x)
        return x, attns


class EncoderStage(nn.Module):
    """ A basic MPViT layer for one encoder stage.
    Args:
        dim (int): Number of input channels.
        patches_resolution (int): Input patch resolution.
        depth (int): Number of encoder blocks.
        num_heads (int): Number of multi-resolution paths.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        use_ghost_ffn (bool, optional): If True, use the modified MLP block. Default: False
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, patches_resolution, depth, num_paths, num_heads,
                mlp_ratio=4., qkv_bias=True, drop=0., drop_path=None,
                use_ghost_ffn=False, norm_layer=nn.LayerNorm):

        super().__init__()
        self.encoder_paths = nn.ModuleList([
            EncoderBlock(dim=dim//num_paths, 
                    patches_resolution=patches_resolution,
                    num_heads=num_heads, mlp_ratio=mlp_ratio,
                    depth=depth, qkv_bias=qkv_bias, drop=drop,
                    drop_path=drop_path if drop_path is not None else 0.,
                    use_ghost_ffn=use_ghost_ffn, norm_layer=norm_layer,
                )
            for i in range(num_paths)
        ])
        self.aggregation = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Hardswish()
        )

    def forward(self, inputs):
        out = []
        for x, path in zip(inputs,self.encoder_paths):
            out.append(path(x))
        out = torch.cat(out, dim=-1)
        out = self.aggregation(out)
        trace = out.detach().clone()
        return out, trace

    def forward_with_attn(self, inputs):
        out = []
        attn = []
        for x, path in zip(inputs,self.encoder_paths):
            o, a = path(x)
            out.append(o)
            attn.append(a)
        out = torch.cat(out, dim=1)
        out = self.aggregation(out)
        return out, attn


class PatchEmbedPath(nn.Module):
    """ Patch Embedding Path.
    Args:
        dim (int): Number of input channels.
        dim_out (int): Number of output channels.
    """

    def __init__(self, dim, dim_out, path_idx):
        super().__init__()
        self.embed = nn.ModuleList([
            DepthSepConv(dim, dim_out if i==path_idx-1 else dim, 
                    3, 2 if i==path_idx-1 else 1, 1)
            for i in range(path_idx)
        ])

    def forward(self, x):
        for module in self.embed:
            x = module(x)
        return x


class PatchEmbedStage(nn.Module):
    """ Multipath Patch Embedding.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of multi-resolution paths.
    """

    def __init__(self, dim, num_paths):
        super().__init__()
        self.embed_paths = nn.ModuleList([
            PatchEmbedPath(dim//2, dim//num_paths, i+1)
                for i in range(num_paths)
        ])

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(sqrt(L))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        out = []
        for path in self.embed_paths:
            x_ = path(x)
            B, C = x_.shape[:2]
            x_ = x_.view(B, C, -1).transpose(-2, -1).contiguous()
            out.append(x_)
        return out


class MPViTEncoder(nn.Module):
    """ MPViT Encoder
    Args:
        img_size (int | tuple(int)): Input image size. Default 256
        in_chans (int): Number of input image channels. Default: 1
        num_classes (int): Number of classes for classification head. Default: 0
        patch_size (int): Patch size. Default: 4
        embed_dim (int): Patch embedding dimension. Default: 64
        depths (tuple(int)): Depth of each CSwin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        use_ghost_ffn (bool, optional): If True, use the modified MLP block. Default: False
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, img_size=256, in_chans=1, num_classes=0, patch_size=4, embed_dim=64,
                 depths=[2,2,6,2], num_paths=[3,3,3,3], num_heads=[2,4,8,16], mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., drop_path_rate=0.1, use_ghost_ffn=False,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))
        
        patches_resolution = img_size // patch_size
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim//2, 7, 2, 3),
            nn.BatchNorm2d(embed_dim//2), nn.Hardswish()
        )

        # stochastic depth decay rule
        cur = 0
        dpr_list = []
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for i in range(self.num_stages):
            dpr_per_stage = dpr[cur:cur + depths[i]]
            dpr_list.append(dpr_per_stage)
            cur += depths[i]

        # build encoder stages
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            dim = int(embed_dim * 2 ** i)
            stage = nn.Sequential(
                PatchEmbedStage(dim=dim, num_paths=num_paths[i]),
                EncoderStage(dim=dim, depth=depths[i],
                    patches_resolution=patches_resolution // (2 ** i),
                    num_paths=num_paths[i], num_heads=num_heads[i],
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    drop=drop_rate, drop_path=dpr[i],
                    norm_layer=norm_layer,
                    use_ghost_ffn=use_ghost_ffn,
                )
            )
            self.stages.append(stage)

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
        for i, stage in enumerate(self.stages):
            x, x_ = stage(x)
            if i != self.num_stages-1:
                x_trace.append(x_)
        return x, x_trace
    
    def forward_attention(self, x):
        x = self.stem(x).flatten(2).transpose(1, 2)  # B ph*pw C
        attention = []  # list of lists (attn per path per stage)
        for stage in self.stages:
            x, attns = stage.forward_with_attn(x)
            attention.append(attns)
        return attention


@encoder_entrypoints.register('mpvit')
def build_encoder(config):
    encoder_config = config.MODEL.ENCODER
    return MPViTEncoder(
            config.DATA.IMAGE_SIZE, config.DATA.IN_CHANS, 0,
            patch_size=encoder_config['PATCH_SIZE'],
            embed_dim=encoder_config['EMBED_DIM'],
            depths=encoder_config['DEPTHS'],
            num_paths=encoder_config['NUM_PATHS'],
            num_heads=encoder_config['NUM_HEADS'],
            mlp_ratio=encoder_config['MLP_RATIO'],
            qkv_bias=encoder_config['QKV_BIAS'],
            drop_rate=encoder_config['DROP_RATE'],
            drop_path_rate=encoder_config['DROP_PATH_RATE'],
            use_ghost_ffn=encoder_config['USE_GHOST_FFN'],
        )
