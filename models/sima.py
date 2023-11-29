# ------------------------------------------------------------
# Implementation of attention mechanism proposed in
# SimA: Simple Softmax-free Attention for Vision Transformers
# https://arxiv.org/pdf/2206.08898.pdf
# ------------------------------------------------------------


import torch
import torch.nn as nn

from math import sqrt
from utils import utils

from models.registry import encoder_entrypoints
from models.registry import decoder_entrypoints


class DropPath(nn.Module):
    """ Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Adapted from:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
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


class Attention(nn.Module):
    """ SimA module
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
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}
    
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

        attn = (k.transpose(-2, -1) @ v) * self.temperature     # B, N, C_, C_

        x = (q @ attn)                                          # B, N, L, C_
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


class BasicEncoderLayer(nn.Module):
    """ A basic SimA layer for one encoder stage.
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
        use_multi_merge (bool, optional): Use multiscale depth separable convolutions while patch merging. Default: False
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the beginning of the layer. Default: None
    """

    def __init__(self, dim, patches_resolution, depth, num_heads, mlp_ratio=4., qkv_bias=True,
                    drop=0., drop_path=0., use_ghost_ffn=False, use_multi_merge=False,
                    norm_layer=nn.LayerNorm, downsample=PatchMerge):

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
    """ A basic SimA layer for one decoder stage.
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
        skip_type ('add' | 'concat', optional): Type of operation for skip connections. Default: 'concat'
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): Upsample layer at the end of the layer. Default: None
    """

    def __init__(self, dim, patches_resolution, depth, num_heads, mlp_ratio=4., qkv_bias=True,
                    drop=0., drop_path=0., use_ghost_ffn=False, skip_type='concat',
                    norm_layer=nn.LayerNorm, upsample=PatchExpand):

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
            SimABlock(dim=dim, patches_resolution=patches_resolution,
                        num_heads=num_heads, mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, drop=drop,
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


class SimAEncoder(nn.Module):
    """ SimA Encoder
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
        use_multi_merge (bool, optional): Use multiscale depth separable convolutions while patch merging. Default: False
        use_ghost_ffn (bool, optional): If True, use the modified MLP block. Default: False
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, img_size=256, in_chans=1, num_classes=0, patch_size=4, embed_dim=64,
                 depths=[2,2,6,2], num_heads=[2,4,8,16], mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., drop_path_rate=0.1, use_ghost_ffn=False,
                 use_multi_merge=False, norm_layer=nn.LayerNorm):

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
                               mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                               drop=drop_rate,
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


class SimADecoder(nn.Module):
    """ SimA Decoder
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
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        skip_type ('add' | 'concat'): Type of operation for skip connections. Default: 'concat'
        expand_first (bool): If True, upsample before prediction head. Default: False
        use_ghost_ffn (bool, optional): If True, use the modified MLP block. Default: False
    """

    def __init__(self, img_size=256, in_chans=1, num_classes=0, patch_size=4, embed_dim=64,
                 depths=[6, 2, 2], num_heads=[8, 4, 2], mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., drop_path_rate=0.1, use_ghost_ffn=False,
                 skip_type='concat', expand_first=False, norm_layer=nn.LayerNorm):
        
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
                               mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                               drop=drop_rate,
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


@encoder_entrypoints.register('sima')
def build_encoder(config):
    encoder_config = config.MODEL.ENCODER
    return SimAEncoder(
            config.DATA.IMAGE_SIZE, config.DATA.IN_CHANS, 0,
            patch_size=encoder_config['PATCH_SIZE'],
            embed_dim=encoder_config['EMBED_DIM'],
            num_heads=encoder_config['NUM_HEADS'],
            depths=encoder_config['DEPTHS'],
            mlp_ratio=encoder_config['MLP_RATIO'],
            qkv_bias=encoder_config['QKV_BIAS'],
            drop_rate=encoder_config['DROP_RATE'],
            drop_path_rate=encoder_config['DROP_PATH_RATE'],
            use_ghost_ffn=encoder_config['USE_GHOST_FFN'],
            use_multi_merge=encoder_config['USE_MULTI_MERGE']
        )


@decoder_entrypoints.register('sima')
def build_decoder(config):
    encoder_config = config.MODEL.ENCODER
    return SimADecoder(
            config.DATA.IMAGE_SIZE, config.DATA.IN_CHANS, config.DATA.NUM_CLASSES,
            patch_size=encoder_config['PATCH_SIZE'],
            embed_dim=encoder_config['EMBED_DIM'],
            num_heads=encoder_config['NUM_HEADS'][-2::-1],
            depths=encoder_config['DEPTHS'][-2::-1],
            mlp_ratio=encoder_config['MLP_RATIO'],
            qkv_bias=encoder_config['QKV_BIAS'],
            drop_rate=encoder_config['DROP_RATE'],
            drop_path_rate=encoder_config['DROP_PATH_RATE'],
            use_ghost_ffn=encoder_config['USE_GHOST_FFN'],
            skip_type=config.MODEL.DECODER.SKIP_TYPE,
            expand_first=config.MODEL.DECODER.EXPAND_FIRST
        )
