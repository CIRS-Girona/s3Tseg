import torch.nn as nn
from math import sqrt
from utils import utils

from models.registry import decoder_entrypoints


class LinearDecoder(nn.Module):
    """ Linear Decoder: A single fully connected linear layer
    Args:
        img_size (int): Input resolution.
        num_classes (int): Number of classes for classification head.
        in_features (int): Embedding dimension of the final encoder layer.
    """

    def __init__(self, img_size, num_classes, in_features):
        super().__init__()
        self.img_size = utils.to_2tuple(img_size)
        self.head = nn.Conv2d(in_features, num_classes, 1, 1, 0)
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

    def forward(self, x, _=None):
        B, L, C = x.shape
        H = W = int(sqrt(L))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.head(x)
        x = nn.functional.interpolate(x, size=self.img_size, mode='bilinear')
        return x


@decoder_entrypoints.register('linear')
def build_decoder(config):
    return LinearDecoder(
            img_size=config.DATA.IMAGE_SIZE,
            num_classes=config.DATA.NUM_CLASSES,
            in_features=config.MODEL.DECODER.IN_FEATURES
        )
