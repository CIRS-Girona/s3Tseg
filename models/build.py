import torch.nn as nn

from models.registry import encoder_entrypoints
from models.registry import decoder_entrypoints


class Model(nn.Module):
    """Represents an encoder-decoder model for semantic segmentation of
    side-scan sonar waterfalls.
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        x, x_trace = self.encoder(x)
        x = self.decoder(x, x_trace[::-1])
        return x


def build_model(config):
    """Helper function to build the appropriate encoder/decoder architecture
    as per user specifications.
    """

    if config.MODEL.ENCODER.TYPE not in encoder_entrypoints:
        raise ValueError(f'Unknown Encoder: {config.MODEL.ENCODER.TYPE}')
    encoder = encoder_entrypoints.get(config.MODEL.ENCODER.TYPE)(config)
    
    config.defrost()
    if config.MODEL.DECODER.TYPE == 'symmetric':
        if config.MODEL.ENCODER.TYPE == 'mpvit':
            raise TypeError('Symmetric Decoder incompatible with MPViT Encoder')
        config.MODEL.DECODER.TYPE = config.MODEL.ENCODER.TYPE
        config.MODEL.DECODER.NAME = config.MODEL.ENCODER.NAME
    config.MODEL.DECODER.IN_FEATURES = encoder.num_features
    config.freeze()

    if config.MODEL.DECODER.TYPE not in decoder_entrypoints:
        raise ValueError(f'Unknown Decoder: {config.MODEL.DECODER.TYPE}')
    decoder = decoder_entrypoints.get(config.MODEL.DECODER.TYPE)(config)

    return Model(encoder, decoder)
