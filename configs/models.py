#===================== Encoders =====================

def mpvit():
    opts = [
        'MODEL.ENCODER.TYPE', 'mpvit',
        'MODEL.ENCODER.NAME', 'mpvit_sima',
        'MODEL.ENCODER.MLP_RATIO', '2',
        'MODEL.ENCODER.EMBED_DIM', '24',
        'MODEL.ENCODER.NUM_HEADS', '(2, 4, 8, 16)',
        'MODEL.ENCODER.NUM_PATHS', '(3, 3, 3, 3)',
        'MODEL.ENCODER.DEPTHS', '(3, 6, 12, 3)',
        'MODEL.ENCODER.QKV_BIAS', 'False',
    ]
    return opts

#----------------------------------------------------

def swin_commons():
    opts = [
        'MODEL.ENCODER.TYPE', 'swin',
        'MODEL.ENCODER.WINDOW_SIZE', '8',
        'MODEL.ENCODER.MLP_RATIO', '4',
        'MODEL.ENCODER.QKV_BIAS', 'True',
        'MODEL.ENCODER.USE_ABS_POS_EMBED', 'False'
    ]
    return opts

def swin_mini():
    opts = swin_commons()
    opts.extend([
        'MODEL.ENCODER.NAME', 'swin_mini',
        'MODEL.ENCODER.EMBED_DIM', '64',
        'MODEL.ENCODER.NUM_HEADS', '(2, 4, 8, 16)',
        'MODEL.ENCODER.DEPTHS', '(1, 2, 7, 2)'
    ])
    return opts

def swin_tiny():
    opts = swin_commons()
    opts.extend([
        'MODEL.ENCODER.NAME', 'swin_tiny',
        'MODEL.ENCODER.EMBED_DIM', '96',
        'MODEL.ENCODER.NUM_HEADS', '(3, 6, 12, 24)',
        'MODEL.ENCODER.DEPTHS', '(2, 2, 6, 2)'
    ])
    return opts

def swin_small():
    opts = swin_commons()
    opts.extend([
        'MODEL.ENCODER.NAME', 'swin_small',
        'MODEL.ENCODER.EMBED_DIM', '96',
        'MODEL.ENCODER.NUM_HEADS', '(3, 6, 12, 24)',
        'MODEL.ENCODER.DEPTHS', '(2, 2, 18, 2)'
    ])
    return opts

def swin_base():
    opts = swin_commons()
    opts.extend([
        'MODEL.ENCODER.NAME', 'swin_base',
        'MODEL.ENCODER.EMBED_DIM', '128',
        'MODEL.ENCODER.NUM_HEADS', '(4, 8, 16, 32)',
        'MODEL.ENCODER.DEPTHS', '(2, 2, 18, 2)'
    ])
    return opts

#----------------------------------------------------

def cswin_commons():
    opts = [
        'MODEL.ENCODER.TYPE', 'cswin',
        'MODEL.ENCODER.SPLIT_SIZE', '(1, 2, 8, 8)',
        'MODEL.ENCODER.MLP_RATIO', '4',
        'MODEL.ENCODER.QKV_BIAS', 'False',
    ]
    return opts

def cswin_small():
    opts = cswin_commons()
    opts.extend([
        'MODEL.ENCODER.NAME', 'cswin_small',
        'MODEL.ENCODER.EMBED_DIM', '64',
        'MODEL.ENCODER.NUM_HEADS', '(2, 4, 8, 16)',
        'MODEL.ENCODER.DEPTHS', '(2, 4, 32, 2)'
    ])
    return opts

def cswin_mini():
    opts = cswin_commons()
    opts.extend([
        'MODEL.ENCODER.NAME', 'cswin_mini',
        'MODEL.ENCODER.EMBED_DIM', '24',
        'MODEL.ENCODER.NUM_HEADS', '(2, 4, 8, 16)',
        'MODEL.ENCODER.DEPTHS', '(1, 2, 21, 1)'
    ])
    return opts

#----------------------------------------------------

def pale_commons():
    opts = [
        'MODEL.ENCODER.TYPE', 'pale',
        'MODEL.ENCODER.GROUP_SIZE', '8',
        'MODEL.ENCODER.MLP_RATIO', '2',
        'MODEL.ENCODER.QKV_BIAS', 'False',
    ]
    return opts

def pale_mini():
    opts = pale_commons()
    opts.extend([
        'MODEL.ENCODER.NAME', 'pale_mini',
        'MODEL.ENCODER.EMBED_DIM', '24',
        'MODEL.ENCODER.NUM_HEADS', '(2, 4, 8, 16)',
        'MODEL.ENCODER.DEPTHS', '(1, 3, 7, 1)'
    ])
    return opts

#----------------------------------------------------

def lsda_commons():
    opts = [
        'MODEL.ENCODER.TYPE', 'lsda',
        'MODEL.ENCODER.GROUP_SIZE', '8',
        'MODEL.ENCODER.MLP_RATIO', '2',
        'MODEL.ENCODER.QKV_BIAS', 'False',
    ]
    return opts

def lsda_mini():
    opts = lsda_commons()
    opts.extend([
        'MODEL.ENCODER.NAME', 'lsda_mini',
        'MODEL.ENCODER.EMBED_DIM', '24',
        'MODEL.ENCODER.NUM_HEADS', '(2, 4, 8, 16)',
        'MODEL.ENCODER.DEPTHS', '(3, 6, 12, 3)'
    ])
    return opts

#----------------------------------------------------

def cmt_commons():
    opts = [
        'MODEL.ENCODER.TYPE', 'cmt',
        'MODEL.ENCODER.KV_SCALE', '(8,4,2,1)',
        'MODEL.ENCODER.MLP_RATIO', '4',
        'MODEL.ENCODER.QKV_BIAS', 'False',
    ]
    return opts

def cmt_mini():
    opts = cmt_commons()
    opts.extend([
        'MODEL.ENCODER.NAME', 'cmt_mini',
        'MODEL.ENCODER.EMBED_DIM', '64',
        'MODEL.ENCODER.NUM_HEADS', '(2, 4, 8, 16)',
        'MODEL.ENCODER.DEPTHS', '(2, 2, 6, 2)'
    ])
    return opts

#----------------------------------------------------

def wavelet_commons():
    opts = [
        'MODEL.ENCODER.TYPE', 'wavelet',
        'MODEL.ENCODER.SR_RATIOS', '(4,2,1,1)',
        'MODEL.ENCODER.MLP_RATIO', '4',
        'MODEL.ENCODER.QKV_BIAS', 'False',
    ]
    return opts

def wavelet_mini():
    opts = wavelet_commons()
    opts.extend([
        'MODEL.ENCODER.NAME', 'wavelet _mini',
        'MODEL.ENCODER.EMBED_DIM', '24',
        'MODEL.ENCODER.NUM_HEADS', '(2, 4, 8, 16)',
        'MODEL.ENCODER.DEPTHS', '(1, 3, 7, 1)'
    ])
    return opts

#----------------------------------------------------

def xca_commons():
    opts = [
        'MODEL.ENCODER.TYPE', 'xca',
        'MODEL.ENCODER.MLP_RATIO', '4',
        'MODEL.ENCODER.QKV_BIAS', 'False',
    ]
    return opts

def xca_mini():
    opts = xca_commons()
    opts.extend([
        'MODEL.ENCODER.NAME', 'xca _mini',
        'MODEL.ENCODER.EMBED_DIM', '24',
        'MODEL.ENCODER.NUM_HEADS', '(2, 4, 8, 16)',
        'MODEL.ENCODER.DEPTHS', '(1, 3, 7, 1)'
    ])
    return opts

#----------------------------------------------------

def facta_commons():
    opts = [
        'MODEL.ENCODER.TYPE', 'facta',
        'MODEL.ENCODER.MLP_RATIO', '4',
        'MODEL.ENCODER.QKV_BIAS', 'False',
    ]
    return opts

def facta_mini():
    opts = facta_commons()
    opts.extend([
        'MODEL.ENCODER.NAME', 'facta _mini',
        'MODEL.ENCODER.EMBED_DIM', '24',
        'MODEL.ENCODER.NUM_HEADS', '(2, 4, 8, 16)',
        'MODEL.ENCODER.DEPTHS', '(1, 3, 7, 1)'
    ])
    return opts

#----------------------------------------------------

def sima_commons():
    opts = [
        'MODEL.ENCODER.TYPE', 'sima',
        'MODEL.ENCODER.MLP_RATIO', '2',
        'MODEL.ENCODER.QKV_BIAS', 'False',
    ]
    return opts

def sima_mini():
    opts = sima_commons()
    opts.extend([
        'MODEL.ENCODER.NAME', 'sima _mini',
        'MODEL.ENCODER.EMBED_DIM', '24',
        'MODEL.ENCODER.NUM_HEADS', '(2, 4, 8, 16)',
        'MODEL.ENCODER.DEPTHS', '(3, 6, 12, 3)'
    ])
    return opts

def sima_tiny():
    opts = sima_commons()
    opts.extend([
        'MODEL.ENCODER.NAME', 'sima_tiny',
        'MODEL.ENCODER.EMBED_DIM', '24',
        'MODEL.ENCODER.NUM_HEADS', '(1, 2, 4, 8)',
        'MODEL.ENCODER.DEPTHS', '(1, 3, 7, 1)'
    ])
    return opts

def sima_micro():
    opts = sima_commons()
    opts.extend([
        'MODEL.ENCODER.NAME', 'sima_micro',
        'MODEL.ENCODER.EMBED_DIM', '12',
        'MODEL.ENCODER.NUM_HEADS', '(1, 2, 4, 8)',
        'MODEL.ENCODER.DEPTHS', '(1, 3, 7, 1)'
    ])
    return opts

def sima_nano():
    opts = sima_commons()
    opts.extend([
        'MODEL.ENCODER.NAME', 'sima_nano',
        'MODEL.ENCODER.EMBED_DIM', '8',
        'MODEL.ENCODER.NUM_HEADS', '(1, 2, 4, 8)',
        'MODEL.ENCODER.DEPTHS', '(1, 1, 3, 1)'
    ])
    return opts

#===================== Decoders =====================

def symmetric():
    opts = [
        'MODEL.DECODER.TYPE', 'symmetric',
        'MODEL.DECODER.SKIP_TYPE', 'concat',
        'MODEL.DECODER.EXPAND_FIRST', False,
    ]
    return opts

#----------------------------------------------------

def linear():
    opts = [
        'MODEL.DECODER.TYPE', 'linear',
        'MODEL.DECODER.NAME', 'linear',
    ]
    return opts

#----------------------------------------------------

def mlp():
    opts = [
        'MODEL.DECODER.TYPE', 'mlp',
        'MODEL.DECODER.NAME', 'segformer',
        'MODEL.DECODER.EMBED_DIM', 24,
        'MODEL.DECODER.DEPTH', 4,
        'MODEL.DECODER.FUSE_OP', 'cat',
    ]
    return opts

#----------------------------------------------------

def conv():
    opts = [
        'MODEL.DECODER.TYPE', 'conv',
        'MODEL.DECODER.NAME', 'segformer',
        'MODEL.DECODER.EMBED_DIM', 24,
        'MODEL.DECODER.DEPTH', 4,
        'MODEL.DECODER.FUSE_OP', 'cat',
    ]
    return opts

#----------------------------------------------------

def atrous():
    opts = [
        'MODEL.DECODER.TYPE', 'atrous',
        'MODEL.DECODER.NAME', 'segformer',
        'MODEL.DECODER.EMBED_DIM', 24,
        'MODEL.DECODER.DEPTH', 4,
        'MODEL.DECODER.FUSE_OP', 'cat',
    ]
    return opts

#----------------------------------------------------

def ftn_commons():
    opts = [
        'MODEL.DECODER.TYPE', 'ftn',
        'MODEL.DECODER.DEPTH', 3,
        'MODEL.DECODER.SKIP_TYPE', 'add',
        'MODEL.DECODER.SCALE_FACTOR', 4, 
        'MODEL.DECODER.EXPAND_FIRST', False,
    ]
    return opts

def ftn():
    opts = ftn_commons()
    opts.extend([
        'MODEL.DECODER.NAME', 'ftn',
        'MODEL.DECODER.SKIP_DEPTH', 3,
    ])
    return opts

def ftn8():
    opts = ftn_commons()
    opts.extend([
        'MODEL.DECODER.NAME', 'ftn8',
        'MODEL.DECODER.SKIP_DEPTH', 2,
    ])
    return opts

def ftn16():
    opts = ftn_commons()
    opts.extend([
        'MODEL.DECODER.NAME', 'ftn16',
        'MODEL.DECODER.SKIP_DEPTH', 1,
    ])
    return opts

def ftn32():
    opts = ftn_commons()
    opts.extend([
        'MODEL.DECODER.NAME', 'ftn32',
        'MODEL.DECODER.SKIP_DEPTH', 0,
    ])
    return opts

#----------------------------------------------------

def ftnL_commons():
    opts = [
        'MODEL.DECODER.TYPE', 'ftnL',
        'MODEL.DECODER.SKIP_TYPE', 'add',
    ]
    return opts

def ftnL():
    opts = ftnL_commons()
    opts.extend([
        'MODEL.DECODER.NAME', 'ftnL',
        'MODEL.DECODER.DEPTH', 4,
    ])
    return opts

def ftnL8():
    opts = ftnL_commons()
    opts.extend([
        'MODEL.DECODER.NAME', 'ftnL8',
        'MODEL.DECODER.DEPTH', 3,
    ])
    return opts

def ftnL16():
    opts = ftnL_commons()
    opts.extend([
        'MODEL.DECODER.NAME', 'ftnL16',
        'MODEL.DECODER.DEPTH', 2,
    ])
    return opts

def ftnL32():
    opts = ftnL_commons()
    opts.extend([
        'MODEL.DECODER.NAME', 'ftnL32',
        'MODEL.DECODER.DEPTH', 1,
    ])
    return opts

#----------------------------------------------------

def ftnC_commons():
    opts = [
        'MODEL.DECODER.TYPE', 'ftnC',
        'MODEL.DECODER.SKIP_TYPE', 'add',
    ]
    return opts

def ftnC():
    opts = ftnC_commons()
    opts.extend([
        'MODEL.DECODER.NAME', 'ftnC',
        'MODEL.DECODER.DEPTH', 4,
    ])
    return opts

def ftn8():
    opts = ftnC_commons()
    opts.extend([
        'MODEL.DECODER.NAME', 'ftnC8',
        'MODEL.DECODER.DEPTH', 3,
    ])
    return opts

def ftn16():
    opts = ftnC_commons()
    opts.extend([
        'MODEL.DECODER.NAME', 'ftnC16',
        'MODEL.DECODER.DEPTH', 2,
    ])
    return opts

def ftn32():
    opts = ftnC_commons()
    opts.extend([
        'MODEL.DECODER.NAME', 'ftnC32',
        'MODEL.DECODER.DEPTH', 1,
    ])
    return opts
