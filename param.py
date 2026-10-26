PARAM_DICT = {
    'VGG11': {
        'channelList': [64, 128, 256, 256, 512, 512, 512, 512],
        'kernelList': [3, 3, 3, 3, 3, 3, 3, 3],
        'poolList': [2, 2, 1, 2, 1, 2, 1, 2]},
    'VGG-DeepSig-new': {
        'channelList': [256, 256, 512, 512, 512],
        'kernelList': [3, 3, 3, 3, 3],
        'poolList': [4, 4, 4, 4, 4]},

    'GRAY-gen': {
        'channelList': [128, 128, 64],
        'kernelList': [3, 3, 3],
        'upsampleList': [2, 2, -1],
        'actList': ['leaky', 'leaky', 'tanh']},
    'RGB-gen': {
        'channelList': [1024, 512, 256, 128, 64],
        'kernelList': [3, 3, 3, 3, 3],
        'upsampleList': [2, 2, 2, 2, -1],
        'actList': ['leaky', 'leaky', 'leaky', 'leaky', 'tanh']},
    'High-gen': {
        'channelList': [1024, 512, 256, 128, 64, 32],
        'kernelList': [3, 3, 3, 3, 3, 3],
        'upsampleList': [2, 2, 2, 2, 2, -1],
        'actList': ['leaky', 'leaky', 'leaky', 'leaky', 'leaky', 'tanh']},

    'GRAY-dis': {
        'channelList': [16, 32, 64, 128],
        'kernelList': [3, 3, 3, 3],
        'strideList': [2, 2, 2, 2],
        'batchList': [False, True, True, True]},
    'RGB-dis': {
        'channelList': [64, 128, 256, 512],
        'kernelList': [3, 3, 3, 3],
        'strideList': [2, 2, 2, 2],
        'batchList': [False, True, True, True]},
    'High-dis': {
        'channelList': [64, 128, 256, 512, 1024],
        'kernelList': [3, 3, 3, 3, 3],
        'strideList': [2, 2, 2, 2, 2],
        'batchList': [False, True, True, True, True]},
}

def ParseParam(paramStr):
    return PARAM_DICT[paramStr]
