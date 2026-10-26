import numpy as np
import torch



def myMetric(outputDigit, outputAnal):
    outputDigitLine = np.reshape(outputDigit, (-1,))
    norm = 3*np.std(outputDigitLine)
    outputDigitNorm = outputDigitLine / norm
    outputAnalNorm = np.reshape(outputAnal, (-1,)) / norm

    signal = np.sqrt(np.mean(np.abs(outputDigitNorm)**2))
    noise = np.sqrt(np.mean(np.abs(outputDigitNorm - outputAnalNorm)**2))
    enob = np.log2(signal/noise)
    rmse = noise
    bit = -np.log2(noise/2)

    return enob, rmse, bit

def myFormuateImage(imageIn, mode='GRAY'):
    image = imageIn
    if mode == 'GRAY':
        image = image.repeat(1, 3, 1, 1)
    image = ((image - image.min()) / (image.max() - image.min()) * 255).to(torch.uint8)

    return image

def LoadState(fileName):
    state = torch.load(fileName, weights_only=True)
    return {key: value for key, value in state.items() if '_quant.' not in key}
