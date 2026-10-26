import argparse
import numpy as np
import os
from scipy.io import loadmat
import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance

from data import DataReader_Image, DataReader_CelebA
from util import myFormuateImage



parser = argparse.ArgumentParser(description='Radio-Frequency Convolutional Neural Networks')
parser.add_argument('--data', default='CelebA', type=str, choices=['MNIST', 'FMNIST', 'CelebA'],
    help='dataset name')
parser.add_argument('--size', default=64, type=int,
    help='rescaled data size')
parser.add_argument("--code", type=int, default=2,
    help="latent code")
parser.add_argument('--show', default=10, type=int, help='number of showing examples per class')
parser.add_argument('--device', default='gpu', type=str, help='running device')
parser.add_argument('--tokenOut', default=None, type=str, help='result token of main_gan_analog.py')



if __name__ == '__main__':
    opt = parser.parse_args()
    DATA = opt.data
    SIZE = opt.size
    CODE = opt.code
    SHOW = opt.show
    DEVICE = opt.device
    TOKEN_OUT = opt.tokenOut

    device = torch.device('cuda' if torch.cuda.is_available() and DEVICE=='gpu' else 'cpu')

    if DATA == 'CelebA':
        trainSet = DataReader_CelebA(phase='train', imageSize=SIZE)
    else:
        trainSet = DataReader_Image(dataset=DATA, phase='train', imageSize=SIZE)
    dataShape, classNum = trainSet.GetShape()
    trainLoader = DataLoader(dataset=trainSet, num_workers=4, batch_size=SHOW*classNum, shuffle=True)
    colorNum = dataShape[0]
    mode = 'GRAY' if colorNum == 1 else 'RGB'

    resultOutPath = '../Result/'+DATA+'_size' + str(SIZE) + '_' + TOKEN_OUT + '/'
    if not os.path.exists(resultOutPath):
        print('Warning: Result Path NOT Found!')

    fid = FrechetInceptionDistance(feature=2048).to(device)

    _, imageReal = next(iter(trainLoader))
    imageReal = imageReal.to(device)

    for codeIdx in range(CODE):
        imageFakeDict = loadmat(resultOutPath+'Test_%d.mat'%(codeIdx))
        for name in ['Digit', 'Anal']:
            imageFake = np.reshape(imageFakeDict['image'+name], (colorNum, SHOW*classNum, SIZE, SIZE)).swapaxes(0, 1)
            imageFake = torch.tensor(imageFake, dtype=torch.float32, device=device)

            fid.update(myFormuateImage(imageReal, mode=mode), real=True)
            fid.update(myFormuateImage(imageFake, mode=mode), real=False)
            print('Code %d, %s FID: %.4f'%(codeIdx, name, fid.compute()))
            fid.reset()
