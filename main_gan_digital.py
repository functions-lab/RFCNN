import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import savemat
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import itertools
from torchvision.utils import save_image
from torchmetrics.image.fid import FrechetInceptionDistance

from data import DataReader_Image, DataReader_CelebA
from analog import Generator, Discriminator
from param import ParseParam
from util import myFormuateImage
from Core.GetParam import GetParam



parser = argparse.ArgumentParser(description='Radio-Frequency Convolutional Neural Networks')
parser.add_argument('--data', default='CelebA', type=str, choices=['MNIST', 'FMNIST', 'CelebA'],
    help='dataset name')
parser.add_argument('--size', default=64, type=int,
    help='rescaled data size')
parser.add_argument('--param', default='RGB', type=str, choices=['GRAY', 'RGB', 'High'],
    help='preset model')
parser.add_argument("--latent", type=int, default=62,
    help="dimensionality of the latent space")
parser.add_argument("--code", type=int, default=2,
    help="latent code")
parser.add_argument('--epoch', default=200, type=int, help='train epoch number')
parser.add_argument('--batch', default=64, type=int, help='batch size')
parser.add_argument('--learn', default=2e-4, type=float, help='learning rate')
parser.add_argument('--exam', default=1000, type=int, help='number of examined images to calculate FID')
parser.add_argument('--show', default=10, type=int, help='number of showing examples per class')
parser.add_argument('--device', default='gpu', type=str, help='running device')
parser.add_argument('--tokenOut', default='test', type=str, help='model saving token')



if __name__ == '__main__':
    opt = parser.parse_args()
    DATA = opt.data
    SIZE = opt.size
    PARAM = opt.param
    LATENT = opt.latent
    CODE = opt.code
    EPOCH = opt.epoch
    BATCH = opt.batch
    LEARN = opt.learn
    EXAM = opt.exam
    SHOW = opt.show
    DEVICE = opt.device
    TOKEN_OUT = opt.tokenOut

    device = torch.device('cuda' if torch.cuda.is_available() and DEVICE=='gpu' else 'cpu')
    param = GetParam()

    if DATA == 'CelebA':
        trainSet = DataReader_CelebA(phase='train', imageSize=SIZE)
    else:
        trainSet = DataReader_Image(dataset=DATA, phase='train', imageSize=SIZE)
    dataShape, classNum = trainSet.GetShape()
    trainLoader = DataLoader(dataset=trainSet, num_workers=4, batch_size=BATCH, shuffle=True)

    paramDictGen = ParseParam(PARAM+'-gen')
    generator = Generator(dataShape=dataShape, classNum=classNum, latent=LATENT, code=CODE,
        channelList=paramDictGen['channelList'], kernelList=paramDictGen['kernelList'], upsampleList=paramDictGen['upsampleList'],
        actList=paramDictGen['actList'], param=param, calFolder=None)
    macNum, paramNum = generator.GetInfo()
    print('Model MAC number: %d, param number: %d'%(macNum, paramNum))
    print('Model parameter number:', [param.size() for param in generator.parameters()], \
        'Total:', sum(param.numel() for param in generator.parameters()))
    paramDictDis = ParseParam(PARAM+'-dis')
    discriminator = Discriminator(dataShape=dataShape, classNum=classNum, code=CODE,
        channelList=paramDictDis['channelList'], kernelList=paramDictDis['kernelList'], strideList=paramDictDis['strideList'], batchList=paramDictDis['batchList'])
    optimizerGen = optim.Adam(generator.parameters(), lr=LEARN, betas=(0.5, 0.999))
    optimizerDis = optim.Adam(discriminator.parameters(), lr=LEARN, betas=(0.5, 0.999))
    optimizerInfo = optim.Adam(itertools.chain(generator.parameters(), discriminator.parameters()), lr=LEARN, betas=(0.5, 0.999))

    resultOutPath = '../Result/'+DATA+'_size' + str(SIZE) + '_' + TOKEN_OUT + '/'
    if not os.path.exists(resultOutPath):
        os.makedirs(resultOutPath)
    print('Model Saved to '+resultOutPath)

    LossAdv = torch.nn.MSELoss()
    LossCate = torch.nn.CrossEntropyLoss()
    LossCont = torch.nn.MSELoss()
    lambdaCat = 1
    lambdaCont = 0.1

    seedTest = torch.zeros((SHOW*classNum, LATENT), device=device)
    labelTest = torch.tensor([num for num in range(classNum) for _ in range(SHOW)], device=device)
    codeTest = torch.zeros((SHOW*classNum, CODE), device=device)

    generator.to(device)
    discriminator.to(device)
    FID = FrechetInceptionDistance(feature=2048).to(device)

    resultList = {
        'lossGen': np.ones((EPOCH))*np.nan, 'lossDis': np.ones((EPOCH))*np.nan, 'lossInfo': np.ones((EPOCH))*np.nan, 
        'fid': np.ones((EPOCH))*np.nan}
    fidMin = np.inf
    for epochIdx in range(EPOCH):
        trainBar = tqdm(trainLoader)
        result = {'dataNum': 0, 'lossGen': 0, 'lossDis': 0, 'lossInfo': 0, 'fid': np.nan}

        imageRealAll = []
        imageFakeAll = []
        for _, data in trainBar:
            dataNum = data.size(0)
            result['dataNum'] += dataNum

            data = data.to(device)

            validReal = torch.ones((dataNum, 1), device=device)
            validFake = torch.zeros((dataNum, 1), device=device)
            imageReal = data

            optimizerGen.zero_grad()
            seed = torch.randn((dataNum, LATENT), device=device)
            targetFake = torch.randint(0, classNum, (dataNum,), device=device)
            codeFake = torch.rand((dataNum, CODE), device=device)*2-1
            imageFake = generator(seed, targetFake, codeFake)
            predFake, _, _ = discriminator(imageFake)
            lossGen = LossAdv(predFake, validReal)
            lossGen.backward()
            optimizerGen.step()

            optimizerDis.zero_grad()
            predReal, _, _ = discriminator(imageReal)
            lossDisReal = LossAdv(predReal, validReal)
            predFake, _, _ = discriminator(imageFake.detach())
            lossDisFake = LossAdv(predFake, validFake)
            lossDis = (lossDisReal + lossDisFake) / 2
            lossDis.backward()
            optimizerDis.step()

            optimizerInfo.zero_grad()
            targetInfo = torch.randint(0, classNum, (dataNum,), device=device)
            seedInfo = torch.randn((dataNum, LATENT), device=device)
            codeInfo = torch.rand((dataNum, CODE), device=device)*2-1

            imageInfo = generator(seedInfo, targetInfo, codeInfo)
            _, predTarget, predCode = discriminator(imageInfo)
            lossInfo = lambdaCat * LossCate(predTarget, targetInfo) + lambdaCont * LossCont(predCode, codeInfo)

            lossInfo.backward()
            optimizerInfo.step()

            if len(imageRealAll) > EXAM//BATCH+1:
                imageRealAll.pop(0)
                imageFakeAll.pop(0)
            imageRealAll.append(imageReal.detach())
            imageFakeAll.append(imageFake.detach())

            result['lossGen'] += lossGen.item() * dataNum
            result['lossDis'] += lossDis.item() * dataNum
            result['lossInfo'] += lossInfo.item() * dataNum
            trainBar.set_description(desc='[%d/%d] G Loss: %.4f D Loss: %.4f Info Loss: %.4f'
                % (epochIdx, EPOCH, 
                result['lossGen']/result['dataNum'], 
                result['lossDis']/result['dataNum'],
                result['lossInfo']/result['dataNum']))

        imageRealAll = torch.cat(imageRealAll, dim=0)
        imageFakeAll = torch.cat(imageFakeAll, dim=0)
        if imageRealAll.size()[0] > EXAM:
            imageRealAll = imageRealAll[-EXAM:, :, :, :]
            imageFakeAll = imageFakeAll[-EXAM:, :, :, :]

        FID.update(myFormuateImage(imageRealAll, mode=PARAM), real=True)
        FID.update(myFormuateImage(imageFakeAll, mode=PARAM), real=False)
        fid = FID.compute()
        FID.reset()
        result['fid'] = fid
        print('[%d/%d] FID: %.4f'%(epochIdx, EPOCH, result['fid']))

        resultList['lossGen'][epochIdx] = result['lossGen']/result['dataNum']
        resultList['lossDis'][epochIdx] = result['lossDis']/result['dataNum']
        resultList['lossInfo'][epochIdx] = result['lossInfo']/result['dataNum']
        resultList['fid'][epochIdx] = result['fid']

        fig, ax = plt.subplots()
        ax.plot(np.arange(0, EPOCH), resultList['lossGen'], label='Generator Loss')
        ax.plot(np.arange(0, EPOCH), resultList['lossDis'], label='Discriminator Loss')
        ax.plot(np.arange(0, EPOCH), resultList['lossInfo'], label='Info Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        fig.savefig(resultOutPath + 'loss.png')
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(np.arange(0, EPOCH), resultList['fid'], label='FID')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('FID')
        ax.legend()
        fig.savefig(resultOutPath + 'fid.png')
        plt.close(fig)

        seedRand = torch.randn((SHOW*classNum, LATENT), device=device)
        imageFakeRand = generator(seedRand, labelTest, codeTest)
        save_image(imageFakeRand.data, resultOutPath+'Rand_%d.png'%epochIdx, nrow=SHOW, normalize=True)

        codeZero = torch.zeros((SHOW*classNum, CODE), device=device)
        codeVary = torch.linspace(-1, 1, SHOW, device=device).repeat(1, classNum)
        for codeIdx in range(CODE):
            codeTest = codeZero.clone()
            codeTest[:, codeIdx] = codeVary
            imageFakeTest = generator(seedTest, labelTest, codeTest)
            save_image(imageFakeTest.data, resultOutPath+'Test_%d_%d.png'%(codeIdx, epochIdx), nrow=SHOW, normalize=True)

        generator.cpu()
        torch.save(generator.state_dict(), resultOutPath+'model_Gen_last.pth')
        if fidMin > fid:
            fidMin = fid
            torch.save(generator.state_dict(), resultOutPath+'model_Gen_opt.pth')
        generator.to(device)
        savemat(resultOutPath+'result.mat', resultList)
