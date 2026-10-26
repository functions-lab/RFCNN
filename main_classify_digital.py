import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import savemat
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import DataReader_Image, DataReader_DeepSig
from analog import ConvNet
from loss import Eval
from param import ParseParam
from util import LoadState
from Core.GetParam import GetParam



parser = argparse.ArgumentParser(description='Radio-Frequency Convolutional Neural Networks')
parser.add_argument('--data', default='CIFAR-10', type=str, choices=['MNIST', 'FMNIST', 'SVHN', 'CIFAR-10', 'DeepSig'],
    help='dataset name')
parser.add_argument('--size', default=32, type=int,
    help='rescaled data size')
parser.add_argument('--model', default='ConvNet2D', type=str, choices=['ConvNet1D', 'ConvNet2D'],
    help='model type, ConvNet1D for DeepSig')
parser.add_argument('--param', default='VGG11', type=str, choices=['VGG11', 'VGG-DeepSig-new'],
    help='preset model')
parser.add_argument('--epoch', default=200, type=int, help='train epoch number')
parser.add_argument('--batch', default=256, type=int, help='batch size')
parser.add_argument('--learn', default=1e-2, type=float, help='learning rate')
parser.add_argument('--decay', default=1e-3, type=float, help='weight decay')
parser.add_argument('--device', default='gpu', type=str, help='running device')
parser.add_argument('--tokenIn', default=None, type=str, help='model loading token')
parser.add_argument('--tokenOut', default='test', type=str, help='model saving token')



if __name__ == '__main__':
    opt = parser.parse_args()
    DATA = opt.data
    SIZE = opt.size
    MODEL = opt.model
    PARAM = opt.param
    EPOCH = opt.epoch
    BATCH = opt.batch
    LEARN = opt.learn
    DECAY = opt.decay
    DEVICE = opt.device
    TOKEN_IN = opt.tokenIn
    TOKEN_OUT = opt.tokenOut

    device = torch.device('cuda' if torch.cuda.is_available() and DEVICE=='gpu' else 'cpu')
    param = GetParam()

    if DATA == 'DeepSig':
        trainSet = DataReader_DeepSig(dataPath='../Data/DeepSig/', waveLen=SIZE, span=[0, 0.9],
            moduSet=list(range(24)), snrSet=list(range(10, 32, 2)))
        testSet = DataReader_DeepSig(dataPath='../Data/DeepSig/', waveLen=SIZE, span=[0.9, 1.0],
            moduSet=list(range(24)), snrSet=[30])
    else:
        trainSet = DataReader_Image(dataset=DATA, phase='train', imageSize=SIZE)
        testSet = DataReader_Image(dataset=DATA, phase='test', imageSize=SIZE)
    dataShape, classNum = testSet.GetShape()
    trainLoader = DataLoader(dataset=trainSet, num_workers=4, batch_size=BATCH, shuffle=True)
    testLoader = DataLoader(dataset=testSet, num_workers=4, batch_size=16, shuffle=False)

    paramDict = ParseParam(PARAM)
    ClassifierModel = ConvNet(dataShape=dataShape, classNum=classNum,
        channelList=paramDict['channelList'], kernelList=paramDict['kernelList'], poolList=paramDict['poolList'],
        is1D=(MODEL=='ConvNet1D'), param=param, calFolder=None)
    macNum, paramNum = ClassifierModel.GetInfo()
    print('Model MAC number: %d, param number: %d'%(macNum, paramNum))
    print('Model parameter number:', [param.size() for param in ClassifierModel.parameters()], \
        'Total:', sum(param.numel() for param in ClassifierModel.parameters()))
    print('Model structure:\n', ClassifierModel)
    optimizer = optim.SGD(ClassifierModel.parameters(), lr=LEARN,
        momentum=0.9, weight_decay=DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)

    isModelLoad = False
    if TOKEN_IN is not None:
        resultInPath = '../Result/'+DATA+'_size' + str(SIZE) + '_' + TOKEN_IN + '/'
        try:
            ClassifierModel.load_state_dict(LoadState(resultInPath+'model_accMax.pth'))
            isModelLoad = True
            print('Model Loaded from '+resultInPath)
        except:
            print('Warning: Model Loading Failed!')

    resultOutPath = '../Result/'+DATA+'_size' + str(SIZE) + '_' + TOKEN_OUT + '/'
    if not os.path.exists(resultOutPath):
        os.makedirs(resultOutPath)
    print('Model Saved to '+resultOutPath)

    ClassifierLoss = torch.nn.CrossEntropyLoss()
    ClassifierEval = Eval(classNum=classNum)

    ClassifierModel.to(device)

    resultList = {
        'lossTrain': np.ones((EPOCH))*np.nan, 'accTrain': np.ones((EPOCH))*np.nan,
        'matTrain': np.ones((EPOCH, classNum, classNum))*np.nan,
        'lossTest': np.ones((EPOCH))*np.nan, 'accTest': np.ones((EPOCH))*np.nan,
        'matTest': np.ones((EPOCH, classNum, classNum))*np.nan}
    accMax = -np.inf
    for epochIdx in range(EPOCH):

        if (not isModelLoad) or (epochIdx > 0):
            ClassifierModel.train()
            result = {'dataNum': 0, 'loss': 0, 'acc': 0, 'mat': np.zeros((classNum, classNum))}
            trainBar = tqdm(trainLoader)
            for target, data in trainBar:
                dataNum = target.size(0)
                result['dataNum'] += dataNum

                target = target.to(device)
                data = data.to(device)

                optimizer.zero_grad()
                pred = ClassifierModel(data)
                loss = ClassifierLoss(pred, target)
                acc, mat = ClassifierEval(
                    pred.detach().cpu().numpy(),
                    target.detach().cpu().numpy())
                loss.backward()
                optimizer.step()
                scheduler.step()

                result['loss'] += loss.item() * dataNum
                result['acc'] += acc * dataNum
                result['mat'] += mat

                trainBar.set_description(desc='[%d/%d] Loss: %.4f Acc: %.4f'
                    % (epochIdx, EPOCH,
                    result['loss'] / result['dataNum'],
                    result['acc'] / result['dataNum']))

            resultList['lossTrain'][epochIdx] = result['loss']/result['dataNum']
            resultList['accTrain'][epochIdx] = result['acc']/result['dataNum']
            resultList['matTrain'][epochIdx, :, :] = result['mat']

        ClassifierModel.eval()
        result = {'dataNum': 0, 'loss': 0, 'acc': 0, 'mat': np.zeros((classNum, classNum))}
        testBar = tqdm(testLoader)
        for target, data in testBar:
            dataNum = target.size(0)
            result['dataNum'] += dataNum

            target = target.to(device)
            data = data.to(device)

            pred = ClassifierModel(data)
            loss = ClassifierLoss(pred, target)
            acc, mat = ClassifierEval(
                pred.detach().cpu().numpy(),
                target.detach().cpu().numpy())

            result['loss'] += loss.item() * dataNum
            result['acc'] += acc * dataNum
            result['mat'] += mat

            testBar.set_description(desc='[%d/%d] Loss: %.4f Acc: %.4f'
                % (epochIdx, EPOCH,
                result['loss'] / result['dataNum'],
                result['acc'] / result['dataNum']))
        resultList['lossTest'][epochIdx] = result['loss']/result['dataNum']
        resultList['accTest'][epochIdx] = result['acc']/result['dataNum']
        resultList['matTest'][epochIdx, :, :] = result['mat']

        fig, ax = plt.subplots()
        ax.plot(np.arange(0, EPOCH), resultList['lossTrain'])
        ax.plot(np.arange(0, EPOCH), resultList['lossTest'])
        fig.savefig(resultOutPath + 'loss.png')
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(np.arange(0, EPOCH), resultList['accTrain'])
        ax.plot(np.arange(0, EPOCH), resultList['accTest'])
        fig.savefig(resultOutPath + 'acc.png')
        plt.close(fig)

        if accMax < resultList['accTest'][epochIdx]:
            accMax = resultList['accTest'][epochIdx]
            ClassifierModel.cpu()
            torch.save(ClassifierModel.state_dict(), resultOutPath+'model_accMax.pth')
            ClassifierModel.to(device)
            fig, ax = plt.subplots()
            ax.imshow(resultList['matTest'][epochIdx, :, :])
            fig.savefig(resultOutPath + 'mat_accMax.png')
            plt.close(fig)
        savemat(resultOutPath+'result.mat', resultList)
