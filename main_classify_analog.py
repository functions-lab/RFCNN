import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat, savemat
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import DataReader_Image, DataReader_DeepSig
from analog import ConvNet
from loss import Eval, AnalyzeFeat
from param import ParseParam
from util import myMetric, LoadState
from Core.GetParam import GetParam



parser = argparse.ArgumentParser(description='Radio-Frequency Convolutional Neural Networks')
parser.add_argument('--data', default='CIFAR-10', type=str, choices=['SVHN', 'CIFAR-10', 'DeepSig'],
    help='dataset name')
parser.add_argument('--size', default=32, type=int,
    help='rescaled data size')
parser.add_argument('--model', default='ConvNet2D', type=str, choices=['ConvNet1D', 'ConvNet2D'],
    help='model type, ConvNet1D for DeepSig')
parser.add_argument('--param', default='VGG11', type=str, choices=['VGG11', 'VGG-DeepSig-new'],
    help='preset model')
parser.add_argument('--exp', action='store_true',
    help='run the hardware experiment on USRPs instead of the default simulation')
parser.add_argument('--snr', default=20, type=float,
    help='SNR in dB of the simulated RF channel, ignored with --exp')
parser.add_argument('--cal', default='./Channel/', type=str,
    help='channel calibration folder used with --exp, empty for skipping')
parser.add_argument('--show', default=100, type=int, help='number of batches to be evaluated')
parser.add_argument('--batch', default=10, type=int, help='batch size')
parser.add_argument('--device', default='gpu', type=str, help='running device')
parser.add_argument('--tokenIn', default='test', type=str, help='model loading token')
parser.add_argument('--tokenOut', default=None, type=str, help='result saving token, default <tokenIn>-sim or <tokenIn>-exp')



if __name__ == '__main__':
    np.random.seed(42)

    opt = parser.parse_args()
    DATA = opt.data
    SIZE = opt.size
    MODEL = opt.model
    PARAM = opt.param
    SNR = None if opt.exp else opt.snr
    CAL = None if (not opt.exp) or len(opt.cal) == 0 else opt.cal
    print('Mode: %s'%('hardware experiment' if opt.exp else 'simulation, SNR=%.1fdB'%(SNR)))
    SHOW = opt.show
    BATCH = opt.batch
    DEVICE = opt.device
    TOKEN_IN = opt.tokenIn
    TOKEN_OUT = opt.tokenOut

    device = torch.device('cuda' if torch.cuda.is_available() and DEVICE=='gpu' else 'cpu')
    param = GetParam()
    param['device'] = device
    if DATA == 'SVHN':
        engineList = [-15, -10, -5, -5, -5, -5, -5, -5, 0]
    elif DATA == 'CIFAR-10':
        engineList = [-15, -15, -10, -10, -5, -5, -5, -5, -5]
    elif DATA == 'DeepSig':
        engineList = [None, -25, -15, -10, -10, -5]

    if DATA == 'DeepSig':
        testSet = DataReader_DeepSig(dataPath='../Data/DeepSig/', waveLen=SIZE, span=[0.9, 1.0],
            moduSet=list(range(24)), snrSet=[30])
    else:
        testSet = DataReader_Image(dataset=DATA, phase='test', imageSize=SIZE)
    dataShape, classNum = testSet.GetShape()
    g = torch.Generator()
    g.manual_seed(42)
    testLoader = DataLoader(dataset=testSet, num_workers=4, batch_size=BATCH, shuffle=True, generator=g)

    paramDict = ParseParam(PARAM)
    ClassifierModel = ConvNet(dataShape=dataShape, classNum=classNum,
        channelList=paramDict['channelList'], kernelList=paramDict['kernelList'], poolList=paramDict['poolList'],
        is1D=(MODEL=='ConvNet1D'), param=param, calFolder=CAL)
    macNum, paramNum = ClassifierModel.GetInfo()
    print('Model MAC number: %d, param number: %d'%(macNum, paramNum))
    print('Model parameter number:', [param.size() for param in ClassifierModel.parameters()], \
        'Total:', sum(param.numel() for param in ClassifierModel.parameters()))

    if TOKEN_IN is not None:
        resultInPath = '../Result/'+DATA+'_size' + str(SIZE) + '_' + TOKEN_IN + '/'
        try:
            ClassifierModel.load_state_dict(LoadState(resultInPath+'model_accMax.pth'))
            print('Model Loaded from '+resultInPath)
        except:
            print('Warning: Model Loading Failed!')

    resultOutPath = '../Result/'+DATA+'_size' + str(SIZE) + '_' + (TOKEN_OUT if TOKEN_OUT is not None else TOKEN_IN+('-exp' if opt.exp else '-sim')) + '/'
    os.makedirs(resultOutPath+'Details/', exist_ok=True)
    print('Model Saved to '+resultOutPath)

    ClassifierLoss = torch.nn.CrossEntropyLoss()
    ClassifierEval = Eval(classNum=classNum)

    ClassifierModel.to(device)

    ClassifierModel.eval()
    ClassifierModel.AnalogInit(batchNum=BATCH)
    _, data = next(iter(testLoader))
    zoomList = ClassifierModel.AnalogZoom(data, engineList=engineList, snr=SNR)
    print(zoomList)

    keyList = ['lossDigit', 'accDigit', 'lossAnal', 'accAnal', 'enob', 'rmse', 'bit']
    result = {key: 0 for key in keyList}
    result.update({'dataNum': 0, 'matDigit': np.zeros((classNum, classNum)), 'matAnal': np.zeros((classNum, classNum))})
    showIdx = 0
    testBar = tqdm(testLoader)
    for target, data in testBar:
        dataNum = target.size(0)
        result['dataNum'] += dataNum

        showIdx += 1
        if showIdx > SHOW:
            break
        fileInfo = resultOutPath+'Details/info_'+str(showIdx)+'.mat'
        fileDigit = resultOutPath+'Details/feat_digit_'+str(showIdx)+'.mat'
        fileAnal = resultOutPath+'Details/feat_anal_'+str(showIdx)+'.mat'
        if os.path.exists(fileInfo) and os.path.exists(fileDigit) and os.path.exists(fileAnal):
            info = loadmat(fileInfo)
        else:
            target = target.to(device)
            data = data.to(device)

            predDigit, featDigit = ClassifierModel.AnalogInfer(data, engineList=[None], snr=SNR)
            lossDigit = ClassifierLoss(predDigit, target).item()
            accDigit, matDigit = ClassifierEval(
                predDigit.detach().cpu().numpy(),
                target.detach().cpu().numpy())
            savemat(fileDigit, AnalyzeFeat(featDigit))

            predAnal, featAnal = ClassifierModel.AnalogInfer(data, engineList=engineList, snr=SNR)
            lossAnal = ClassifierLoss(predAnal, target).item()
            accAnal, matAnal = ClassifierEval(
                predAnal.detach().cpu().numpy(),
                target.detach().cpu().numpy())
            savemat(fileAnal, AnalyzeFeat(featAnal))

            enob, rmse, bit = myMetric(predDigit.detach().cpu().numpy(), predAnal.detach().cpu().numpy())
            info = {
                'lossDigit': lossDigit, 'accDigit': accDigit, 'matDigit': matDigit,
                'lossAnal': lossAnal, 'accAnal': accAnal, 'matAnal': matAnal,
                'enob': enob, 'rmse': rmse, 'bit': bit}
            savemat(fileInfo, info)

        for key in keyList:
            result[key] += np.squeeze(info[key]) * dataNum
        result['matDigit'] += info['matDigit']
        result['matAnal'] += info['matAnal']
        resultList = {key: result[key]/result['dataNum'] for key in keyList}
        resultList.update({'matDigit': result['matDigit'], 'matAnal': result['matAnal']})

        testBar.set_description(desc='Loss: %.4f-%.4f Acc: %.4f-%.4f, ENOB: %.2f RMSE: %.4f Bit: %.4f'
            % (resultList['lossDigit'], resultList['lossAnal'],
            resultList['accDigit'], resultList['accAnal'],
            resultList['enob'], resultList['rmse'], resultList['bit']))

        for name in ['Digit', 'Anal']:
            fig, ax = plt.subplots()
            ax.imshow(resultList['mat'+name])
            ax.set_title('acc=%.2f'%(resultList['acc'+name]))
            fig.savefig(resultOutPath + 'mat_%s.png'%(name.lower()))
            plt.close(fig)
        savemat(resultOutPath+'result_anal.mat', resultList)
