import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import savemat
import torch

from data import DataReader_Image, DataReader_CelebA
from analog import Generator
from loss import AnalyzeFeat
from param import ParseParam
from util import LoadState
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
parser.add_argument('--exp', action='store_true',
    help='run the hardware experiment on USRPs instead of the default simulation')
parser.add_argument('--snr', default=20, type=float,
    help='SNR in dB of the simulated RF channel, ignored with --exp')
parser.add_argument('--cal', default='./Channel/', type=str,
    help='channel calibration folder used with --exp, empty for skipping')
parser.add_argument('--show', default=10, type=int, help='number of showing examples per class')
parser.add_argument('--batch', default=10, type=int, help='batch size')
parser.add_argument('--device', default='gpu', type=str, help='running device')
parser.add_argument('--tokenIn', default='test', type=str, help='model loading token')
parser.add_argument('--tokenOut', default=None, type=str, help='result saving token, default <tokenIn>-sim or <tokenIn>-exp')



if __name__ == '__main__':
    opt = parser.parse_args()
    DATA = opt.data
    SIZE = opt.size
    PARAM = opt.param
    LATENT = opt.latent
    CODE = opt.code
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
    engineList = {
        'GRAY': [-5, -5, -10, -10],
        'RGB': [0, 0, -10, -10, -10, -15],
        'High': [-5, -5, -10, -10, -15, -15, -15]}[PARAM]

    if DATA == 'CelebA':
        trainSet = DataReader_CelebA(phase='train', imageSize=SIZE)
    else:
        trainSet = DataReader_Image(dataset=DATA, phase='train', imageSize=SIZE)
    dataShape, classNum = trainSet.GetShape()
    colorNum = dataShape[0]

    paramDictGen = ParseParam(PARAM+'-gen')
    generator = Generator(dataShape=dataShape, classNum=classNum, latent=LATENT, code=CODE,
        channelList=paramDictGen['channelList'], kernelList=paramDictGen['kernelList'], upsampleList=paramDictGen['upsampleList'],
        actList=paramDictGen['actList'], param=param, calFolder=CAL)
    macNum, paramNum = generator.GetInfo()
    print('Model MAC number: %d, param number: %d'%(macNum, paramNum))
    print('Model parameter number:', [param.size() for param in generator.parameters()], \
        'Total:', sum(param.numel() for param in generator.parameters()))

    if TOKEN_IN is not None:
        resultInPath = '../Result/'+DATA+'_size' + str(SIZE) + '_' + TOKEN_IN + '/'
        try:
            generator.load_state_dict(LoadState(resultInPath+'model_Gen_opt.pth'))
            print('Model Loaded from '+resultInPath)
        except:
            print('Warning: Model Loading Failed!')

    resultOutPath = '../Result/'+DATA+'_size' + str(SIZE) + '_' + (TOKEN_OUT if TOKEN_OUT is not None else TOKEN_IN+('-exp' if opt.exp else '-sim')) + '/'
    os.makedirs(resultOutPath+'Details/', exist_ok=True)
    print('Model Saved to '+resultOutPath)

    generator.eval()
    generator.AnalogInit(batchNum=BATCH)

    seedTest = torch.zeros((SHOW*classNum, LATENT), device=device)
    labelTest = torch.tensor([num for num in range(classNum) for _ in range(SHOW)], device=device)
    codeTest = torch.zeros((SHOW*classNum, CODE), device=device)
    zoomList = generator.AnalogZoom(seedTest[:BATCH, :], labelTest[:BATCH], codeTest[:BATCH, :], engineList=engineList, snr=SNR)

    codeZero = torch.zeros((SHOW*classNum, CODE), device=device)
    codeVary = torch.linspace(-1, 1, SHOW, device=device).repeat(1, classNum)
    for codeIdx in range(CODE):
        codeTest = codeZero.clone()
        codeTest[:, codeIdx] = codeVary

        imageDict = {name: np.ones((colorNum, SHOW*classNum, SIZE, SIZE)) * np.nan for name in ['digit', 'anal']}
        for iterIdx in range(SHOW*classNum//BATCH):
            label = labelTest[iterIdx*BATCH: (iterIdx+1)*BATCH]
            code = codeTest[iterIdx*BATCH: (iterIdx+1)*BATCH, :]
            seed = seedTest[iterIdx*BATCH: (iterIdx+1)*BATCH, :]

            for name, engineListNow in [('digit', [None]), ('anal', engineList)]:
                imageFake, feat = generator.AnalogInfer(seed, label, code, engineList=engineListNow, snr=SNR)
                image = imageDict[name]
                image[:, iterIdx*BATCH: (iterIdx+1)*BATCH, :, :] = np.swapaxes(imageFake.cpu().detach().numpy(), 0, 1)
                savemat(resultOutPath+'Details/Test_%d_%s_feat_%d.mat'%(codeIdx, name, iterIdx), AnalyzeFeat(feat))

                fig, ax = plt.subplots()
                imageTemp = np.reshape(np.reshape(image, (colorNum, SHOW, classNum, SIZE, SIZE)).swapaxes(2, 3), (colorNum, SHOW*SIZE, classNum*SIZE))
                imageTemp = (np.transpose(imageTemp, (1, 2, 0))+1)/2
                ax.imshow(np.clip(imageTemp, a_min=0, a_max=1))
                fig.savefig(resultOutPath+'Test_%d_%s.png'%(codeIdx, name))
                plt.close(fig)

            savemat(resultOutPath+'Test_%d.mat'%(codeIdx), {
                'imageDigit': np.reshape(imageDict['digit'], (colorNum, SHOW, classNum, SIZE, SIZE)),
                'imageAnal': np.reshape(imageDict['anal'], (colorNum, SHOW, classNum, SIZE, SIZE))})
