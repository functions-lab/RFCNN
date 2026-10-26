import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Core.interface_torch import ConvAcc, FCAcc



def WeightInit(layer, method='kaiming'):
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        if method == 'kaiming':
            nn.init.kaiming_normal_(layer.weight)
        elif method == 'normal':
            nn.init.normal_(layer.weight, 0.0, 0.02)

def GetEngine(engineList, layerIdx):
    return engineList[min(layerIdx, len(engineList)-1)]

class OutputLayer(nn.Module):
    def forward(self, x_0):
        return torch.real(x_0)

class ReLULayer(nn.Module):
    def forward(self, x_0):
        return torch.complex(F.relu(torch.real(x_0)), F.relu(torch.imag(x_0)))

class LeakyLayer(nn.Module):
    def __init__(self, scale=0.2):
        super(LeakyLayer, self).__init__()

        self.Leaky = nn.LeakyReLU(negative_slope=scale)

    def forward(self, x_0):
        return torch.complex(self.Leaky(torch.real(x_0)), self.Leaky(torch.imag(x_0)))

class TanhLayer(nn.Module):
    def forward(self, x_0):
        return torch.complex(F.tanh(torch.real(x_0)), F.tanh(torch.imag(x_0)))

class ACTLayer(nn.Module):
    def __init__(self, act='relu'):
        super(ACTLayer, self).__init__()

        if act == 'relu':
            self.activation = ReLULayer()
        elif act == 'leaky':
            self.activation = LeakyLayer(scale=0.2)
        elif act == 'tanh':
            self.activation = TanhLayer()
        elif act == 'none':
            self.activation = nn.Identity()
        elif act == 'output':
            self.activation = OutputLayer()

    def forward(self, x_0):
        return self.activation(x_0)

class Pool2DLayer(nn.Module):
    def __init__(self, kernelX=2, kernelY=2):
        super(Pool2DLayer, self).__init__()

        self.Pool = nn.MaxPool2d(kernel_size=[kernelX, kernelY], stride=[kernelX, kernelY])

    def forward(self, x_0):
        return self.Pool(torch.real(x_0)) + 1j * self.Pool(torch.imag(x_0))

class PixelShuffleLayer(nn.Module):
    def __init__(self, pixel=2):
        super(PixelShuffleLayer, self).__init__()

        self.PixelShuffle = nn.PixelShuffle(upscale_factor=pixel)

    def forward(self, x_0):
        return self.PixelShuffle(torch.real(x_0)) + 1j * self.PixelShuffle(torch.imag(x_0))

class BatchNorm2DLayer(nn.Module):
    def __init__(self, featureNum, eps=1e-5):
        super(BatchNorm2DLayer, self).__init__()

        self.BatchNorm_real = nn.BatchNorm2d(num_features=featureNum, eps=eps)
        self.BatchNorm_imag = nn.BatchNorm2d(num_features=featureNum, eps=eps)

    def forward(self, x_0):
        return self.BatchNorm_real(torch.real(x_0)) + 1j * self.BatchNorm_imag(torch.imag(x_0))



class ConvBlock(nn.Module):
    def __init__(self, channelIn, channelOut, imageSize=[32, 32], kernel=[3, 3], pool=[2, 2],
        pixel=-1, act='relu', param={}, calFolder=None):
        super(ConvBlock, self).__init__()

        self.calFolder = calFolder
        upsample = pixel if pixel > 1 else 1

        self.Conv_digit = nn.Conv2d(
            in_channels=channelIn, out_channels=channelOut*(upsample**2),
            kernel_size=[kernel[0], kernel[1]], padding=[(kernel[0]-1)//2, (kernel[1]-1)//2], bias=False,
            dtype=torch.complex64)
        self.Conv_anal = None
        self.analArgs = dict(chanInNum=channelIn, chanOutNum=channelOut*(upsample**2),
            imageSize=[imageSize[0], imageSize[1]], kernelSize=[kernel[0], kernel[1]], param=param)
        self.Conv_pixel = PixelShuffleLayer(pixel=pixel) if pixel > 1 else nn.Identity()
        self.Conv_act = nn.Sequential(
            BatchNorm2DLayer(featureNum=channelOut),
            ACTLayer(act=act),
            Pool2DLayer(kernelX=pool[0], kernelY=pool[1]))
        self.zoom = None

        self.apply(WeightInit)

        self.macNum = 4 * channelOut * channelIn * kernel[0] * kernel[1] * imageSize[0] * imageSize[1] * (upsample**2)
        self.paramNum = channelOut * channelIn * kernel[0] * kernel[1] * (upsample**2)

    def forward(self, x_0):
        x_1 = self.Conv_digit(x_0.type(torch.cfloat))
        x_2 = self.Conv_pixel(x_1)
        x_3 = self.Conv_act(x_2)
        return x_3

    def AnalogInit(self, batchNum=1):
        self.Conv_anal = ConvAcc(**self.analArgs)
        if self.calFolder is not None:
            self.Conv_anal.Calibrate(folder=self.calFolder)
        kernel = self.Conv_digit.weight
        self.Conv_anal.Init(torch.swapaxes(kernel, 0, 1), batchNum=batchNum)

    def AnalogZoom(self, x_0, engine=0, snr=None):
        if engine is None:
            zoom = 1.0
        else:
            x_0 = x_0.type(torch.cfloat)
            x_1_anal = self.Conv_anal.InferAnal(x_0, engine=engine, snr=snr)
            x_1_digit = self.Conv_digit(x_0)
            zoom = torch.std(torch.flatten(x_1_digit)) / torch.std(torch.flatten(x_1_anal))
        self.zoom = zoom
        return zoom

    def AnalogInfer(self, x_0, engine=0, snr=None):
        x_0 = x_0.type(torch.cfloat)

        if engine is not None:
            x_1 = self.Conv_anal.InferAnal(x_0, engine=engine, snr=snr)
            if self.zoom is not None:
                x_1 *= self.zoom
        else:
            x_1 = self.Conv_digit(x_0)
        x_2 = self.Conv_pixel(x_1)
        x_3 = self.Conv_act(x_2)

        return x_3, (x_1, x_3)

    def GetInfo(self):
        return self.macNum, self.paramNum



class FCBlock(nn.Module):
    def __init__(self, hiddenIn, hiddenOut, act='relu', param={}, calFolder=None):
        super(FCBlock, self).__init__()

        self.calFolder = calFolder

        self.FC_digit = nn.Linear(in_features=hiddenIn, out_features=hiddenOut, bias=False,
            dtype=torch.complex64)
        self.FC_anal = None
        self.analArgs = dict(inputSize=hiddenIn, outputSize=hiddenOut, param=param)
        self.FC_act = ACTLayer(act=act)
        self.zoom = None

        self.apply(WeightInit)

        self.macNum = 4 * hiddenIn * hiddenOut
        self.paramNum = hiddenIn * hiddenOut

    def forward(self, x_0):
        x_1 = self.FC_digit(x_0.type(torch.cfloat))
        x_2 = self.FC_act(x_1)
        return x_2

    def AnalogInit(self, batchNum=1):
        self.FC_anal = FCAcc(**self.analArgs)
        if self.calFolder is not None:
            self.FC_anal.Calibrate(folder=self.calFolder)
        weight = self.FC_digit.weight
        self.FC_anal.Init(torch.swapaxes(weight, 0, 1), batchNum=batchNum)

    def AnalogZoom(self, x_0, engine=0, snr=None):
        if engine is None:
            zoom = 1.0
        else:
            x_0 = x_0.type(torch.cfloat)
            x_1_anal = self.FC_anal.InferAnal(x_0, engine=engine, snr=snr)
            x_1_digit = self.FC_digit(x_0)
            zoom = torch.std(torch.flatten(x_1_digit)) / torch.std(torch.flatten(x_1_anal))
        self.zoom = zoom
        return zoom

    def AnalogInfer(self, x_0, engine=0, snr=None):
        x_0 = x_0.type(torch.cfloat)

        if engine is not None:
            x_1 = self.FC_anal.InferAnal(x_0, engine=engine, snr=snr)
            if self.zoom is not None:
                x_1 *= self.zoom
        else:
            x_1 = self.FC_digit(x_0)
        x_2 = self.FC_act(x_1)

        return x_2, (x_1, x_2)

    def GetInfo(self):
        return self.macNum, self.paramNum



class ConvNet(nn.Module):
    def __init__(self, dataShape, classNum, channelList, kernelList, poolList,
        is1D=False, param={}, calFolder=None):
        super(ConvNet, self).__init__()

        assert np.shape(dataShape)[0] == 3
        channelNum, imageSizeX, imageSizeY = dataShape
        if is1D:
            assert imageSizeX == 1

        self.macNum = 0
        self.paramNum = 0

        assert len(channelList) == len(kernelList)
        assert len(channelList) == len(poolList)
        ConvPreList = []
        for channel, kernel, pool in zip(channelList, kernelList, poolList):
            kernelXY = [1, kernel] if is1D else [kernel, kernel]
            poolXY = [1, pool] if is1D else [pool, pool]
            if (imageSizeX<poolXY[0])or(imageSizeY<poolXY[1]):
                poolXY = [1, 1]

            Conv = ConvBlock(channelIn=channelNum, channelOut=channel,
                imageSize=[imageSizeX, imageSizeY], kernel=kernelXY, pool=poolXY,
                act='relu', param=param, calFolder=calFolder)
            imageSizeX //= poolXY[0]
            imageSizeY //= poolXY[1]
            channelNum = channel
            ConvPreList.append(Conv)
            macNumTemp, paramNumTemp = Conv.GetInfo()
            self.macNum += macNumTemp
            self.paramNum += paramNumTemp
        self.ConvPreList = nn.ModuleList(ConvPreList)

        self.FCLast = FCBlock(hiddenIn=channelNum*imageSizeX*imageSizeY, hiddenOut=classNum,
            act='output', param=param, calFolder=calFolder)
        macNumTemp, paramNumTemp = self.FCLast.GetInfo()
        self.macNum += macNumTemp
        self.paramNum += paramNumTemp

        self.apply(WeightInit)

    def forward(self, x_0):
        x_1 = x_0.type(torch.cfloat)
        for Conv in self.ConvPreList:
            x_1 = Conv(x_1)
        x_2 = torch.flatten(x_1, start_dim=1, end_dim=-1)
        return self.FCLast(x_2)

    def AnalogInit(self, batchNum=1):
        for Conv in self.ConvPreList:
            Conv.AnalogInit(batchNum=batchNum)
        self.FCLast.AnalogInit(batchNum=batchNum)

    def AnalogZoom(self, x_0, engineList=[0], snr=None):
        layerNum = len(self.ConvPreList)
        zoomList = []

        x_1 = x_0.type(torch.cfloat)
        for ConvIdx, Conv in enumerate(self.ConvPreList):
            print('Measure Zoom for Conv layer [%d/%d]'%(ConvIdx, layerNum))
            zoomList.append(Conv.AnalogZoom(x_1, engine=GetEngine(engineList, ConvIdx), snr=snr))
            x_1 = Conv(x_1)

        print('Measure Zoom for FC layer')
        x_2 = torch.flatten(x_1, start_dim=1, end_dim=-1)
        zoomList.append(self.FCLast.AnalogZoom(x_2, engine=GetEngine(engineList, layerNum), snr=snr))

        return zoomList

    def AnalogInfer(self, x_0, engineList=[0], snr=None):
        layerNum = len(self.ConvPreList)
        dataInput = x_0.type(torch.cfloat)

        dataConvList = []
        x_1 = dataInput
        for ConvIdx, Conv in enumerate(self.ConvPreList):
            print('Inference for Conv layer [%d/%d]'%(ConvIdx, layerNum))
            x_1, data = Conv.AnalogInfer(x_1, engine=GetEngine(engineList, ConvIdx), snr=snr)
            dataConvList.append(data)

        print('Inference for FC layer')
        x_2 = torch.flatten(x_1, start_dim=1, end_dim=-1)
        x_3, data = self.FCLast.AnalogInfer(x_2, engine=GetEngine(engineList, layerNum), snr=snr)

        return x_3, (dataInput, dataConvList, [data])

    def GetInfo(self):
        return self.macNum, self.paramNum



class Generator(nn.Module):
    def __init__(self, dataShape, classNum, latent=62, code=2,
        channelList=[128, 128, 64], kernelList=[3, 3, 3], upsampleList=[2, 2, -1],
        actList=['leaky', 'leaky', 'tanh'], param={}, calFolder=None):
        super(Generator, self).__init__()

        self.classNum = classNum

        assert np.shape(dataShape)[0] == 3
        channelNum, imageSizeX, imageSizeY = dataShape

        self.macNum = 0
        self.paramNum = 0

        assert len(channelList) == len(kernelList)
        assert len(channelList) == len(upsampleList)
        assert len(channelList) == len(actList)
        ConvList = []
        for ConvIdx in reversed(range(len(channelList))):
            channel = channelList[ConvIdx]
            kernel = kernelList[ConvIdx]
            act = actList[ConvIdx]
            upsample = upsampleList[ConvIdx]

            if upsample > 0:
                imageSizeX //= upsample
                imageSizeY //= upsample

            Conv = ConvBlock(channelIn=channel, channelOut=channelNum,
                imageSize=[imageSizeX, imageSizeY], kernel=[kernel, kernel], pool=[1, 1],
                pixel=upsample, act=act, param=param, calFolder=calFolder)
            channelNum = channel
            ConvList.append(Conv)

            macNumTemp, paramNumTemp = Conv.GetInfo()
            self.macNum += macNumTemp
            self.paramNum += paramNumTemp
        self.ConvList = nn.ModuleList(ConvList)

        self.channelNumMid = channelNum
        self.imageSizeMidX = imageSizeX
        self.imageSizeMidY = imageSizeY

        featIn = latent+classNum+code
        featOut = channelNum*imageSizeX*imageSizeY
        self.FC = FCBlock(hiddenIn=featIn, hiddenOut=featOut, act='none', param=param, calFolder=calFolder)
        macNumTemp, paramNumTemp = self.FC.GetInfo()
        self.macNum += macNumTemp
        self.paramNum += paramNumTemp

        def weight_init_wrapper(layer):
            WeightInit(layer, method='normal')
        self.apply(weight_init_wrapper)

    def __Input__(self, seed, target, code):
        return torch.cat((seed, F.one_hot(target, num_classes=self.classNum), code), -1)

    def __Reshape__(self, x):
        return torch.reshape(x, (-1, self.channelNumMid, self.imageSizeMidX, self.imageSizeMidY))

    def forward(self, seed, target, code):
        x_1 = self.__Reshape__(self.FC(self.__Input__(seed, target, code)))
        for Conv in reversed(self.ConvList):
            x_1 = Conv(x_1)
        return torch.real(x_1)

    def AnalogInit(self, batchNum=1):
        for Conv in self.ConvList:
            Conv.AnalogInit(batchNum=batchNum)
        self.FC.AnalogInit(batchNum=batchNum)

    def AnalogZoom(self, seed, target, code, engineList=[0], snr=None):
        x_0 = self.__Input__(seed, target, code)
        zoomList = [self.FC.AnalogZoom(x_0, engine=GetEngine(engineList, 0), snr=snr)]

        x_1 = self.__Reshape__(self.FC(x_0))
        for ConvIdx, Conv in enumerate(reversed(self.ConvList)):
            zoomList.append(Conv.AnalogZoom(x_1, engine=GetEngine(engineList, ConvIdx+1), snr=snr))
            x_1 = Conv(x_1)

        return zoomList

    def AnalogInfer(self, seed, target, code, engineList=[0], snr=None):
        dataInput = self.__Input__(seed, target, code).type(torch.cfloat)

        x_1, data = self.FC.AnalogInfer(dataInput, engine=GetEngine(engineList, 0), snr=snr)
        dataFCList = [data]

        dataConvList = []
        x_2 = self.__Reshape__(x_1)
        for ConvIdx, Conv in enumerate(reversed(self.ConvList)):
            x_2, data = Conv.AnalogInfer(x_2, engine=GetEngine(engineList, ConvIdx+1), snr=snr)
            dataConvList.append(data)

        return torch.real(x_2), (dataInput, dataConvList, dataFCList)

    def GetInfo(self):
        return self.macNum, self.paramNum



class Discriminator(nn.Module):
    def __init__(self, dataShape, classNum, code=2,
        channelList=[16, 32, 64, 128], kernelList=[3, 3, 3, 3], strideList=[2, 2, 2, 2], batchList=[False, True, True, True]):
        super(Discriminator, self).__init__()

        assert np.shape(dataShape)[0] == 3
        channelNum, imageSizeX, imageSizeY = dataShape

        assert len(channelList) == len(kernelList)
        assert len(channelList) == len(strideList)
        assert len(channelList) == len(batchList)
        ConvList = []
        for channel, kernel, stride, batch in zip(channelList, kernelList, strideList, batchList):
            Conv = nn.Sequential(
                nn.Conv2d(in_channels=channelNum, out_channels=channel,
                    kernel_size=[kernel, kernel], stride=stride, padding=[(kernel-1)//2, (kernel-1)//2]),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(p=0.25),
                nn.BatchNorm2d(num_features=channel) if batch else nn.Identity())

            channelNum = channel
            imageSizeX = (imageSizeX+stride-1) // stride
            imageSizeY = (imageSizeY+stride-1) // stride
            ConvList.append(Conv)
        self.ConvList = nn.ModuleList(ConvList)

        featNum = channelNum * imageSizeX * imageSizeY

        self.FCAdv = nn.Linear(featNum, 1)
        self.FCClass = nn.Sequential(
            nn.Linear(featNum, classNum),
            nn.Softmax(dim=-1))
        self.FCCode = nn.Linear(featNum, code)

        def weight_init_wrapper(layer):
            WeightInit(layer, method='normal')
        self.apply(weight_init_wrapper)

    def forward(self, x_0):
        x_1 = x_0
        for Conv in self.ConvList:
            x_1 = Conv(x_1)
        x_1 = torch.flatten(x_1, start_dim=1, end_dim=-1)

        return self.FCAdv(x_1), self.FCClass(x_1), self.FCCode(x_1)
