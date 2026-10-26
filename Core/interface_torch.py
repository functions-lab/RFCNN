import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
from scipy import interpolate
from scipy.io import loadmat, savemat
import torch
import torch.optim as optim
from tqdm import tqdm

from Core.Tx2Rx_mixer.Tx2Rx_mult import Transmission
from Core.Tx2Rx_mixer.Tx2Rx_sim_torch import Simulation
from Core.GetParam import GetParam



def __Zadoff__(zadoffLen, u=43, q=0, device='cpu'):
    c_f = zadoffLen % 2
    zadoff = torch.zeros((zadoffLen), dtype=torch.complex64, device=device)
    zadoffIndex = torch.arange(zadoffLen, device=device)
    zadoff = torch.exp(-1j*math.pi*u*zadoffIndex*(zadoffIndex+c_f+2*q)/zadoffLen)
    return zadoff

def __Random__(headNum, headLen, device='cpu'):
    phase = 2*math.pi*torch.rand((headNum, headLen), device=device)
    head = torch.exp(1j*phase)
    return head

def __Convolution__(input_1, input_2, outputWin):
    dataNum, inputLen_1 = input_1.size()
    _, inputLen_2 = input_2.size()
    outputLen = outputWin[1] - outputWin[0]

    output = torch.zeros((dataNum, outputLen)).to(input_1)
    for outputIdx in range(outputWin[0], outputWin[1]):
        startIdx = max([0, outputIdx-inputLen_2+1])
        endIdx = min([inputLen_1-1, outputIdx])
        output[:, outputIdx-outputWin[0]] = torch.sum(input_1[:, startIdx: endIdx+1] * input_2[:, inputLen_2-outputIdx+startIdx-1: inputLen_2-outputIdx+endIdx], dim=1)

    return output

def DeNoise(waveOrig, methodList=['median', 'poly']):
    waveLen = np.shape(waveOrig)[0]
    window = int(np.ceil(waveLen*0.01))

    waveOut = waveOrig
    for method in methodList:
        waveIn = waveOut
        if method == 'median':
            waveOut_amp = scipy.signal.medfilt(np.abs(waveIn), kernel_size=2*window+1)
            waveOut_amp[: window] = np.median(np.abs(waveIn[:(2*window+1)]))
            waveOut_amp[-window:] = np.median(np.abs(waveIn[-(2*window+1):]))
            waveOut_phase = scipy.signal.medfilt(np.angle(waveIn), kernel_size=2*window+1)
            waveOut_phase[: window] = np.median(np.unwrap(np.angle(waveIn[:(2*window+1)])))
            waveOut_phase[-window:] = np.median(np.unwrap(np.angle(waveIn[-(2*window+1):])))
            waveOut = waveOut_amp * np.exp(1j*waveOut_phase)
        elif method == 'poly':
            poly_amp = np.polyfit(np.arange(window, waveLen-window), np.abs(waveIn[window: -window]), deg=3)
            poly_phase = np.polyfit(np.arange(window, waveLen-window), np.unwrap(np.angle(waveIn[window: -window])), deg=2)
            waveOut = np.polyval(poly_amp, np.arange(waveLen)) * np.exp(1j*np.polyval(poly_phase, np.arange(waveLen)))
        else:
            print("Warning: Denoising Method NOT Found!")

    return waveOut

def Channel2File(channel_1, channel_2, channelOut, fileName, inputLen_1, inputLen_2, outputWin):
    assert np.shape(channel_1)[0] == inputLen_1
    assert np.shape(channel_2)[0] == inputLen_2
    assert np.shape(channelOut)[0] == outputWin[1] - outputWin[0]
    freqAxis_1 = np.arange(0, inputLen_1) / inputLen_1
    freqAxis_2 = np.arange(0, inputLen_2) / inputLen_1
    freqAxisOut = np.arange(outputWin[0], outputWin[1]) / inputLen_1
    savemat(fileName, {
        'freqAxis_1': freqAxis_1,
        'channel_1': channel_1,
        'freqAxis_2': freqAxis_2,
        'channel_2': channel_2,
        'freqAxisOut': freqAxisOut,
        'channelOut': channelOut})

def File2Channel(fileName, inputLen_1, inputLen_2, outputWin):
    freqAxis_1 = np.arange(0, inputLen_1) / inputLen_1
    freqAxis_2 = np.arange(0, inputLen_2) / inputLen_1
    freqAxisOut = np.arange(outputWin[0], outputWin[1]) / inputLen_1

    freqAxisOrig_1 = loadmat(fileName)['freqAxis_1'].squeeze(0)
    channelOrig_1 = loadmat(fileName)['channel_1'].squeeze(0)
    interpolater_1 = interpolate.interp1d(freqAxisOrig_1, channelOrig_1, kind='nearest', fill_value="extrapolate")
    channel_1 = interpolater_1(freqAxis_1)

    freqAxisOrig_2 = loadmat(fileName)['freqAxis_2'].squeeze(0)
    channelOrig_2 = loadmat(fileName)['channel_2'].squeeze(0)
    interpolater_2 = interpolate.interp1d(freqAxisOrig_2, channelOrig_2, kind='nearest', fill_value="extrapolate")
    channel_2 = interpolater_2(freqAxis_2)

    freqAxisOrigOut = loadmat(fileName)['freqAxisOut'].squeeze(0)
    channelOrigOut = loadmat(fileName)['channelOut'].squeeze(0)
    interpolaterOut = interpolate.interp1d(freqAxisOrigOut, channelOrigOut, kind='nearest', fill_value="extrapolate")
    channelOut = interpolaterOut(freqAxisOut)

    return channel_1, channel_2, channelOut

def PlotScatter(fileName, x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=0, marker='+')
    fig.savefig(fileName)
    plt.close(fig)

def PlotChannel(fileName, x, y_1, y_2):
    fig, ax = plt.subplots()
    ax.plot(x, y_1, linewidth=0, marker='+')
    ax.plot(x, y_2, linewidth=1, marker='')
    fig.savefig(fileName)
    plt.close(fig)

def PlotCurve(fileName, x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=1, marker='')
    fig.savefig(fileName)
    plt.close(fig)



class ConvCore():
    def __init__(self, inputLen_1, inputLen_2, outputWin, headNum=10, param=GetParam()):
        super(ConvCore, self).__init__()

        self.inputLen_1 = inputLen_1
        self.inputLen_2 = inputLen_2
        self.outputWin = outputWin
        assert headNum >= 2
        self.headNum = headNum

        inputPad = param['inputPad']
        outputPad = param['outputPad']
        cpRate = param['cpRate']
        cpInt = param['cpInt']
        waveAmp = param['waveAmp']
        sync = param['sync']
        waveLenMax = param['waveLenMax']
        device = param['device']
        transmission = Transmission(param)
        simulation = Simulation(param)
        self.waveAmp = waveAmp
        self.sync = sync
        self.waveLenMax = waveLenMax
        self.device = device
        self.transmission = transmission
        self.simulation = simulation

        inputLenMax = max([inputLen_1, inputLen_2])
        inputLenMin = min([inputLen_1, inputLen_2])
        inputPadLen = math.ceil(2*inputPad*inputLenMax)
        inputFFTSize = inputLenMax + inputPadLen
        inputCpLen = math.ceil(inputFFTSize*(cpRate+cpInt))
        outputLen = outputWin[1] - outputWin[0]
        outputPadLen = math.ceil(2*outputPad*outputLen)
        outputFFTSize = outputLen + outputPadLen
        outputCpInt = cpInt
        outputCpLen = math.ceil(outputFFTSize*cpRate)
        self.inputFFTSize = inputFFTSize
        self.inputCpLen = inputCpLen
        self.outputFFTSize = outputFFTSize
        self.outputPadLen = outputPadLen
        self.outputCpInt = outputCpInt
        self.outputCpLen = outputCpLen

        shrink = [inputFFTSize, outputFFTSize]
        shrinkGCD = math.gcd(shrink[0], shrink[1])
        shrink[0] //= shrinkGCD
        shrink[1] //= shrinkGCD
        freqDelta = 0
        if inputLenMax % 2 == 1:
            freqDelta -= 0.5
        if inputLenMin % 2 == 1:
            freqDelta += 0.5
        if outputLen % 2 == 1:
            freqDelta += 0.5
        freqOffset = ((outputWin[0]+outputWin[1]-inputLen_1-inputLen_2)/2 + freqDelta)/inputFFTSize
        self.shrink = shrink
        self.freqOffset = freqOffset

        self.inputHead_1 = __Random__(headNum, inputLen_1, device=device)
        self.waveHead_1, _ = self.__WaveGenerate__(self.inputHead_1, waveAmp=waveAmp[0], isScale=False)
        self.inputHead_2 = __Random__(headNum, inputLen_2, device=device)
        self.waveHead_2, _ = self.__WaveGenerate__(torch.conj(self.inputHead_2), waveAmp=waveAmp[1], isScale=False)
        self.outputHead = __Convolution__(self.inputHead_1, self.inputHead_2, outputWin)
        self.channel_1 = None
        self.channel_2 = None
        self.channelOut = None

    def __AddPadding__(self, input_, padLen):
        device = self.device
        testNum, inputLen = input_.size()

        inputPadded = torch.zeros((testNum, inputLen+padLen), dtype=torch.complex64, device=device)
        offset = padLen // 2
        inputPadded[:, offset: offset+inputLen] = input_ * math.sqrt((inputLen+padLen)/inputLen)
        return inputPadded

    def __RemovePadding__(self, outputPadded, padLen):
        offset = padLen // 2
        output = outputPadded[:, offset: offset-padLen]
        return output

    def __Modulate__(self, input_, waveAmp=-1):
        fftSize = self.inputFFTSize
        cpLen = self.inputCpLen

        freqOffset = math.ceil(fftSize/2)
        spec = torch.roll(input_, freqOffset, dims=1)

        waveOrig = torch.fft.ifft(spec, axis=1) * math.sqrt(fftSize)
        if waveAmp > 0:
            waveNorm = waveAmp * waveOrig
        else:
            waveNorm = waveOrig

        if cpLen > 0:
            waveMult = torch.tile(waveNorm, (1, math.ceil(cpLen/fftSize/2)))
            waveCP = torch.concatenate((waveMult[:, -math.floor(cpLen/2):], waveNorm, waveMult[:, 0: math.ceil(cpLen/2)]), axis=1)
        else:
            waveCP = waveNorm

        return waveCP

    def __WaveGenerate__(self, input_, waveAmp=-1, isScale=True):
        fftSize = self.inputFFTSize
        inputLen = input_.size()[1]
        padLen = fftSize - inputLen

        if isScale:
            scale = torch.sqrt(torch.mean(torch.abs(input_)**2, axis=1))
            inputNormed = input_ / scale.unsqueeze(1).repeat(1, inputLen)
            inputPadded = self.__AddPadding__(inputNormed, padLen)
        else:
            scale = None
            inputPadded = self.__AddPadding__(input_, padLen)
        wave = self.__Modulate__(inputPadded, waveAmp=waveAmp)

        return wave, scale

    def __DeModulate__(self, waveCP):
        padLen = self.outputPadLen
        fftSize = self.outputFFTSize
        cpInt = self.outputCpInt
        cpLen = self.outputCpLen

        if cpLen > 0:
            wave = waveCP[:, math.floor(cpLen/2): -math.ceil(cpLen/2)]
        else:
            wave = waveCP
        fftSize = wave.size()[1]

        spec = torch.fft.fft(wave, axis=1)[:, ::(cpInt+1)] / math.sqrt(fftSize)

        freqOffset = math.ceil(fftSize/(cpInt+1)/2) - 1
        outputPadded = torch.roll(spec, freqOffset, dims=1)
        output = self.__RemovePadding__(outputPadded, padLen)

        return output

    def __ChannelTime__(self, outputHeadEst):
        outputHeadGt = self.outputHead
        device = self.device
        symNum, subNum = outputHeadGt.size()
        sync = self.sync

        symAxis = torch.arange(symNum).to(device)
        subAxis = torch.arange(subNum).to(device)

        channelMap = outputHeadEst * torch.conj(outputHeadGt)
        amp = torch.sqrt(torch.mean(torch.abs(outputHeadEst)**2)) / torch.sqrt(torch.mean(torch.abs(outputHeadGt)**2))

        channelTimeList = torch.mean(channelMap, axis=1)
        if sync == 1:
            phaseTimeSlope = 0
        else:
            phaseTimeSlope = torch.mean(torch.angle(channelTimeList[1:] * torch.conj(channelTimeList[:-1])))
        phaseTimeAxis = torch.exp(1j*phaseTimeSlope*symAxis).to(outputHeadEst)

        channelFreqList = torch.mean(channelMap, axis=0)
        phaseFreqSlope = torch.mean(torch.angle(channelFreqList[1:] * torch.conj(channelFreqList[:-1])))
        phaseFreqAxis = torch.exp(1j*phaseFreqSlope*subAxis).to(outputHeadEst)

        phaseMap = phaseTimeAxis.unsqueeze(1).repeat(1, subNum) * phaseFreqAxis.unsqueeze(0).repeat(symNum, 1)
        phaseBias = torch.angle(torch.mean(channelMap/phaseMap))

        return phaseTimeSlope, phaseFreqSlope, phaseBias, amp

    def __ChannelCore__(self, outputHeadEst, phaseTimeSlope, phaseFreqSlope, phaseBias, epoch=10000, folder=None):
        inputHead_1 = self.inputHead_1
        inputHead_2 = self.inputHead_2
        outputHeadGt = self.outputHead
        outputWin = self.outputWin
        device = self.device
        inputLen_1 = inputHead_1.size()[1]
        inputLen_2 = inputHead_2.size()[1]
        symNum, outputLen = outputHeadGt.size()

        phaseTimeAxis = phaseTimeSlope * torch.arange(symNum, device=device)
        phaseFreqAxis = phaseFreqSlope * torch.arange(outputLen, device=device)
        phaseMap = torch.exp(1j*phaseBias) * torch.outer(torch.exp(1j*phaseTimeAxis), torch.exp(1j*phaseFreqAxis))
        outputHeadEstNew = outputHeadEst / phaseMap

        if self.channel_1 is None:
            channel_1 = torch.ones((inputLen_1), dtype=torch.cfloat, requires_grad=True, device=device)
        else:
            channel_1 = torch.tensor(self.channel_1.detach().clone(), dtype=torch.cfloat, requires_grad=True, device=device)
        if self.channel_2 is None:
            channel_2 = torch.ones((inputLen_2), requires_grad=True, device=device)
        else:
            channel_2 = torch.tensor(self.channel_2.detach().clone(), requires_grad=True, device=device)
        if self.channelOut is None:
            channelOutInit = torch.mean(torch.abs(outputHeadEstNew)/torch.abs(outputHeadGt), axis=0)
            channelOut = torch.tensor(channelOutInit.detach().clone(), dtype=torch.cfloat, requires_grad=True, device=device)
        else:
            channelOut = torch.tensor(self.channelOut.detach().clone(), dtype=torch.cfloat, requires_grad=True, device=device)
        optimizer = optim.Adam([channel_1, channel_2, channelOut], lr=1e-3)

        trainBar = tqdm(range(epoch))
        rmseList = np.ones((epoch)) * np.nan
        snrList = np.ones((epoch)) * np.nan
        enobList = np.ones((epoch)) * np.nan
        for epochIdx in trainBar:
            optimizer.zero_grad()
            inputHeadCal_1 = inputHead_1 * channel_1
            inputHeadCal_2 = inputHead_2 * channel_2
            outputHead = __Convolution__(inputHeadCal_1, inputHeadCal_2, outputWin)
            outputHeadCal = outputHead * channelOut
            outputHeadError = outputHeadCal - outputHeadEstNew
            loss = torch.mean(torch.real(outputHeadError * torch.conj(outputHeadError)))
            loss.backward()
            optimizer.step()

            rmse = math.sqrt(loss)
            noise = math.sqrt(torch.mean(torch.abs(outputHeadError)**2))
            signal = math.sqrt(torch.mean(torch.abs(outputHeadEstNew)**2))
            snr = 20*math.log10(signal/noise)
            enob = math.log2(signal/noise)
            trainBar.set_description(desc='[%d/%d] RMSE: %.5f SNR: %.3f ENOB: %.3f'%(epochIdx, epoch, rmse, snr, enob))
            rmseList[epochIdx] = rmse
            snrList[epochIdx] = snr
            enobList[epochIdx] = enob
            if epochIdx>0 and (snrList[epochIdx]-snrList[epochIdx-1])<1e-4:
                break

        channelFilter_1 = DeNoise(channel_1.detach().cpu().numpy(), methodList=['median', 'poly'])
        channelFilter_1 = torch.tensor(channelFilter_1.copy(), dtype=torch.cfloat, device=device)
        channelFilter_2 = DeNoise(channel_2.detach().cpu().numpy(), methodList=['median', 'poly'])
        channelFilter_2 = torch.tensor(np.real(channelFilter_2).copy(), dtype=torch.cfloat, device=device)
        channelOutFilter = DeNoise(channelOut.detach().cpu().numpy(), methodList=['median', 'poly'])
        channelOutFilter = torch.tensor(channelOutFilter.copy(), dtype=torch.cfloat, device=device)

        if folder is not None:
            PlotScatter(folder+'/scatter_abs.png',
                torch.abs(outputHeadCal).detach().cpu().numpy(), torch.abs(outputHeadEstNew).detach().cpu().numpy())
            PlotScatter(folder+'/scatter_angle.png',
                torch.angle(outputHeadCal).detach().cpu().numpy(), torch.angle(outputHeadEstNew).detach().cpu().numpy())
            PlotScatter(folder+'/scatter_real.png',
                torch.real(outputHeadCal).detach().cpu().numpy(), torch.real(outputHeadEstNew).detach().cpu().numpy())
            PlotScatter(folder+'/scatter_imag.png',
                torch.imag(outputHeadCal).detach().cpu().numpy(), torch.imag(outputHeadEstNew).detach().cpu().numpy())

            PlotChannel(folder+'/channel_1_amp.png',
                np.arange(inputLen_1), torch.abs(channel_1).detach().cpu().numpy(), torch.abs(channelFilter_1).detach().cpu().numpy())
            PlotChannel(folder+'/channel_1_phase.png',
                np.arange(inputLen_1), torch.angle(channel_1).detach().cpu().numpy(), torch.angle(channelFilter_1).detach().cpu().numpy())
            PlotChannel(folder+'/channel_2_amp.png',
                np.arange(inputLen_2), torch.abs(channel_2).detach().cpu().numpy(), torch.abs(channelFilter_2).detach().cpu().numpy())
            PlotChannel(folder+'/channel_2_phase.png',
                np.arange(inputLen_2), torch.angle(channel_2).detach().cpu().numpy(), torch.angle(channelFilter_2).detach().cpu().numpy())
            PlotChannel(folder+'/channelOut_amp.png',
                np.arange(outputLen), torch.abs(channelOut).detach().cpu().numpy(), torch.abs(channelOutFilter).detach().cpu().numpy())
            PlotChannel(folder+'/channelOut_phase.png',
                np.arange(outputLen), torch.angle(channelOut).detach().cpu().numpy(), torch.angle(channelOutFilter).detach().cpu().numpy())

            PlotCurve(folder+'/curve_rmse.png', np.arange(epochIdx), rmseList[:epochIdx])
            PlotCurve(folder+'/curve_snr.png', np.arange(epochIdx), snrList[:epochIdx])
            PlotCurve(folder+'/curve_enob.png', np.arange(epochIdx), enobList[:epochIdx])

        return channelFilter_1.detach(), channelFilter_2.detach(), channelOutFilter.detach()

    def __Channel__(self, outputHeadEst_before, outputHeadEst_after, channelOut, testNum):
        headNum = self.headNum
        device = self.device
        outputLen = outputHeadEst_before.size()[1]

        phaseTimeSlope_before, phaseFreqSlope_before, phaseBias_before, amp_before = self.__ChannelTime__(outputHeadEst_before)
        phaseTimeSlope_after, phaseFreqSlope_after, phaseBias_after, amp_after = self.__ChannelTime__(outputHeadEst_after)
        phaseTimeSlope = (phaseTimeSlope_before + phaseTimeSlope_after) / 2
        phaseFreqSlope = (phaseFreqSlope_before + phaseFreqSlope_after) / 2
        phaseBias = torch.angle(torch.exp(1j*phaseBias_before) + torch.exp(1j*(phaseBias_after-(headNum+testNum)*phaseTimeSlope)))
        amp = (amp_before + amp_after) / 2

        phaseTimeAxis = phaseTimeSlope * torch.arange(headNum, headNum+testNum, device=device)
        phaseFreqAxis = phaseFreqSlope * torch.arange(outputLen, device=device)
        channelMap = torch.exp(1j*phaseBias) * torch.outer(torch.exp(1j*phaseTimeAxis), channelOut*torch.exp(1j*phaseFreqAxis)) * amp

        return channelMap

    def __WaveAnalyze__(self, wave, scale, channelOut):
        headNum = self.headNum

        outputHeaded = self.__DeModulate__(wave)

        outputOrig = outputHeaded[headNum: -headNum]
        testNum, outputLen = outputOrig.size()
        outputHeadEst_before = outputHeaded[:headNum, :]
        outputHeadEst_after = outputHeaded[-headNum:, :]
        channelMap = self.__Channel__(outputHeadEst_before, outputHeadEst_after, channelOut, testNum)

        output = scale.unsqueeze(1).repeat(1, outputLen) * outputOrig / channelMap
        return output

    def Calibrate(self, epoch=10000, folder=None):
        headNum = self.headNum
        transmission = self.transmission
        shrink = self.shrink
        freqOffset = self.freqOffset
        waveInput_1 = self.waveHead_1
        waveInput_2 = self.waveHead_2
        device = self.device

        if folder is not None and os.path.exists(folder+'/channel.mat'):
            print('Loading Existing Channel Calibration!')
            channelTemp_1, channelTemp_2, channelTempOut = File2Channel(folder+'/channel.mat', self.inputLen_1, self.inputLen_2, self.outputWin)
            channel_1 = torch.tensor(channelTemp_1, dtype=torch.cfloat, device=device)
            channel_2 = torch.tensor(channelTemp_2, dtype=torch.cfloat, device=device)
            channelOut = torch.tensor(channelTempOut, dtype=torch.cfloat, device=device)
        else:
            print('Generating New Channel Calibration!')
            if folder is not None:
                os.makedirs(folder, exist_ok=True)

            waveInputAll = torch.stack((torch.flatten(waveInput_1), torch.flatten(waveInput_2)), axis=0).detach().cpu().numpy()
            waveOutputAll, _, _ = transmission.Tx2Rx(waveInputAll, rxNum=1, padAmpList=[0, 0], shrink=shrink, freqOffset=freqOffset)
            waveOutput = torch.reshape(torch.tensor(waveOutputAll).to(waveInput_1), (headNum, -1))

            output = self.__DeModulate__(waveOutput)
            phaseTimeSlope, phaseFreqSlope, phaseBias, amp = self.__ChannelTime__(output)
            channel_1, channel_2, channelOut = self.__ChannelCore__(output/amp, phaseTimeSlope, phaseFreqSlope, phaseBias, epoch=epoch, folder=folder)

            if folder is not None:
                Channel2File(
                    channel_1.detach().cpu().numpy(), channel_2.detach().cpu().numpy(), channelOut.detach().cpu().numpy(), 
                    folder+'/channel.mat', self.inputLen_1, self.inputLen_2, self.outputWin)

        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.channelOut = channelOut

    def LoadWave_1(self, input_):
        inputLen = self.inputLen_1
        channel = self.channel_1
        assert input_.size()[1] == inputLen

        waveInputSim, scaleSim = self.__WaveGenerate__(input_, waveAmp=self.waveAmp[0], isScale=True)

        inputCal = input_ / channel if channel is not None else input_
        waveInputExp, scaleExp = self.__WaveGenerate__(inputCal, waveAmp=self.waveAmp[0], isScale=True)

        self.waveInputSim_1 = waveInputSim
        self.scaleSim_1 = scaleSim
        self.waveInputExp_1 = waveInputExp
        self.scaleExp_1 = scaleExp

    def LoadWave_2(self, input_):
        inputLen = self.inputLen_2
        channel = self.channel_2
        assert input_.size()[1] == inputLen

        waveInputSim, scaleSim = self.__WaveGenerate__(torch.conj(input_), waveAmp=self.waveAmp[1], isScale=True)

        inputCal = input_ / channel if channel is not None else input_
        waveInputExp, scaleExp = self.__WaveGenerate__(torch.conj(inputCal), waveAmp=self.waveAmp[1], isScale=True)

        self.waveInputSim_2 = waveInputSim
        self.scaleSim_2 = scaleSim
        self.waveInputExp_2 = waveInputExp
        self.scaleExp_2 = scaleExp

    def CalcPAPR(self):
        waveInput_1 = torch.flatten(self.waveInputExp_1)
        waveInput_2 = torch.flatten(self.waveInputExp_2)
        waveAmp = self.waveAmp

        PAPR_1 = 20*torch.log10(torch.max(torch.abs(waveInput_1))/waveAmp[0])
        PAPR_2 = 20*torch.log10(torch.max(torch.abs(waveInput_2))/waveAmp[1])

        return PAPR_1, PAPR_2

    def ConductAnal(self, engine=0, snr=None):
        headNum = self.headNum
        transmission = self.transmission
        simulation = self.simulation
        shrink = self.shrink
        freqOffset = self.freqOffset
        outputLen = self.outputWin[1] - self.outputWin[0]
        waveHead_1 = self.waveHead_1
        waveHead_2 = self.waveHead_2
        waveLenMax = self.waveLenMax
        device = self.device

        if snr is None:
            print('Experiment Mode!')
            waveInput_1 = self.waveInputExp_1
            waveInput_2 = self.waveInputExp_2
            scale = self.scaleExp_1 * self.scaleExp_2
            channelOut = self.channelOut

            testNum, waveLen = waveInput_1.size()
            testNumMax = math.floor(waveLenMax/waveLen)

            output = torch.zeros((testNum, outputLen), dtype=torch.cfloat, device=device)
            testOffset = 0
            while testOffset < testNum:
                testNumNow = min(testNum - testOffset, testNumMax)

                scaleNow = scale[testOffset: testOffset+testNumNow]
                waveInputNow_1 = waveInput_1[testOffset: testOffset+testNumNow, :]
                waveInputHeaded_1 = torch.concatenate((waveHead_1, waveInputNow_1, waveHead_1), axis=0)
                waveInputAll_1 = 10**(engine/20)*torch.flatten(waveInputHeaded_1)

                waveInputNow_2 = waveInput_2[testOffset: testOffset+testNumNow, :]
                waveInputHeaded_2 = torch.concatenate((waveHead_2, waveInputNow_2, waveHead_2), axis=0)
                waveInputAll_2 = torch.flatten(waveInputHeaded_2)
                waveInputAll = torch.stack((waveInputAll_1, waveInputAll_2), axis=0).detach().cpu().numpy()

                waveOutputAll, snrList, cfoList = transmission.Tx2Rx(waveInputAll, rxNum=1, gainOffsetList=[-engine, 0], padAmpList=[0, 1], shrink=shrink, freqOffset=freqOffset)
                waveOutputNow = torch.reshape(torch.tensor(waveOutputAll).to(waveInput_1), (testNumNow+2*headNum, -1))
                output[testOffset: testOffset+testNumNow] = self.__WaveAnalyze__(waveOutputNow, scaleNow, channelOut)
                SNR = snrList[0]
                CFO = cfoList[0]
                testOffset += testNumNow

            SNR = sum(snrList) / len(snrList)
            CFO = sum(cfoList) / len(cfoList)
        else:
            print('Simulation Mode!')
            waveInput_1 = self.waveInputSim_1
            waveInput_2 = self.waveInputSim_2
            scale = self.scaleSim_1 * self.scaleSim_2
            channelOut = torch.ones((outputLen), device=device)

            testNum, waveLen = waveInput_1.size()
            waveInputHeaded_1 = torch.concatenate((waveHead_1, waveInput_1, waveHead_1), axis=0)
            waveInputHeaded_2 = torch.concatenate((waveHead_2, waveInput_2, waveHead_2), axis=0)

            waveOutputList = []
            for testIdx in range(testNum+2*headNum):
                waveOutput = simulation.Tx2Rx(waveInputHeaded_1[testIdx, :], waveInputHeaded_2[testIdx, :], \
                    shrink=shrink[0]/shrink[1], freqOffset=freqOffset, snr=snr)
                waveOutputList.append(waveOutput)
            SNR = snr
            CFO = 0
            waveOutput = torch.reshape(torch.stack(waveOutputList, axis=0), (testNum+2*headNum, -1))
            output = self.__WaveAnalyze__(waveOutput, scale, channelOut)

        return output, SNR, CFO

class ConvAcc():
    def __init__(self, chanInNum, chanOutNum, imageSize, kernelSize, chanZad=41, cycleShift=[41, 137], param=GetParam()):
        self.chanInNum = chanInNum
        self.chanOutNum = chanOutNum
        self.imageSize = imageSize
        self.kernelSize = kernelSize
        assert kernelSize[0] % 2 == 1
        assert kernelSize[1] % 2 == 1

        padSize = [(k-1)//2 for k in kernelSize]
        specLen = 2*(imageSize[1]+padSize[1])*padSize[0]+padSize[1]+(imageSize[1]+padSize[1])*imageSize[0]
        if specLen % 2 == 1:
            specLen += 1
        self.padSize = padSize
        self.specLen = specLen
        self.device = param['device']

        shrinkMax = param['shrinkMax']
        self.chanInNumNewList = []
        self.convCoreList = []
        self.maskList = []
        for chanInNumNew, groupNum in [(shrinkMax, chanInNum // shrinkMax), (chanInNum % shrinkMax, 1)]:
            if chanInNumNew == 0 or groupNum == 0:
                continue
            outputWin = [chanInNumNew*specLen-1, chanInNumNew*specLen+specLen-1]
            convCore = ConvCore(inputLen_1=specLen*chanInNumNew, inputLen_2=specLen*chanInNumNew, outputWin=outputWin, param=param)
            mask = self.__Mask__(chanInNumNew, chanZad, cycleShift)
            for _ in range(groupNum):
                self.chanInNumNewList.append(chanInNumNew)
                self.convCoreList.append(copy.deepcopy(convCore))
                self.maskList.append(mask)

    def __Mask__(self, chanInNumNew, chanZad, cycleShift):
        chanOutNum = self.chanOutNum
        imageSize = self.imageSize
        kernelSize = self.kernelSize
        padSize = self.padSize
        device = self.device

        chanInAxis = __Zadoff__(chanInNumNew, chanZad, device=device)
        phaseShift = [cycleShift[0]*2*math.pi/imageSize[0], cycleShift[1]*2*math.pi/imageSize[1]]
        imageAxisX = torch.exp(1j*phaseShift[0]/360 * torch.arange(imageSize[0], device=device))
        imageAxisY = torch.exp(1j*phaseShift[1]/360 * torch.arange(imageSize[1], device=device))
        kernelAxisX = torch.exp(1j*phaseShift[0]/360 * torch.arange(-padSize[0], +padSize[0]+1, device=device))
        kernelAxisY = torch.exp(1j*phaseShift[1]/360 * torch.arange(-padSize[1], +padSize[1]+1, device=device))
        inputMask = chanInAxis.unsqueeze(1).unsqueeze(2).repeat(1, imageSize[0], imageSize[1]) * \
            imageAxisX.unsqueeze(0).unsqueeze(2).repeat(chanInNumNew, 1, imageSize[1]) * \
            imageAxisY.unsqueeze(0).unsqueeze(1).repeat(chanInNumNew, imageSize[0], 1)
        kernelMask = torch.conj(chanInAxis.unsqueeze(1).unsqueeze(2).repeat(1, kernelSize[0], kernelSize[1]) * \
            kernelAxisX.unsqueeze(0).unsqueeze(2).repeat(chanInNumNew, 1, kernelSize[1]) * \
            kernelAxisY.unsqueeze(0).unsqueeze(1).repeat(chanInNumNew, kernelSize[0], 1))
        kernelMask = kernelMask.unsqueeze(1).repeat(1, chanOutNum, 1, 1)
        outputMask = torch.conj(imageAxisX.unsqueeze(0).unsqueeze(2).repeat(chanOutNum, 1, imageSize[1]) * \
            imageAxisY.unsqueeze(0).unsqueeze(1).repeat(chanOutNum, imageSize[0], 1))

        return inputMask, kernelMask, outputMask

    def __Encode__(self, input_):
        chanOutNum = self.chanOutNum
        imageSize = self.imageSize
        padSize = self.padSize
        specLen = self.specLen
        batchNum = self.batchNum
        chanInNum = input_.size()[1]

        specInputList = torch.zeros((batchNum, chanInNum, specLen), dtype=torch.complex64, device=self.device)
        for batchIdx in range(batchNum):
            for chanInIdx in range(chanInNum):
                inputNow = input_[batchIdx, chanInIdx, :, :]
                offset = (padSize[1]+imageSize[1])*padSize[0]+padSize[1]
                for imageIdxX in range(imageSize[0]):
                    specInputList[batchIdx, chanInIdx, offset: offset+imageSize[1]] = inputNow[imageIdxX, :]
                    offset = offset + imageSize[1] + padSize[1]

        specInputListMult = specInputList.unsqueeze(1).repeat(1, chanOutNum, 1, 1)
        specInputListMult = torch.permute(specInputListMult, (1, 0, 2, 3))
        specInputListLine = torch.reshape(specInputListMult, (batchNum*chanOutNum, chanInNum*specLen))

        return specInputListLine

    def __Decode__(self, specOutputListLine):
        chanOutNum = self.chanOutNum
        imageSize = self.imageSize
        padSize = self.padSize
        batchNum = self.batchNum

        specOutputList = torch.reshape(specOutputListLine, (chanOutNum, batchNum, self.specLen))
        specOutputList = torch.permute(specOutputList, (1, 0, 2))
        output = torch.zeros((batchNum, chanOutNum, imageSize[0], imageSize[1]), dtype=torch.complex64, device=self.device)
        for batchIdx in range(batchNum):
            for chanOutIdx in range(chanOutNum):
                specOutputNow = specOutputList[batchIdx, chanOutIdx, :]
                offset = 0
                for imageIdxX in range(imageSize[0]):
                    output[batchIdx, chanOutIdx, imageIdxX, :] = specOutputNow[offset: offset+imageSize[1]]
                    offset = offset + imageSize[1] + padSize[1]

        return output

    def Calibrate(self, epoch=10000, folder=None):
        for convCore in self.convCoreList:
            convCore.Calibrate(epoch=epoch, folder=folder)

    def Init(self, kernelTensor, batchNum=1):
        kernel = kernelTensor.detach()
        assert kernel.size()[0] == self.chanInNum
        assert kernel.size()[1] == self.chanOutNum
        assert kernel.size()[2] == self.kernelSize[0]
        assert kernel.size()[3] == self.kernelSize[1]
        chanOutNum = self.chanOutNum
        imageSize = self.imageSize
        kernelSize = self.kernelSize
        padSize = self.padSize
        specLen = self.specLen

        chanInNumOffset = 0
        for chanInNumNew, convCore, mask in \
            zip(self.chanInNumNewList, self.convCoreList, self.maskList):

            _, kernelMask, _ = mask
            kernelNew = kernel[chanInNumOffset: chanInNumOffset+chanInNumNew, :, :, :] * kernelMask
            chanInNumOffset += chanInNumNew

            specKernelList = torch.zeros((chanOutNum, chanInNumNew, specLen), dtype=torch.complex64, device=self.device)
            for chanInIdx in range(chanInNumNew):
                for chanOutIdx in range(chanOutNum):
                    offset = 0
                    for kernelIdxX in range(kernelSize[0]):
                        specKernelList[chanOutIdx, chanInIdx, offset: offset+kernelSize[1]] = kernelNew[chanInIdx, chanOutIdx, kernelIdxX, :]
                        offset = offset + imageSize[1] + padSize[1]
            specKernelListMult = specKernelList.unsqueeze(0).repeat(batchNum, 1, 1, 1)
            specKernelListMult = torch.permute(specKernelListMult, (1, 0, 2, 3))
            specKernelListLine = torch.reshape(specKernelListMult, (batchNum*chanOutNum, chanInNumNew*specLen))

            convCore.LoadWave_2(specKernelListLine)

        self.batchNum = batchNum

    def InferAnal(self, inputTensor, engine=0, snr=None):
        input_ = inputTensor.detach()
        assert input_.size()[0] == self.batchNum
        assert input_.size()[1] == self.chanInNum
        assert input_.size()[2] == self.imageSize[0]
        assert input_.size()[3] == self.imageSize[1]

        output = torch.zeros((self.batchNum, self.chanOutNum, self.imageSize[0], self.imageSize[1]), dtype=torch.complex64, device=self.device)
        chanInNumOffset = 0
        for chanInNumNew, convCore, mask in zip(self.chanInNumNewList, self.convCoreList, self.maskList):
            inputMask, _, outputMask = mask
            inputNew = input_[:, chanInNumOffset: chanInNumOffset+chanInNumNew, :, :]
            chanInNumOffset += chanInNumNew

            specInputListLine = self.__Encode__(inputNew*inputMask)
            convCore.LoadWave_1(specInputListLine)

            inputPAPR, kernelPAPR = convCore.CalcPAPR()
            specOutputListLine, SNR, CFO = convCore.ConductAnal(engine=engine, snr=snr)

            output += self.__Decode__(specOutputListLine) * outputMask

            print('PAPR input=%.2fdB kernel=%.2fdB'%(inputPAPR, kernelPAPR))
            print('SNR=%.2fdB CFO=%.2fHz'%(SNR, CFO))

        return output

class FCAcc():
    def __init__(self, inputSize, outputSize, inputZad=41, param=GetParam()):
        self.inputSize = inputSize
        self.outputSize = outputSize

        outputSizeNew = outputSize if outputSize%2==0 else outputSize+1
        self.outputSizeNew = outputSizeNew
        self.device = param['device']

        shrinkMax = param['shrinkMax']
        self.inputSizeNewList = []
        self.specLenList = []
        self.convCoreList = []
        self.maskList = []
        for inputSizeNew, groupNum in [(shrinkMax, inputSize // shrinkMax), (inputSize % shrinkMax, 1)]:
            if inputSizeNew == 0 or groupNum == 0:
                continue
            specLen = inputSizeNew * outputSizeNew
            outputWin = [inputSizeNew*outputSizeNew-1, (inputSizeNew+1)*outputSizeNew-1]
            convCore = ConvCore(inputLen_1=specLen, inputLen_2=specLen, outputWin=outputWin, param=param)

            inputMask = __Zadoff__(inputSizeNew, inputZad, device=self.device)
            outputMask = __Zadoff__(outputSizeNew, inputZad, device=self.device)
            weightMask = torch.conj(inputMask.unsqueeze(1).repeat(1, outputSizeNew) * outputMask.unsqueeze(0).repeat(inputSizeNew, 1))

            for _ in range(groupNum):
                self.inputSizeNewList.append(inputSizeNew)
                self.specLenList.append(specLen)
                self.convCoreList.append(copy.deepcopy(convCore))
                self.maskList.append((inputMask, weightMask, outputMask))

    def __Encode__(self, input_, specLen):
        outputSizeNew = self.outputSizeNew
        inputSizeNew = input_.size()[1]

        specInputList = torch.zeros((self.batchNum, specLen), dtype=torch.complex64, device=self.device)
        for inputIdx in range(inputSizeNew):
            specInputList[:, (inputIdx+1)*outputSizeNew-1] = input_[:, inputIdx]

        return specInputList

    def __Decode__(self, specOutputList):
        return torch.fliplr(specOutputList)[:, : self.outputSize]

    def Calibrate(self, epoch=10000, folder=None):
        for convCore in self.convCoreList:
            convCore.Calibrate(epoch=epoch, folder=folder)

    def Init(self, weightTensor, batchNum=1):
        weight = weightTensor.detach()
        assert weight.size()[0] == self.inputSize
        assert weight.size()[1] == self.outputSize
        outputSize = self.outputSize
        outputSizeNew = self.outputSizeNew

        inputSizeOffset = 0
        for inputSizeNew, convCore, mask in \
            zip(self.inputSizeNewList, self.convCoreList, self.maskList):

            _, weightMask, _ = mask
            weightNew = torch.zeros((inputSizeNew, outputSizeNew), dtype=torch.complex64, device=self.device)
            weightNew[:, : outputSize] = weight[inputSizeOffset: inputSizeOffset+inputSizeNew, :] * weightMask
            inputSizeOffset += inputSizeNew

            specWeightList = torch.flatten(weightNew)
            specWeightListMult = specWeightList.unsqueeze(0).repeat(batchNum, 1)
            convCore.LoadWave_2(specWeightListMult)

        self.batchNum = batchNum

    def InferAnal(self, inputTensor, engine=0, snr=None):
        input_ = inputTensor.detach()
        assert input_.size()[0] == self.batchNum
        assert input_.size()[1] == self.inputSize
        outputSize = self.outputSize

        output = torch.zeros((self.batchNum, outputSize), dtype=torch.complex64, device=self.device)
        inputSizeOffset = 0
        for inputSizeNew, specLen, convCore, mask in zip(self.inputSizeNewList, self.specLenList, self.convCoreList, self.maskList):
            inputMask, _, outputMask = mask
            inputNew = input_[:, inputSizeOffset: inputSizeOffset+inputSizeNew] * inputMask
            inputSizeOffset += inputSizeNew

            specInputList = self.__Encode__(inputNew, specLen)
            convCore.LoadWave_1(specInputList)

            inputPAPR, weightPAPR = convCore.CalcPAPR()
            specOutputList, SNR, CFO = convCore.ConductAnal(engine=engine, snr=snr)

            output += self.__Decode__(specOutputList) * outputMask[:outputSize]

            print('PAPR input=%.2fdB weight=%.2fdB'%(inputPAPR, weightPAPR))
            print('SNR=%.2fdB CFO=%.2fHz'%(SNR, CFO))

        return output
