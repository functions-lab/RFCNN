import numpy as np
import torch



class Simulation():
    def __init__(self, param):
        super(Simulation, self).__init__()

        self.sampleRate = param['sampleRate']
        self.upsample = param['upsample']
        self.carrierTx = param['carrierTx']
        self.device = param['device']

    def __AddNoise__(self, signalIn, snr):
        signalLen = signalIn.size()[0]
        sigma = torch.mean(torch.abs(signalIn)**2) / (10**(snr/10))
        signalOut = signalIn + torch.randn((signalLen)).to(signalIn)*torch.sqrt(sigma/2) + 1j*torch.randn((signalLen)).to(signalIn)*torch.sqrt(sigma/2)
        return signalOut

    def __Upsampling__(self, waveLow, rateLow, rateHigh):
        waveLenLow = waveLow.size()[0]
        waveLenHigh = int(np.round(waveLenLow/rateLow*rateHigh))
        mid = (waveLenLow+1)//2

        specLow = torch.fft.fft(waveLow) * rateHigh / rateLow
        specHigh = torch.zeros((waveLenHigh)).to(waveLow)
        specHigh[:mid] = specLow[:mid]
        specHigh[-mid:] = specLow[-mid:]
        waveHigh = torch.fft.ifft(specHigh)
        return waveHigh

    def __Filter__(self, waveIn, rateLow, rateHigh):
        specLen = int(np.round(waveIn.size()[0]/rateHigh*rateLow))
        mid = (specLen+1)//2

        specIn = torch.fft.fft(waveIn)
        specOut = specIn
        specOut[mid: -mid] = 0
        waveOut = torch.fft.ifft(specOut)
        return waveOut

    def Tx2Rx(self, packetTx_1, packetTx_2, shrink=1, freqOffset=0, snr=30):
        sampleRateTx = self.sampleRate
        sampleRateRx = self.sampleRate/shrink
        upsample = self.upsample
        carrierTx = self.carrierTx

        assert packetTx_1.size()[0] == packetTx_2.size()[0]
        packetTxLen = packetTx_1.size()[0]

        waveAxis = torch.arange(upsample*packetTxLen).to(packetTx_1)/upsample/sampleRateTx

        packetTx_1 = self.__Upsampling__(packetTx_1, sampleRateTx, upsample*sampleRateTx)
        packetTx_2 = self.__Upsampling__(packetTx_2, sampleRateTx, upsample*sampleRateTx)
        if carrierTx[0] > carrierTx[1]:
            waveRx = packetTx_1 * torch.conj(packetTx_2) * torch.exp(-1j*2*torch.pi*freqOffset*sampleRateTx*waveAxis)
        else:
            waveRx = torch.conj(packetTx_1) * packetTx_2 * torch.exp(-1j*2*torch.pi*freqOffset*sampleRateTx*waveAxis)
        waveRx = self.__Filter__(waveRx, sampleRateRx, upsample*sampleRateTx)
        waveRx = self.__AddNoise__(waveRx, snr)
        packetRx = waveRx[int(np.round(upsample*shrink/2))::int(np.round(upsample*shrink))]

        return packetRx
