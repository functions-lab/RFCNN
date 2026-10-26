import matplotlib.pyplot as plt
from numba import jit
import numpy as np
import os, shutil
from scipy.io import loadmat


@jit(nopython=False, parallel=False)
def GetCorr(wave, winLen, deltaLen):

    waveLen = np.shape(wave)[0]
    waveConj = np.conj(wave)
    corrAll = np.ones((waveLen-deltaLen-winLen+1)) * np.nan

    zadoff_1 = wave[: winLen]
    zadoff_2 = waveConj[deltaLen: deltaLen+winLen]
    product = np.sum(zadoff_1 * zadoff_2)
    energy_1 = np.sum(np.abs(zadoff_1)**2)

    energy_2 = np.sum(np.abs(zadoff_2)**2)
    corrAll[0] = np.abs(product) / np.sqrt(energy_1*energy_2)

    for offset in range(waveLen-deltaLen-winLen):
        iqIn_1 = wave[offset+winLen]
        iqIn_2 = waveConj[offset+deltaLen+winLen]
        iqOut_1 = wave[offset]
        iqOut_2 = waveConj[offset+deltaLen]

        product += - iqOut_1*iqOut_2 + iqIn_1*iqIn_2
        energy_1 += - np.abs(iqOut_1)**2 + np.abs(iqIn_1)**2
        energy_2 += - np.abs(iqOut_2)**2 + np.abs(iqIn_2)**2
        res = np.abs(product) / (energy_1*energy_2)**0.5
        corrAll[offset+1] = res

    return corrAll

class Transmission():
    def __init__(self, param):
        super(Transmission, self).__init__()

        assert len(param['carrierTx']) == len(param['gainTx'])
        assert len(param['carrierTx']) <= 2*(len(param['deviceTx'].split(','))+1)
        self.deviceTx = param['deviceTx']
        self.clockTx = param['clockTx']
        self.carrierTx = param['carrierTx']
        self.gainTx = param['gainTx']
        self.channelTx = len(param['carrierTx'])

        assert len(param['carrierRx']) == len(param['gainRx'])
        assert len(param['carrierRx']) <= 2*(len(param['deviceRx'].split(','))+1)
        self.deviceRx = param['deviceRx']
        self.clockRx = param['clockRx']
        self.carrierRx = param['carrierRx']
        self.gainRx = param['gainRx']
        self.channelRx = len(param['carrierRx'])
        self.waveAmp = param['waveAmp']

        self.sampleRate = param['sampleRate']
        self.sync = param['sync']



    def __CheckSaturation__(self, packet, threshold=1):
        ratio = sum(np.abs(packet)>threshold)/np.shape(packet)[0]
        if ratio > 0:
            print('Warning: Packet Saturated for '+str(ratio*100)+'%')
    
    def __Wave2File__(self, wave, fileName):
        file = open(fileName, 'wb')
        wave.astype(np.complex64).tofile(file)
        file.close()

    def __File2Wave__(self, fileName):
        file = open(fileName, 'rb')
        wave = np.fromfile(file, dtype=np.complex64)
        file.close()
        return wave

    def __ZadoffDetection__(self, wave, winLen, deltaLen, threshold=0.8, neighbor=100, plotName=None):
        waveLen = np.shape(wave)[0]
        corrAll = GetCorr(wave.astype(np.complex128), winLen, deltaLen)

        if plotName is not None:
            fig, ax = plt.subplots()
            ax.plot(np.arange(len(corrAll)), corrAll)
            fig.savefig(plotName+'.png')
            plt.close(fig)

        offsetList = []
        corrList = []
        for offset in range(neighbor, waveLen-deltaLen-winLen-neighbor):
            corr = corrAll[offset]
            if corr > threshold:
                if corr < np.max(corrAll[offset-neighbor: offset]):
                    continue
                if corr < np.max(corrAll[offset+1: offset+neighbor+1]):
                    continue
                offsetList.append(offset)
                corrList.append(corr)

        return offsetList, corrList

    def __GetEnergy__(self, wave):
        waveCal = wave
        energy = np.mean(np.abs(waveCal)**2)
        return energy



    def Tx2Rx(self, packetTxList, rxNum, gainOffsetList=None, padAmpList=None, shrink=[1, 1], freqOffset=0):
        thisPath = os.path.abspath(__file__)[:-len(os.path.basename(__file__))]

        txNum, packetTxLen = np.shape(packetTxList)
        packetRxLen = int(packetTxLen/shrink[0]*shrink[1])
        sampleRateTx = self.sampleRate
        sampleRateRx = self.sampleRate/shrink[0]*shrink[1]
        for txIdx in range(txNum):
            self.__CheckSaturation__(packetTxList[txIdx, :])
        assert txNum <= self.channelTx
        assert txNum <= len(self.waveAmp)
        assert rxNum <= self.channelRx

        padLen = int(max(0.1e-3*sampleRateTx*shrink[0], round(0.1*packetTxLen)))
        capNum = 3
        noiseNum = 1000
        zadoffSet = np.array([139, 839])
        zadoffTxSet = zadoffSet * shrink[0]
        zadoffRxSet = zadoffSet * shrink[1]

        bufferPath = thisPath+"Buffer/"
        if os.path.isdir(bufferPath):
            shutil.rmtree(bufferPath)
        os.mkdir(bufferPath)

        headTx = np.zeros((txNum, sum(zadoffTxSet)*3), dtype=np.complex64)
        offset = 0
        for zadoffLen in zadoffTxSet:
            zadoff = np.exp(1j*2*np.pi*np.random.rand(txNum, zadoffLen))
            headTx[:, offset: offset+2*zadoffLen] = np.tile(zadoff, 2)
            offset += 3 * zadoffLen
        headTxLen = sum(zadoffTxSet)*3
        headRxLen = sum(zadoffRxSet)*3

        padTxList = np.zeros((txNum, padLen), dtype=complex)
        repeatNum = int(np.ceil(padLen/packetTxLen))
        for txIdx in range(txNum):
            padAmp = 0 if padAmpList is None else padAmpList[txIdx]
            padTxList[txIdx, :] = padAmp * np.repeat(packetTxList[txIdx, :], repeatNum)[: padLen]

        waveTxLen = 2*padLen + headTxLen + packetTxLen
        waveRxLen = int(waveTxLen/shrink[0]*shrink[1])

        fileTxStr = ''
        freqTxStr = ''
        gainTxStr = ''
        for txIdx in range(txNum):
            headAmp = self.waveAmp[txIdx]
            gainOffset = 0 if gainOffsetList is None else gainOffsetList[txIdx]
            waveTx = np.concatenate((padTxList[txIdx, :], headAmp*headTx[txIdx, :], packetTxList[txIdx, :], padTxList[txIdx, :]), axis=0)
            fileTxStr += bufferPath+'Tx_'+str(txIdx)+'.bin,'
            freqTxStr += str(self.carrierTx[txIdx]) + ','
            gainTxStr += str(self.gainTx[txIdx]+gainOffset) + ','
            self.__Wave2File__(waveTx, bufferPath+'Tx_'+str(txIdx)+'.bin')
        resultFileTx = bufferPath + 'resultTx.mat'
        
        fileRxStr = ''
        freqRxStr = ''
        gainRxStr = ''
        for rxIdx in range(rxNum):
            fileRxStr += bufferPath+'Rx_'+str(rxIdx)+'.bin,'
            freqRxStr += str(self.carrierRx[rxIdx]+freqOffset*sampleRateTx) + ','
            gainRxStr += str(self.gainRx[rxIdx]) + ','
        resultFileRx = bufferPath + 'resultRx.mat'
        
        waveRxList = np.nan * np.ones((rxNum, capNum*waveRxLen), dtype=complex)
        offsetZadoffList = np.zeros((rxNum), dtype=int)
        offsetPacketList = np.zeros((rxNum), dtype=int)
        while True:
            os.system('bash '+thisPath+'Tx2Rx_mult.sh ' + \
                str(max(1.0, (capNum+1)*waveTxLen/sampleRateTx))+' '+ str(self.sync)+' '+ \
                self.deviceTx+' '+str(sampleRateTx)+' '+str(self.clockTx)+' '+ \
                fileTxStr+' '+freqTxStr+' '+gainTxStr+' '+resultFileTx+' '+ \
                self.deviceRx+' '+str(sampleRateRx)+' '+str(self.clockRx)+' '+ \
                fileRxStr+' '+freqRxStr+' '+gainRxStr+' '+resultFileRx)

            fig, ax = plt.subplots(rxNum)
            isFound = True
            for rxIdx in range(rxNum):
                axNow = ax if rxNum == 1 else ax[rxIdx]

                waveTemp = self.__File2Wave__(bufferPath+'Rx_'+str(rxIdx)+'.bin')
                waveRx = waveTemp[-capNum*waveRxLen:]
                waveRxList[rxIdx, :] = waveRx
                axNow.plot(np.arange(capNum*waveRxLen), 20*np.log10(np.abs(waveRx)+1e-10))
                fig.savefig(bufferPath+'detection.png')

                offsetList, corrList = self.__ZadoffDetection__( \
                    waveRx[: waveRxLen+headRxLen], zadoffRxSet[-1], zadoffRxSet[-1], threshold=0.7, plotName=bufferPath+'corralation_1')
                offsetListAfter, _ = self.__ZadoffDetection__( \
                    waveRx[waveRxLen: 2*waveRxLen+headRxLen], zadoffRxSet[-1], zadoffRxSet[-1], threshold=0.7, plotName=bufferPath+'corralation_2')
                if (len(offsetList)==0)or(len(offsetListAfter)==0):
                    isFound = False
                    break
                offsetIdx = np.argmax(np.array(corrList))
                offsetZadoff = offsetList[offsetIdx]-3*sum(zadoffRxSet[: -1])
                offsetPacket = offsetZadoff + headRxLen
                if offsetZadoff <= 0:
                    isFound = False
                    break
                offsetZadoffList[rxIdx] = offsetZadoff
                offsetPacketList[rxIdx] = offsetPacket

                axNow.vlines(offsetZadoff, ymin=-100, ymax=0, color='black')
                axNow.vlines(offsetPacket, ymin=-100, ymax=0, color='black')
                axNow.vlines(offsetPacket+packetRxLen, ymin=-100, ymax=0, color='black')
                axNow.set_ylim(bottom=-100, top=0)
                fig.savefig(bufferPath+'detection.png')
            if isFound:
                break

        packetRxList = np.nan * np.ones((rxNum, packetRxLen), dtype=complex)
        cfoList = np.nan * np.ones((rxNum))
        snrList = np.nan * np.ones((rxNum))
        for rxIdx in range(rxNum):
            waveRx = waveRxList[rxIdx, :]
            offsetZadoff = offsetZadoffList[rxIdx]
            offsetPacket = offsetPacketList[rxIdx]

            if self.sync == 0:
                cfoSet = []
                offset = offsetZadoff
                for zadoffLen in zadoffRxSet:
                    zadoff_1 = waveRx[offset: offset+zadoffLen]
                    zadoff_2 = waveRx[offset+zadoffLen: offset+2*zadoffLen] * np.exp(1j*2*np.pi*freqOffset*sampleRateTx*zadoffLen/sampleRateRx)
                    cfoTemp = -sampleRateRx/zadoffLen*np.angle(np.sum(zadoff_1 * np.conj(zadoff_2)))/2/np.pi
                    cfoSet.append(cfoTemp)
                    offset += 3*zadoffLen

                    waveRx[offsetZadoff: offsetZadoff+headRxLen] = \
                        waveRx[offsetZadoff: offsetZadoff+headRxLen] * np.exp(-1j*2*np.pi*np.arange(headRxLen)/sampleRateRx*cfoTemp)
                cfo = sum(cfoSet)
            else:
                resultRx = loadmat(resultFileRx)
                cfo = self.carrierRx[rxIdx]+freqOffset*sampleRateTx - resultRx['freqRx'][rxIdx]
            cfoList[rxIdx] = cfo
            packetRx = waveRx[offsetPacket: offsetPacket+packetRxLen]
            packetRx = packetRx * np.exp(-1j*2*np.pi*np.arange(packetRxLen)/sampleRateRx*cfo)
            packetRxList[rxIdx, :] = packetRx

            noiseList = []
            for noiseIdx in range(noiseNum):
                startIdx = int(capNum*waveRxLen/noiseNum*noiseIdx)
                endIdx = int(capNum*waveRxLen/noiseNum*noiseIdx) + int(np.ceil(40e-6*sampleRateRx))
                noiseSym = waveRx[startIdx: endIdx]
                noise = self.__GetEnergy__(noiseSym)
                noiseList.append(noise)
            noise = np.percentile(noiseList, 10)
            signal = self.__GetEnergy__(packetRx) - noise
            snr = 10 * np.log10(signal/noise)
            snrList[rxIdx] = snr

            axNow = ax if rxNum == 1 else ax[rxIdx]
            axNow.set_title("CFO: %.2fHz SNR: %.2fdB"%(cfo, snr))

        fig.savefig(bufferPath+'detection.png')
        plt.close(fig)

        return packetRxList, snrList, cfoList
