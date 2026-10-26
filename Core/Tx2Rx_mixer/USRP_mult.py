#!/usr/bin/env python3


from gnuradio import blocks
import pmt
from gnuradio import gr
import sys
import signal
from gnuradio import uhd
import time

import numpy as np
from scipy.io import savemat

import argparse
parser = argparse.ArgumentParser(description='USRP control script')
parser.add_argument('--addr', default='192.168.70.3,192.168.70.8', type=str,
    help='USRP IP addresses, split by ","')
parser.add_argument('--rate', default=25e6, type=float,
    help='Sampling rate in Hz for both channels')
parser.add_argument('--time', default=1.0, type=float,
    help='Time duration in s for both channels, -1 for infinity loop with key exit')
parser.add_argument('--sync', default=0, type=int, choices=[0, 1],
    help='The TX/RX synchronization, 0 for unsync and 1 for sync')
parser.add_argument('--clock', default=200e6, type=float,
    help='The TX/RX clock rate')
parser.add_argument('--save', default=None, type=str,
    help='The saving file name')
parser.add_argument('--fileTx', default='.', type=str,
    help='Binary file to load IQ samples, split by ",", "." for not used')
parser.add_argument('--freqTx', default='3e9', type=str,
    help='Carrier frequency in Hz, split by ",", "." for not used')
parser.add_argument('--gainTx', default='20', type=str,
    help='Power gain in dB, split by ",", "." for not used')
parser.add_argument('--fileRx', default='.', type=str,
    help='Binary file to save IQ samples, split by ",", "." for not used')
parser.add_argument('--freqRx', default='3e9', type=str,
    help='Carrier frequency in Hz, split by ",", "." for not used')
parser.add_argument('--gainRx', default='10', type=str,
    help='Power gain in dB, split by ",", "." for not used')



def mySplit(inputStr, symbol=','):
    return [elem for elem in inputStr.split(symbol) if elem]

class USRP_Control(gr.top_block):

    def __init__(self, optDict):
        ADDR = optDict['addr'].split(',')
        RATE = optDict['rate']
        SYNC = 'internal' if optDict['sync']==0 else 'external'
        CLOCK = optDict['clock']
        SAVE = optDict['save']
        FILE_TX = mySplit(optDict['fileTx'])
        FREQ_TX = mySplit(optDict['freqTx'])
        GAIN_TX = mySplit(optDict['gainTx'])
        FILE_RX = mySplit(optDict['fileRx'])
        FREQ_RX = mySplit(optDict['freqRx'])
        GAIN_RX = mySplit(optDict['gainRx'])
        
        addrStr = ''
        for addrIdx in range(len(ADDR)):
            addrStr = addrStr + 'addr'+str(addrIdx)+'='+ADDR[addrIdx]+','
        if addrStr[-1] == ',':
            addrStr = addrStr[:-1]

        gr.top_block.__init__(self, "Not titled yet")

        rateTxGt = np.nan
        freqTxGt = np.ones((len(FILE_TX))) * np.nan
        gainTxGt = np.ones((len(FILE_TX))) * np.nan
        if any([fileTx!='.' for fileTx in FILE_TX]):
            self.uhd_usrp_sink = uhd.usrp_sink(
                ",".join(("", addrStr+", master_clock_rate="+str(CLOCK)+", clock_source="+SYNC+", time_source="+SYNC)),
                uhd.stream_args(cpu_format="fc32", args='', channels=list(range(0, len(FILE_TX)))), "")
            self.uhd_usrp_sink.set_clock_source(SYNC, 0)
            self.uhd_usrp_sink.set_time_source(SYNC, 0)
            self.uhd_usrp_sink.set_samp_rate(RATE)
            self.uhd_usrp_sink.set_time_unknown_pps(uhd.time_spec(0))
            rateTxGt = self.uhd_usrp_sink.get_samp_rate()

            for channelIdx in range(len(FILE_TX)):
                fileTx = FILE_TX[channelIdx]
                freqTx = float(FREQ_TX[-1]) if channelIdx>=len(FREQ_TX) else float(FREQ_TX[channelIdx])
                gainTx = float(GAIN_TX[-1]) if channelIdx>=len(GAIN_TX) else float(GAIN_TX[channelIdx])

                if fileTx != '.':
                    self.uhd_usrp_sink.set_center_freq(freqTx, channelIdx)
                    self.uhd_usrp_sink.set_antenna("TX/RX", channelIdx)
                    self.uhd_usrp_sink.set_gain(gainTx, channelIdx)
                    freqTxGt[channelIdx] = self.uhd_usrp_sink.get_center_freq(channelIdx)
                    gainTxGt[channelIdx] = self.uhd_usrp_sink.get_gain(channelIdx)

                    self.blocks_file_source = blocks.file_source(gr.sizeof_gr_complex*1, fileTx, True, 0, 0)
                    self.blocks_file_source.set_begin_tag(pmt.PMT_NIL)

                    self.connect((self.blocks_file_source, 0), (self.uhd_usrp_sink, channelIdx))
                else:
                    self.uhd_usrp_sink.set_center_freq(10e6, channelIdx)
                    self.uhd_usrp_sink.set_antenna("TX/RX", channelIdx)
                    self.uhd_usrp_sink.set_gain(1, channelIdx)

                    self.blocks_null_source = blocks.null_source(gr.sizeof_gr_complex*1)

                    self.connect((self.blocks_null_source, 0), (self.uhd_usrp_sink, channelIdx))

        rateRxGt = np.nan
        freqRxGt = np.ones((len(FILE_RX))) * np.nan
        gainRxGt = np.ones((len(FILE_RX))) * np.nan
        if any([fileRx!='.' for fileRx in FILE_RX]):
            self.uhd_usrp_source = uhd.usrp_source(
                ",".join(("", addrStr+", master_clock_rate="+str(CLOCK)+", clock_source="+SYNC+", time_source="+SYNC)),
                uhd.stream_args(cpu_format="fc32", args='', channels=list(range(0, len(FILE_RX)))))
            self.uhd_usrp_source.set_clock_source(SYNC, 0)
            self.uhd_usrp_source.set_time_source(SYNC, 0)
            self.uhd_usrp_source.set_samp_rate(RATE)
            self.uhd_usrp_source.set_time_unknown_pps(uhd.time_spec(0))
            rateRxGt = self.uhd_usrp_source.get_samp_rate()

            for channelIdx in range(len(FILE_RX)):
                fileRx = FILE_RX[channelIdx]
                freqRx = float(FREQ_RX[-1]) if channelIdx>=len(FREQ_RX) else float(FREQ_RX[channelIdx])
                gainRx = float(GAIN_RX[-1]) if channelIdx>=len(GAIN_RX) else float(GAIN_RX[channelIdx])

                if fileRx != '.':
                    self.uhd_usrp_source.set_center_freq(freqRx, channelIdx)
                    self.uhd_usrp_source.set_antenna("RX2", channelIdx)
                    self.uhd_usrp_source.set_gain(gainRx, channelIdx)
                    freqRxGt[channelIdx] = self.uhd_usrp_source.get_center_freq(channelIdx)
                    gainRxGt[channelIdx] = self.uhd_usrp_source.get_gain(channelIdx)

                    self.blocks_file_sink = blocks.file_sink(gr.sizeof_gr_complex*1, fileRx, False)
                    self.blocks_file_sink.set_unbuffered(False)

                    self.connect((self.uhd_usrp_source, channelIdx), (self.blocks_file_sink, 0))
                else:
                    self.uhd_usrp_source.set_center_freq(10e6, channelIdx)
                    self.uhd_usrp_source.set_antenna("RX2", channelIdx)
                    self.uhd_usrp_source.set_gain(1, channelIdx)

                    self.blocks_null_sink = blocks.null_sink(gr.sizeof_gr_complex*1)

                    self.connect((self.uhd_usrp_source, channelIdx), (self.blocks_null_sink, 0))

        if SAVE is not None:
            savemat(SAVE, {
                'rateTx': rateTxGt, 'freqTx': freqTxGt, 'gainTx': gainTxGt,
                'rateRx': rateRxGt, 'freqRx': freqRxGt, 'gainRx': gainRxGt})



def main(optDict, top_block_cls=USRP_Control, options=None):
    TIME = optDict['time']

    tb = top_block_cls(optDict)

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    if TIME > 0:
        time.sleep(TIME)
    else:
        try:
            input('Press Enter to quit: ')
        except EOFError:
            pass
    tb.stop()
    tb.wait()



if __name__ == '__main__':
    opt = parser.parse_args()
    optDict = {
        'addr': opt.addr,
        'rate': opt.rate,
        'time': opt.time,
        'sync': opt.sync,
        'clock': opt.clock,
        'save': opt.save,
        'fileTx': opt.fileTx,
        'freqTx': opt.freqTx,
        'gainTx': opt.gainTx,
        'fileRx': opt.fileRx,
        'freqRx': opt.freqRx,
        'gainRx': opt.gainRx,
    }
    main(optDict)
