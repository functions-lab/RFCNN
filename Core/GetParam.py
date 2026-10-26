def GetParam():
    param = {}

    paramWave = {
        'inputPad': 0.5,
        'outputPad': 0.5,
        'cpRate': 0,
        'cpInt': 0,
        'waveAmp': [0.2, 0.15]}
    param.update(paramWave)

    paramHard = {
        'deviceTx': '192.168.70.5',
        'clockTx': 200e6,
        'carrierTx': [1.000e9, 0.915e9],
        'gainTx': [14, 31],

        'deviceRx': '192.168.70.9',
        'clockRx': 200e6,
        'carrierRx': [0.085e9],
        'gainRx': [5],

        'sampleRate': 50e6,
        'sync': 0,

        'shrinkMax': 200,
        'waveLenMax': 100000000,
        }
    param.update(paramHard)

    paramSim = {
        'upsample': 2,
        'device': 'cpu'}
    param.update(paramSim)

    return param
