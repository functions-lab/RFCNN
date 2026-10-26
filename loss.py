import numpy as np
from torch import nn



class Eval(nn.Module):
    def __init__(self, classNum=10):
        super(Eval, self).__init__()

        self.classNum = classNum

    def forward(self, pred, target):
        classNum = self.classNum
        dataNum = np.shape(target)[0]

        mat = np.zeros((classNum, classNum))
        for dataIdx in range(dataNum):
            predMax = np.argmax(pred[dataIdx, :])
            mat[predMax, target[dataIdx]] += 1
        acc = np.trace(mat) / np.sum(mat)

        return acc, mat

def AnalyzeFeat(feat):
    featInput, featConvList, featFCList = feat
    featDict = {'featInput': featInput.detach().cpu().numpy()}

    for name, featList in [('Conv', featConvList), ('FC', featFCList)]:
        for featIdx, (featMid, featOut) in enumerate(featList, start=1):
            featDict['feat%s_mid_%d'%(name, featIdx)] = featMid.detach().cpu().numpy()
            featDict['feat%s_out_%d'%(name, featIdx)] = featOut.detach().cpu().numpy()
        featDict['feat%sNum'%(name)] = len(featList)

    return featDict
