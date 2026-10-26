import csv
import numpy as np
import pickle
from scipy.io import loadmat
import torch
from torch.utils.data.dataset import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm



def LoadData_MNIST(dataPath, skipHeader=False):
    targetList = []
    imageList = []
    with open(dataPath, mode='r') as dataCSV:
        lineList = list(csv.reader(dataCSV))
        if skipHeader:
            lineList = lineList[1:]
        for line in tqdm(lineList, desc='Loading'):
            targetList.append(int(line[0]))
            image = np.reshape(np.array([int(pixel) for pixel in line[1:]]), (28, 28)) / 255.0
            imageList.append(image[np.newaxis, :, :])
    return targetList, imageList

def LoadData_SVHN(dataPath):
    dict = loadmat(dataPath)
    targetList = [int(target) for target in list(dict['y'].squeeze(1)-1)]
    imageList = []
    for dataIdx in range(len(targetList)):
        image = dict['X'][:, :, :, dataIdx] / 255.0
        imageList.append(np.swapaxes(image, 0, 2))
    return targetList, imageList

def LoadData_CIFAR10(dataPath):
    with open(dataPath, 'rb') as data:
        dict = pickle.load(data, encoding='bytes')
        targetList = list(dict[b'labels'])
        imageList = []
        for dataIdx in tqdm(range(np.shape(dict[b'data'])[0]), desc='Loading'):
            imageList.append(np.reshape(dict[b'data'][dataIdx, :], (3, 32, 32)) / 255.0)
    return targetList, imageList

class DataReader_Image(Dataset):
    def __init__(self, dataset, phase, dataPath='', imageSize=28):
        super(DataReader_Image, self).__init__()

        normalizeRGB = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        augmentRGB = [transforms.RandomCrop(imageSize, padding=imageSize//8+1), transforms.RandomHorizontalFlip()]
        if dataset == 'MNIST':
            fileName = 'mnist_train.csv' if phase == 'train' else 'mnist_test.csv'
            targetList, imageOrigList = LoadData_MNIST(dataPath+'../Data/MNIST/'+fileName)
            self.transform = transforms.Normalize([0.5], [0.5])
        elif dataset == 'FMNIST':
            fileName = 'fashion-mnist_train.csv' if phase == 'train' else 'fashion-mnist_test.csv'
            targetList, imageOrigList = LoadData_MNIST(dataPath+'../Data/FMNIST/'+fileName, skipHeader=True)
            self.transform = transforms.Normalize([0.5], [0.5])
        elif dataset == 'SVHN':
            fileName = 'train_32x32.mat' if phase == 'train' else 'test_32x32.mat'
            targetList, imageOrigList = LoadData_SVHN(dataPath+'../Data/SVHN/'+fileName)
            self.transform = transforms.Compose((augmentRGB if phase == 'train' else []) + [normalizeRGB])
        elif dataset == 'CIFAR-10':
            fileList = ['data_batch_%d'%(idx) for idx in range(1, 6)] if phase == 'train' else ['test_batch']
            targetList = []
            imageOrigList = []
            for file in fileList:
                targetNowList, imageOrigNowList = LoadData_CIFAR10(dataPath+'../Data/CIFAR-10/cifar-10-batches-py/'+file)
                targetList += targetNowList
                imageOrigList += imageOrigNowList
            self.transform = transforms.Compose((augmentRGB if phase == 'train' else []) + [normalizeRGB])
        else:
            raise ValueError('Dataset %s NOT Found!'%(dataset))
        self.classNum = 10
        assert list(np.shape(imageOrigList[0])[1:]) == [imageSize, imageSize], \
            '--size must be the native resolution of %s'%(dataset)

        self.targetList = targetList
        self.imageList = imageOrigList

    def __getitem__(self, index):
        imageTensor = torch.from_numpy(self.imageList[index]).float()
        return self.targetList[index], self.transform(imageTensor)

    def __len__(self):
        return len(self.targetList)

    def GetShape(self):
        return list(np.shape(self.imageList[0])), self.classNum



class DataReader_DeepSig(Dataset):
    def __init__(self, dataPath, waveLen=1024, span=[0, 1],
        moduSet=list(range(24)), snrSet=list(range(-20, 32, 2))):
        super(DataReader_DeepSig, self).__init__()

        self.classNum = len(moduSet)

        dataList = []
        targetList = []
        for modu in tqdm(moduSet):
            for snr in snrSet:
                waveAll = loadmat(dataPath+'data_'+str(modu+1)+'_'+str(snr)+'.mat')['wave']

                dataNumAll = np.shape(waveAll)[0]
                startIdx = int(span[0]*dataNumAll)
                endIdx = int(span[1]*dataNumAll)
                if waveLen > np.shape(waveAll)[1]:
                    continue
                offset = (np.shape(waveAll)[1]-waveLen)//2
                for dataIdx in range(startIdx, endIdx):
                    wave = waveAll[dataIdx, offset: offset+waveLen]
                    dataList.append(wave[np.newaxis, np.newaxis, :].astype(complex))
                    targetList.append(moduSet.index(modu))

        self.dataList = dataList
        self.targetList = targetList

    def __getitem__(self, index):
        dataTensor = torch.from_numpy(self.dataList[index]).to(torch.cfloat)
        return self.targetList[index], dataTensor

    def __len__(self):
        return len(self.targetList)

    def GetShape(self):
        return list(np.shape(self.dataList[0])), self.classNum



class DataReader_CelebA(Dataset):
    def __init__(self, phase, dataPath='./', imageSize=64):
        super(DataReader_CelebA, self).__init__()

        self.transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(imageSize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.dataset = datasets.CelebA(root=dataPath+"../Data/CelebA", split=phase, download=True)
        self.dataShape = [3, imageSize, imageSize]

    def __getitem__(self, index):
        image, target = self.dataset.__getitem__(index)
        return target, self.transform(image)

    def __len__(self):
        return self.dataset.__len__()

    def GetShape(self):
        return self.dataShape, 10
