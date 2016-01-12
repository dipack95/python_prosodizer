import pandas as pd
import numpy as np
from scipy.stats import f
import matplotlib.pyplot as plt
import os

np.set_printoptions(threshold=np.inf)

targetPath = os.path.abspath("../sounds/sounds/hack-the-talk-exotel-master/training_dataset")

zcrFiles = [os.path.join(path, name)
             for path, dirs, files in os.walk(targetPath)
             for name in files if ("angry" in name and name.endswith(("-zeroCrossing.csv")))]

powerFiles = [os.path.join(path, name)
             for path, dirs, files in os.walk(targetPath)
             for name in files if ("angry" in name and name.endswith(("-powerSpectrum.csv")))]

entropyFiles = [os.path.join(path, name)
             for path, dirs, files in os.walk(targetPath)
             for name in files if ("angry" in name and name.endswith(("-entropy.csv")))]


def splitSignal(powerFile, zcrFile, entropyFile):

    power = np.nan_to_num(np.array(pd.read_csv(powerFile, header=None), dtype='float64'))
    zcr = np.nan_to_num(np.array(pd.read_csv(zcrFile, header=None), dtype='float64'))
    entropy = np.nan_to_num(np.array(pd.read_csv(entropyFile, header=None), dtype='float64'))
    zcr = np.ravel(zcr)
    entropy = np.ravel(entropy)
    divFrames = []

    jump = 25
    divFrameLength = 100
    startIndex = 0
    endIndex = divFrameLength
    fileLen = len(power)
    rangeLen = fileLen // endIndex
    
    if fileLen > rangeLen:
        paddingLength = (rangeLen * endIndex) + endIndex - fileLen
        power = np.lib.pad(power, ((0, paddingLength), (0, 0)), 'constant', constant_values = 0)
        zcr = np.lib.pad(zcr, (0, paddingLength), 'constant', constant_values = 0)
        entropy = np.lib.pad(entropy, (0, paddingLength), 'constant', constant_values = 0)
    
    fileLen = len(power)

    while (endIndex != fileLen):
        # Taking only rows from startIndex to endIndex
        tempPower = power[startIndex:endIndex, ]
        tempZcr = zcr[startIndex:endIndex]
        tempEntropy = entropy[startIndex:endIndex]
        print(np.mean(tempPower), np.mean(tempZcr), np.mean(tempEntropy))
        startIndex += jump
        endIndex += jump

def main():

    
    allFiles = np.dstack([powerFiles, entropyFiles, zcrFiles])

    for File in allFiles:
        for f in File:
            powerFile = f[0]
            entropyFile = f[1]
            zcrFile = f[2]
            print powerFile, zcrFile, entropyFile
            splitSignal(powerFile, zcrFile, entropyFile)
if __name__ == '__main__':
    main()
