import pandas as pd
import numpy as np
from scipy.stats import f
import matplotlib.pyplot as plt
import os

# np.set_printoptions(threshold=np.inf)

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


# def splitSignal(localFile):

#     power = np.nan_to_num(np.array(pd.read_csv(localFile + '-powerSpectrum.csv', header=None), dtype='float64'))
#     zcr = np.nan_to_num(np.array(pd.read_csv(localFile + '-zeroCrossing.csv', header=None), dtype='float64'))
#     entropy = np.nan_to_num(np.array(pd.read_csv(localFile + '-entropy.csv', header=None), dtype='float64'))
#     mfcc = np.nan_to_num(np.array(pd.read_csv(localFile + '-mfcc.csv', header=None), dtype='float64'))
    
#     zcr = np.ravel(zcr)
#     entropy = np.ravel(entropy)
    
#     divFrames = []

#     # jump = 15
#     # divFrameLength = 66.67
#     jump = 25
#     divFrameLength = 100
#     startIndex = 0
#     endIndex = divFrameLength
#     fileLen = len(power)
#     rangeLen = np.ceil(fileLen / endIndex)

#     if fileLen > rangeLen:
#         paddingLength = (rangeLen * endIndex) + endIndex - fileLen
#         power = np.lib.pad(power, ((0, paddingLength), (0, 0)), 'constant', constant_values = 0)
#         zcr = np.lib.pad(zcr, (0, paddingLength), 'constant', constant_values = 0)
#         entropy = np.lib.pad(entropy, (0, paddingLength), 'constant', constant_values = 0)
#         mfcc = np.lib.pad(mfcc, ((0, paddingLength), (0, 0)), 'constant', constant_values = 0)
    
#     fileLen = len(power)

#     while (endIndex <= (((fileLen * 15) - 1) / 10)):
#         # Taking only rows from startIndex to endIndex
#         tempPower = power[startIndex:endIndex, ]
#         tempZcr = zcr[startIndex:endIndex]
#         tempEntropy = entropy[startIndex:endIndex]
#         tempMfcc = mfcc[startIndex:endIndex, ]
#         print(startIndex * 0.01, "->", endIndex * 0.01, np.mean(tempPower), np.mean(tempEntropy), np.mean(tempMfcc))
#         startIndex += jump
#         endIndex += jump

def splitSignal(localFile):

    power = np.nan_to_num(np.array(pd.read_csv(localFile + '-powerSpectrum.csv', header=None), dtype='float64'))
    
    blockLength = 100.0
    startIndex = 0.0
    endIndex = blockLength
    durationOffset = 25.0
    frameStart = 0
    frameEnd = blockLength
    frameOffset = 17.0

    avgPower = np.mean(power, axis = 1)

    lengthOfFile = (((power.shape[0] * power.shape[1]) * 0.600732601) // 10000)
    numOfJumps = (lengthOfFile * 100 / durationOffset)
    for i in range(0, int(numOfJumps)):
        tempPower = np.mean(avgPower[frameStart:frameEnd])
        if not (np.isnan(tempPower)):
            print(startIndex * 0.01, endIndex * 0.01, tempPower)
            # print(avgPower[frameStart:frameEnd])
            startIndex += durationOffset
            endIndex += durationOffset
            frameStart += frameOffset
            frameEnd += frameOffset
        else:
            break
    # print(avgPower)

def main():

    
    # allFiles = np.dstack([powerFiles, entropyFiles, zcrFiles])

    # for File in allFiles:
    #     for f in File:
    #         powerFile = f[0]
    #         entropyFile = f[1]
    #         zcrFile = f[2]
    #         print powerFile, zcrFile, entropyFile
    #         splitSignal(powerFile, zcrFile, entropyFile)
    localFileName = '/home/dipack/College/Fourth_Year/Final_Year_Project/csv/training_dataset/angry/training_angry_9.wav'
    print(localFileName.split('/')[-1].split('.wav')[0])
    splitSignal(localFileName)

if __name__ == '__main__':
    main()
