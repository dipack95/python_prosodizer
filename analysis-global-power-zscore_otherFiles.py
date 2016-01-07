import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

targetPath = os.path.abspath("../csv/Training_Data")

mfccFiles = [os.path.join(path, name)
             for path, dirs, files in os.walk(targetPath)
             for name in files if name.endswith(("-mfcc.csv"))]

del1Files = [os.path.join(path, name)
             for path, dirs, files in os.walk(targetPath)
             for name in files if name.endswith(("-del1.csv"))]

del2Files = [os.path.join(path, name)
             for path, dirs, files in os.walk(targetPath)
             for name in files if name.endswith(("-del2.csv"))]

entropyFiles = [os.path.join(path, name)
                for path, dirs, files in os.walk(targetPath)
                for name in files if name.endswith(("-entropy.csv"))]

rmsFiles = [os.path.join(path, name)
            for path, dirs, files in os.walk(targetPath)
            for name in files if name.endswith(("-rms.csv"))]

zeroCrossingFiles = [os.path.join(path, name)
                     for path, dirs, files in os.walk(targetPath)
                     for name in files if name.endswith(("-zeroCrossing.csv"))]

powerFiles = [os.path.join(path, name)
              for path, dirs, files in os.walk(targetPath)
              for name in files if name.endswith(("-powerSpectrum.csv"))]


def customZScore(data, mean, standardDeviation):
    zs = []
    for temp in data:
        zs.append((temp - mean) / standardDeviation)

    return zs

def powerAndEntropy(powerFile, entropyFile):
    # print("Power file: " + str(powerFile.split("/")[-1]) + " Entropy file: " + str(entropyFile.split("/")[-1]))
    
    power = np.nan_to_num(np.array(pd.read_csv(powerFile, header=None), dtype='float64'))
    entropy = np.nan_to_num(np.array(pd.read_csv(entropyFile, header=None), dtype='float64'))

    if power.shape[0] != entropy.shape[1]:
        print("Number of frames in power array is not equal to number of entropy values in entropy array.")
        return

    avgPowerOfFrames = []
    
    for frame in power:
        sum = 0
        for tempVal in frame:
            sum += tempVal

        sum /= power.shape[1]
        avgPowerOfFrames.append(sum)

    avgPowerOfFrames = np.array(avgPowerOfFrames)
    entropy = np.reshape(entropy, entropy.shape[1])

    # print("avgPowerOfFrames: ", str(avgPowerOfFrames.shape), " entropy: ", str(entropy.shape))

    return avgPowerOfFrames, entropy

def averagePowerOfFrames(powerFile):    
    power = np.nan_to_num(np.array(pd.read_csv(powerFile, header=None), dtype='float64'))

    avgPowerOfFrames = []
    
    for frame in power:
        sum = 0
        for tempVal in frame:
            sum += tempVal

        sum /= power.shape[1]
        avgPowerOfFrames.append(sum)

    avgPowerOfFrames = np.array(avgPowerOfFrames)

    return avgPowerOfFrames

def totalPowerZScoreCalc(avgPowerAll, startIndexForFiles):
    normalCount = 0
    angryCount = 0
    powerFileMentions = {}
    totalZS = stats.zscore(avgPowerAll)

    keys = startIndexForFiles.keys()
    startIndexForFilesArray = np.array(list(startIndexForFiles.values()))

    for i in range(0, len(totalZS)):
        if totalZS[i] > 3:
            targetIndex = min(range(len(startIndexForFilesArray)), key=lambda x: abs(startIndexForFilesArray[x] - i))
            filename = startIndexForFilesArray[targetIndex]
            
            for k, v in startIndexForFiles.items():
                if v == filename:
                    filename = k
            
            if filename in powerFileMentions:
                powerFileMentions[filename] += 1
            else:
                powerFileMentions[filename] = 1

    for k, v in powerFileMentions.items():
        if ("normal" in k) or ("neutral" in k):
            normalCount += 1
        elif("angry" in k):
            angryCount += 1

        print(k.split('/')[-1], v)

    return totalZS, powerFileMentions, angryCount, normalCount


def splitSignal(powerFile):
    # startIndex = 
            
    # endIndex = 30

    # for i in range(0, np.int(np.floor( (len(avgPowerOfFrames) - 30) / 15) ) ):
    #     startIndex += 15
    #     endIndex += 15

    power = np.nan_to_num(np.array(pd.read_csv(powerFile, header=None), dtype='float64'))
    divFrames = []

    jump = 50
    divFrameLength = 200
    startIndex = 0
    endIndex = divFrameLength
    fileLen = len(power)
    rangeLen = fileLen // endIndex
    
    if fileLen > rangeLen:
        paddingLength = (rangeLen * endIndex) + endIndex - fileLen
        power = np.lib.pad(power, ((0, paddingLength), (0, 0)), 'constant', constant_values = 0)
    
    fileLen = len(power)

    while (endIndex != fileLen):
        # Taking only rows from startIndex to endIndex
        tempPower = power[startIndex:endIndex, ]
        avgPower = np.mean(tempPower)
        divFrames = np.append(divFrames, avgPower)
        print(startIndex * 10 / 1000, endIndex * 10 / 1000)
        startIndex += jump
        endIndex += jump

    return divFrames

def meanForEntireFile(filename):
    meanForFile = np.nan_to_num(np.array(pd.read_csv(filename, header=None), dtype='float64'))
    return np.mean(meanForFile)

names = ['training', 'OldGuy', 'YoungWoman']
# names = ['OldGuy']

def main():
    emotions = ["normal", "angry"]

    numberOfPlots = len(emotions)

    # powerEntropyPlot = plt.figure()
    # powerEntropyPlot.suptitle('Power vs Entropy')

    powerDivFrameFiles = []
    allAvgPower = {}
    avgPowerAll = []
    minDivFrames = []
    startIndexForFiles = {}

    meanPowerForNormalFiles = []
    meanPowerForAngryFiles = []

    meanEntropyForNormalFiles = []
    meanEntropyForAngryFiles = []

    meanNormal = 0
    meanAngry = 0

    count = 0
    totalCount = 0

    startPowerIndex = 0

    targetFiles = []
    
    # for name in names:
    #     for emotion in emotions:
    #         targetFiles = []

    #         for filename in powerFiles:
    #             if name in filename:
    #                 if emotion == "normal":
    #                     if ("normal" in filename) or ("neutral" in filename):
    #                         targetFiles.append(filename)
    #                 elif emotion in filename:
    #                     targetFiles.append(filename)

    #         for filename in entropyFiles:
    #             if name in filename:
    #                 if emotion == "normal":
    #                     if ("normal" in filename) or ("neutral" in filename):
    #                         targetFiles.append(filename)
    #                         meanEntropyForNormalFiles = np.append(meanEntropyForNormalFiles, meanForEntireFile(filename))
    #                 elif emotion in filename:
    #                     targetFiles.append(filename)
    #                     meanEntropyForAngryFiles = np.append(meanEntropyForAngryFiles, meanForEntireFile(filename))

    #         # avgPowerOfFrames, entropyOfFrames = powerAndEntropy(targetFiles[0], targetFiles[1])         

    #         # actualPlot = powerEntropyPlot.add_subplot((numberOfPlots * 100) + 10 + emotions.index(emotion))
    #         # actualPlot.set_title(targetFiles[0].split('/')[-1].split('.wav')[0])
    #         # actualPlot.plot(np.abs(avgPowerOfFrames), np.abs(entropyOfFrames), 'o')

    #         # if ("angry" in targetFiles[0]):
    #         #     meanPowerForAngryFiles = np.append(meanPowerForAngryFiles, meanForEntireFile(targetFiles[0]))
    #         # elif("normal" in targetFiles[0]) or ("neutral" in targetFiles[0]):
    #         #     meanPowerForNormalFiles = np.append(meanPowerForNormalFiles, targetFiles[0])

    # print(targetFiles)
    
    # globalPowerPlot = plt.figure()
    # globalPowerPlot.suptitle('Global Power Plot')
    # gPlot = globalPowerPlot.add_subplot(111)
    # gPlot.set_title('All files plot')
    # gPlot.plot(meanEntropyForAngryFiles, 'o', c='r')
    # gPlot.plot(meanEntropyForNormalFiles, 'o', c='b')

    # targetFiles = ['/home/dipack/College/Fourth_Year/Final_Year_Project/csv/Training_Data/angry/training_angry_3.wav-powerSpectrum.csv', '/home/dipack/College/Fourth_Year/Final_Year_Project/csv/Training_Data/angry/training_angry_2.wav-entropy.csv']

    # avgPowerOfFrames, entropyOfFrames = powerAndEntropy(targetFiles[0], targetFiles[1])

    localFileName = '/home/dipack/College/Fourth_Year/Final_Year_Project/csv/Training_Data/neutral/training_neutral_3.wav-powerSpectrum.csv'
    localFilePlot = plt.figure()
    localFilePlot.suptitle('Local Frame Plot')
    lPlot = localFilePlot.add_subplot(111)
    lPlot.set_title(localFileName.split('/')[-1].split('.wav')[0])
    lPlot.plot(splitSignal(localFileName), '-o')

    plt.show()

if __name__ == '__main__':
    main()
