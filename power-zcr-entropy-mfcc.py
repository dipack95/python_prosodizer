import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os

from scipy.stats import f
from scipy import stats

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

def splitSignal(localFile):

    power = np.nan_to_num(np.array(pd.read_csv(localFile + '-powerSpectrum.csv', header=None), dtype='float64'))
    mfcc = np.nan_to_num(np.array(pd.read_csv(localFile + '-mfcc.csv', header=None), dtype='float64'))
    zeroCrossing = np.nan_to_num(np.array(pd.read_csv(localFile + '-zeroCrossing.csv', header=None), dtype='float64'))
    entropy = np.nan_to_num(np.array(pd.read_csv(localFile + '-entropy.csv', header=None), dtype='float64'))
    zeroCrossing = np.ravel(zeroCrossing)
    entropy = np.ravel(entropy)

    numOfFrames = power.shape[0]

    signalFileName = localFile.replace("/csv/", "/sounds/")
    (rate, signal) = wav.read(signalFileName)
    # In Seconds
    lengthOfSignal = len(signal) / rate  

    # In Milliseconds
    frameLength = 25.0
    overlapLength = 10.0
    blockDuration = 1000.0
    blockOverlap = 250.0
    numOfJumps = np.ceil((lengthOfSignal * 1000.0) / blockOverlap)

    blockLength = blockDuration / overlapLength 
    startIndex = 0.0
    endIndex = blockLength
    
    durationOffset = frameLength
    frameStart = 0
    frameEnd = blockLength
    frameOffset = np.floor(numOfFrames / numOfJumps)

    print(frameOffset, blockLength, numOfJumps, lengthOfSignal, numOfFrames)

    avgPower = np.mean(power, axis = 1)

    print("Filename:", localFile.split('/')[-1].split('.wav')[0], "Length:", lengthOfSignal)

    for i in range(0, int(numOfJumps)):
        
        tempPower = np.mean(avgPower[frameStart:frameEnd])
        tempMfcc = mfcc[frameStart:frameEnd, ]
        tempZcr = np.mean(zeroCrossing[frameStart:frameEnd])
        tempEntropy = np.mean(entropy[frameStart:frameEnd])

        ratio = []
        for t in tempMfcc:
            a, b, c, d, e, f = t[0], t[1], t[2], t[10], t[11], t[12]
            av1 = sum(t[0:3]) / 3
            av2 = sum(t[10:13]) / 3
            ratio = np.append(ratio, av1 / av2)
            
        intraAverage = np.nan_to_num(np.array(ratio))
        tempMfcc = np.mean(intraAverage)

        zsPower = stats.zscore(avgPower[frameStart:frameEnd])
        zsZcr = stats.zscore(zeroCrossing[frameStart:frameEnd])
        zsEntropy = stats.zscore(entropy[frameStart:frameEnd])
        zsMfcc = stats.zscore(intraAverage)

        if not (np.isnan(tempPower)):
            # Use the following print statement to print times, and labels
            print(startIndex * 0.01, "->", endIndex * 0.01, "Power:", tempPower, "Mfcc:", tempMfcc, "Entropy:", tempEntropy, "Zcr:", tempZcr)
            # Use the following print statement to just print the values
            # print(tempPower, round(tempMfcc, 7), tempEntropy)
            # print(startIndex * 0.01, endIndex * 0.01, frameStart, frameEnd)
            # print(avgPower[frameStart:frameEnd])
            startIndex += durationOffset
            endIndex += durationOffset
            frameStart += frameOffset
            frameEnd += frameOffset
        else:
            break
    # print(avgPower)

def main():

    localFileName = '/home/dipack/College/Fourth_Year/Final_Year_Project/csv/training_dataset/angry/training_angry_1.wav'
    # localFileName = '/home/dipack/College/Fourth_Year/Final_Year_Project/csv/Women/Pallavi/pallavi-normal.wav'
    splitSignal(localFileName)

if __name__ == '__main__':
    main()
