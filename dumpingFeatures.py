import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os
import sys

from scipy.stats import f
from scipy import stats

np.set_printoptions(threshold=np.inf)

allFilesWomen = [os.path.join(path, name)
             for path, dirs, files in os.walk("../sounds/Women/")
             for name in files if (".wav" in name)]
allFilesMen = [os.path.join(path, name)
             for path, dirs, files in os.walk("../sounds/Men/")
             for name in files if (".wav" in name)]

def splitSignal(localFile):

    if "anger" in localFile or "angry" in localFile:
        label = "Angry"
    elif "normal" in localFile or "neutral" in localFile:
        label = "Neutral"
    else:
        return

    # print(localFile)

    power = np.nan_to_num(np.array(pd.read_csv(localFile + '_powerSpectrum.csv', header=None), dtype='float64'))
    mfcc = np.nan_to_num(np.array(pd.read_csv(localFile + '_mfcc.csv', header=None), dtype='float64'))
    entropy = np.nan_to_num(np.array(pd.read_csv(localFile + '_entropy.csv', header=None), dtype='float64'))
    energy = np.nan_to_num(np.array(pd.read_csv(localFile + '_energyOfFrames.csv', header=None), dtype='float64'))
    entropy = np.ravel(entropy)
    energy = np.log10(np.ravel(energy))

    av = np.mean(mfcc, axis = 0)

    for m in mfcc:
      for i in range(len(m)):
        m[i] = ( m[i] - av[i] )

    avpow = np.mean(power, axis = 0)

    for p in power:
      for i in range(len(p)):
        p[i] = ( p[i] - avpow[i] )

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

    firstMfccs = mfcc[:,0]
    tenMfccs = mfcc[:, 0:10]
    avgPower = np.mean(power, axis = 1)
    vMfcc = np.var(mfcc, axis = 1)

    for i in range(0, int(numOfJumps)):
        
        tempPower = np.mean(avgPower[frameStart:frameEnd])
        tempFirstMfcc = np.mean(firstMfccs[frameStart:frameEnd])
        varMfcc = np.mean(vMfcc[frameStart:frameEnd])
        tempEntropy = np.mean(entropy[frameStart:frameEnd])
        tempEnergy = np.mean(energy[frameStart:frameEnd])
        tempTenMfcc = np.mean(tenMfccs[frameStart:frameEnd], axis = 0)

        if not (np.isnan(tempPower)):
            print(tempTenMfcc)
            # print(startIndex * 0.01, endIndex * 0.01, varMfcc, tempFirstMfcc,tempPower, tempEntropy, tempEnergy)
            # print(varMfcc, ",", tempFirstMfcc, ", ", tempPower, ", ", tempEntropy, ",", tempEnergy)
            startIndex += durationOffset
            endIndex += durationOffset
            frameStart += frameOffset
            frameEnd += frameOffset
        else:
            break

def main():
    # print(np.append(allFilesMen, allFilesWomen))
    # localFileName = sys.argv[1]
    localFileName = "../csv/Men/Leo_Wolf/leo_angry_wolf.wav"
    print(localFileName.split('/')[-1].split('.wav')[0])
    print("Variance MFCC, First MFCC, Mean Power, Mean entropy, Mean Energy")
    # for root, directories, filenames in os.walk('../sounds/'):
    #     for filename in filenames: 
    #         if ".wav" in filename and ".csv" not in filename:
    #             localFileName = os.path.join(root,filename) 
    #             splitSignal(localFileName)
    # localFileName = "../csv/Women/Lorelai/lorelai_angry.wav"
    splitSignal(localFileName)

if __name__ == '__main__':
    main()
