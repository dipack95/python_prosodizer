import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os

from scipy.stats import f
from scipy import stats

# np.set_printoptions(threshold=np.inf)

# targetPath = os.path.abspath("../sounds/sounds/hack-the-talk-exotel-master/training_dataset")

def splitSignal(localFile):

    power = np.nan_to_num(np.array(pd.read_csv(localFile + '_powerSpectrum.csv', header=None), dtype='float64'))
    mfcc = np.nan_to_num(np.array(pd.read_csv(localFile + '_mfcc.csv', header=None), dtype='float64'))
    zeroCrossing = np.nan_to_num(np.array(pd.read_csv(localFile + '_zeroCrossing.csv', header=None), dtype='float64'))
    entropy = np.nan_to_num(np.array(pd.read_csv(localFile + '_entropy.csv', header=None), dtype='float64'))
    labels = np.array(pd.read_csv(localFile + '_labels.csv', header=None))
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

    if numOfJumps != labels.shape[0]:
        print("Number of labels and number of jumps to make are not equal: Labels:", labels.shape[0], "Jumps:", numOfJumps)
        return

    blockLength = blockDuration / overlapLength 
    startIndex = 0.0
    endIndex = blockLength
    
    durationOffset = frameLength
    frameStart = 0
    frameEnd = blockLength
    frameOffset = np.floor(numOfFrames / numOfJumps)

    print(frameOffset, blockLength, numOfJumps, lengthOfSignal, numOfFrames)

    print("Filename:", localFile.split('/')[-1].split('.wav')[0], "Length:", lengthOfSignal, "Number of labels:", labels.shape, "Number of jumps:", numOfJumps)
    for i in range(0, int(numOfJumps)):
        print(startIndex * 0.01, "->", endIndex * 0.01, frameStart, frameEnd)
        # Prints the first MFCC of each frame in the block, and prints its block label
        for j in range(np.int(frameStart), np.int(frameEnd)):
            print(j * 0.015, mfcc[j][0], labels[i])
        startIndex += durationOffset
        endIndex += durationOffset
        frameStart += frameOffset
        frameEnd += frameOffset
    return

def main():
    localFileName = "/home/dipack/College/Fourth_Year/Final_Year_Project/csv/Men/Ajinkya/Ajinkya_angry.wav"
    splitSignal(localFileName)

if __name__ == '__main__':
    main()