import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

targetPath = os.path.abspath("../csv")

np.set_printoptions(threshold=np.inf)

mfccFiles = [os.path.join(path, name)
             for path, dirs, files in os.walk(targetPath)
             for name in files if name.endswith(("_mfcc.csv"))]

del1Files = [os.path.join(path, name)
             for path, dirs, files in os.walk(targetPath)
             for name in files if name.endswith(("_del1.csv"))]

del2Files = [os.path.join(path, name)
             for path, dirs, files in os.walk(targetPath)
             for name in files if name.endswith(("_del2.csv"))]

entropyFiles = [os.path.join(path, name)
                for path, dirs, files in os.walk(targetPath)
                for name in files if name.endswith(("_entropy.csv"))]

rmsFiles = [os.path.join(path, name)
            for path, dirs, files in os.walk(targetPath)
            for name in files if name.endswith(("_rms.csv"))]

zeroCrossingFiles = [os.path.join(path, name)
                     for path, dirs, files in os.walk(targetPath)
                     for name in files if name.endswith(("_zeroCrossing.csv"))]

powerFiles = [os.path.join(path, name)
              for path, dirs, files in os.walk(targetPath)
              for name in files if name.endswith(("_powerSpectrum.csv"))]

def printTimes(startIndex, endIndex, jump, localFile):
    mfcc = np.nan_to_num(np.array(pd.read_csv(localFile + '_mfcc.csv', header=None), dtype='float64'))
    power = np.nan_to_num(np.array(pd.read_csv(localFile + '_powerSpectrum.csv', header=None), dtype='float64'))
    count = 0

    for i in range(len(mfcc)):
        print(startIndex, "ms ->", endIndex, "ms Mean Mfcc:", np.mean(mfcc[i]), "Mean Power:", np.mean(power[i]))
        startIndex += jump
        endIndex += jump
        count += 1

    return

def powerEntropyMfccZC(localFile):
    # print("Power file: " + str(powerFile.split("/")[-1]) + " Entropy file: " + str(entropyFile.split("/")[-1]))
    
    power = np.nan_to_num(np.array(pd.read_csv(localFile + '_powerSpectrum.csv', header=None), dtype='float64'))
    entropy = np.nan_to_num(np.array(pd.read_csv(localFile + '_entropy.csv', header=None), dtype='float64'))
    mfcc = np.nan_to_num(np.array(pd.read_csv(localFile + '_mfcc.csv', header=None), dtype='float64'))
    zeroCrossing = np.nan_to_num(np.array(pd.read_csv(localFile + '_zeroCrossing.csv', header=None), dtype='float64'))

    if power.shape[0] != entropy.shape[1]:
        print("Number of frames in power array is not equal to number of entropy values in entropy array.")
        return

    if (power.shape[0] != mfcc.shape[0]):
        print("Number of frames in power array is not equal to number of frames in mfcc array.")
        return

    avgPowerOfFrames = []
    avgPowerOfFrames = np.array(np.append(avgPowerOfFrames, np.mean(power, axis=1)))

    entropy = np.reshape(entropy, entropy.shape[1])

    ratioOfMfccs = []
    for frame in mfcc:
        avg1 = sum(frame[0:3]) / 3
        avg2 = sum(frame[10:13]) / 3
        ratioOfMfccs = np.append(ratioOfMfccs, avg1 / avg2)
    ratioOfMfccs = np.nan_to_num(ratioOfMfccs)

    zeroCrossing = np.reshape(zeroCrossing, zeroCrossing.shape[1])

    # print("avgPowerOfFrames: ", str(avgPowerOfFrames.shape), " entropy: ", str(entropy.shape))

    return avgPowerOfFrames, entropy, ratioOfMfccs, zeroCrossing

def main():
    localFileName = '/home/dipack/College/Fourth_Year/Final_Year_Project/csv/Men/Ajinkya/Ajinkya_angry.wav'
    # localFileName = '/home/dipack/College/Fourth_Year/Final_Year_Project/csv/Women/Pallavi/pallavi-normal.wav'
    # localFileName = '/home/dipack/College/Fourth_Year/Final_Year_Project/csv/Men/JE/je_angry.wav
    print(localFileName.split('/')[-1].split('.wav')[0])
    printTimes(0, 25, 15, localFileName)
    # avgPowerOfFrames, entropy, ratioOfMfccs, zeroCrossing = powerEntropyMfccZC(localFileName)
    # zsPower = stats.zscore(avgPowerOfFrames)
    # zsMfcc = stats.zscore(ratioOfMfccs)
    # zsEntropy = stats.zscore(entropy)
    # zsZC = stats.zscore(zeroCrossing)

    # print(localFileName.split('/')[-1].split('.wav')[0])
    # startIndex = 0
    # endIndex = 25
    # jump = 15

    # for i in range(len(ratioOfMfccs)):
    #     print(startIndex, "ms ->", endIndex, "ms -- MFCC:", ratioOfMfccs[i], zsMfcc[i], "-- Power:", avgPowerOfFrames[i], zsPower[i], "-- Entropy:", entropy[i], zsEntropy[i],"-- ZeroC:", zeroCrossing[i], zsZC[i])
    #     startIndex += jump
    #     endIndex += jump

if __name__ == '__main__':
    main()
