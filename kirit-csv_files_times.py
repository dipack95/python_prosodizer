import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing

targetPath = os.path.abspath("../csv")

# np.set_printoptions(threshold=np.inf)

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

def scaleData(data):
    minMax = preprocessing.MinMaxScaler(feature_range = (0, 1000))
    normalizedData = minMax.fit_transform(data)
    return normalizedData

def printTimes(startIndex, endIndex, jump, localFile):
    mfcc = np.nan_to_num(np.array(pd.read_csv(localFile + '-mfcc.csv', header=None), dtype='float64'))
    power = np.nan_to_num(np.array(pd.read_csv(localFile + '-powerSpectrum.csv', header=None), dtype='float64'))
    count = 0

    for i in range(len(mfcc)):
        print(startIndex, "ms ->", endIndex, "ms Mean Mfcc:", np.mean(mfcc[i]), "Var Mfcc:", np.var(mfcc[i]), "first Mfcc:", mfcc[i][0])
        startIndex += jump
        endIndex += jump
        count += 1

    return

def powerEntropyMfccZC(localFile):
    # print("Power file: " + str(powerFile.split("/")[-1]) + " Entropy file: " + str(entropyFile.split("/")[-1]))
    
    power = np.nan_to_num(np.array(pd.read_csv(localFile + '-powerSpectrum.csv', header=None), dtype='float64'))
    entropy = np.nan_to_num(np.array(pd.read_csv(localFile + '-entropy.csv', header=None), dtype='float64'))
    mfcc = np.nan_to_num(np.array(pd.read_csv(localFile + '-mfcc.csv', header=None), dtype='float64'))
    zeroCrossing = np.nan_to_num(np.array(pd.read_csv(localFile + '-zeroCrossing.csv', header=None), dtype='float64'))

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

    zeroCrossing = np.reshape(zeroCrossing, zeroCrossing.shape[1])

    # print("avgPowerOfFrames: ", str(avgPowerOfFrames.shape), " entropy: ", str(entropy.shape))

    return avgPowerOfFrames, entropy, ratioOfMfccs, zeroCrossing

def main():
    localFileName = '/home/dipack/College/Fourth_Year/Final_Year_Project/sounds/Men/JK/jk_angry.wav'
    # localFileName = '/home/dipack/College/Fourth_Year/Final_Year_Project/csv/training_dataset/angry/training_angry_10.wav'
    print(localFileName.split('/')[-1].split('.wav')[0])
    printTimes(0, 25, 15, localFileName)
    
if __name__ == '__main__':
    main()