import pandas as pd
import numpy as np
import os

targetPath = os.path.abspath("../csv")

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

def computeGlobalValuesForPower(names, emotions, fileSet):

    targetFiles = []
    targetValues = []

    for filename in fileSet:
        for name in names:
            for emotion in emotions:
                if name in filename:
                    if emotion == "normal":
                        if ("normal" in filename) or ("neutral" in filename):
                            targetFiles.append(filename)
                    elif emotion in filename:
                        targetFiles.append(filename)

    print(len(targetFiles))

    for filename in targetFiles:
        power = np.nan_to_num(np.array(pd.read_csv(filename, header=None), dtype='float64'))

        avgPowerOfFrames = []
        
        for frame in power:
            sum = 0
            for tempVal in frame:
                sum += tempVal

            sum /= power.shape[1]
            avgPowerOfFrames.append(sum)

        avgPowerOfFrames = np.array(avgPowerOfFrames)
        for tempAvg in avgPowerOfFrames:
            targetValues.append(tempAvg)

    mean = np.mean(targetValues)
    standardDeviation = np.std(targetValues)
    variance = np.var(targetValues)

    return mean, standardDeviation, variance

def computeGlobalValuesForEntropy(names, emotions, fileSet):

    targetFiles = []
    targetValues = []

    for filename in fileSet:
        for name in names:
            for emotion in emotions:
                if name in filename:
                    if emotion == "normal":
                        if ("normal" in filename) or ("neutral" in filename):
                            targetFiles.append(filename)
                    elif emotion in filename:
                        targetFiles.append(filename)

    for filename in targetFiles:
        entropy = np.nan_to_num(np.array(pd.read_csv(filename, header=None), dtype='float64'))

        entropy = np.array(np.reshape(entropy, entropy.shape[1]))
        
        for temp in entropy:
            targetValues.append(temp)

    mean = np.mean(targetValues)
    standardDeviation = np.std(targetValues)
    variance = np.var(targetValues)

    return mean, standardDeviation, variance

globalMean = 0
globalStandardDeviation = 0
globalVar = 0

names = ['Ajinkya','Ari','Arrow','Arun','Bhargav','Cruise','DadPatil','DC','Dipack','Doctor','Goswami','Harsh','Harvey','JE','JK','Simmons','Kapish','KL','Leonardo','Liam','Loki','Louis','Malkan','Manas','Mathur','MichaelCera','Nicholson','Nishant','OldGuy','Puneet','RandomMan','Rishi','Robert','Rohit','Sad','Tobymac','Tushar','Tyrion','Vaas','Akansha','Arzoo','Emily','Isha','Lorelai','MomPatil','Olivia','Pallavi','Paris','PDB','Pooja','Pritha','Sayalee','Tanvi','Unsorted','YoungWoman']

def main():

    emotions = ["normal", "angry"]

    globalMean, globalStandardDeviation, globalVar = computeGlobalValuesForPower(names, emotions, powerFiles)
    
    print(globalMean)
    print(globalStandardDeviation)
    print(globalVar)

if __name__ == '__main__':
    main()