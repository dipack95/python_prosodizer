import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

'''
        dfDEL1 = np.array(pd.read_csv(filename + "-del1.csv", sep=',',header=None), dtype = 'float64' )
        dfDEL2 = np.array(pd.read_csv(filename + "-del2.csv", sep=',',header=None), dtype = 'float64' )
        dfENT = np.array(pd.read_csv(filename + "-entropy.csv", sep=',',header=None), dtype = 'float64' )
        dfRMS = np.array(pd.read_csv(filename + "-rms.csv", sep=',',header=None), dtype = 'float64' )
'''

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

def computeGlobalValuesForPower(names, emotions, fileSet):

    targetFiles = []
    targetValues = []

    # for person in names:
    #     for emotion in emotions:
    #         for filename in fileSet:
    #             if person in filename:
    #                 if emotion == "normal":
    #                     if ("normal" in filename) or ("neutral" in filename):
    #                         targetFiles.append(filename)
    #                 elif emotion in filename:
    #                     targetFiles.append(filename)

    for filename in fileSet:
        if "angry" in filename or "anger" in filename or "Angry" in filename or "normal" in filename or "neutral" in filename or "Normal" in filename or "calm" in filename:
            for name in names:
                if name in filename:
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

    # for person in names:
    #     for emotion in emotions:
    #         for filename in fileSet:
    #             if person in filename:
    #                 if emotion == "normal":
    #                     if ("normal" in filename) or ("neutral" in filename):
    #                         targetFiles.append(filename)
    #                 elif emotion in filename:
    #                     targetFiles.append(filename)

    for filename in fileSet:
        if "angry" in filename or "anger" in filename or "Angry" in filename or "normal" in filename or "neutral" in filename or "Normal" in filename or "calm" in filename:
            for name in names:
                if name in filename:
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

def customZScore(data, mean, standardDeviation):
    zs = []
    for temp in data:
        zs.append((temp - mean) / standardDeviation)

    return zs

# names = ['Ajinkya','Ari','Arrow','Arun','Bhargav','Cruise','DadPatil','DC','Dipack','Doctor','Goswami','Harsh','Harvey','JE','JK','Simmons','Kapish','KL','Leonardo','Liam','Loki','Louis','Malkan','Manas','Mathur','MichaelCera','Nicholson','Nishant','OldGuy','Puneet','RandomMan','Rishi','Robert','Rohit','Sad','Tobymac','Tushar','Tyrion','Vaas','Akansha','Arzoo','Emily','Isha','Lorelai','MomPatil','Olivia','Pallavi','Paris','PDB','Pooja','Pritha','Sayalee','Tanvi','Unsorted','YoungWoman']
names = ['Pallavi']

def main():

    emotions = ["normal", "angry"]

    numberOfPlots = len(emotions)
    normal = 0
    angry = 0
    
    powerEntropyPlot = plt.figure()
    powerEntropyPlot.suptitle('Power vs Entropy')

    globalMean, globalStandardDeviation, globalVar = computeGlobalValuesForPower(names, emotions, powerFiles)

    globalMean = 0.333817846598
    globalStandardDeviation = 2.57808951914
    globalVar = 6.64654556871
    
    print(globalMean)
    print(globalStandardDeviation)
    print(globalVar)


    for person in names:
        result1 = 0
        result2 = 0
        for emotion in emotions:
            targetFiles = []
            outlierCount = 0

            for filename in powerFiles:
                if person in filename:
                    if emotion == "normal":
                        if ("normal" in filename) or ("neutral" in filename):
                            targetFiles.append(filename)
                    elif emotion in filename:
                        targetFiles.append(filename)

            for filename in entropyFiles:
                if person in filename:
                    if emotion == "normal":
                        if ("normal" in filename) or ("neutral" in filename):
                            targetFiles.append(filename)
                    elif emotion in filename:
                        targetFiles.append(filename)

            print(targetFiles)

            if(len(targetFiles) < 2):
                print(person, "does not have sufficient number of data files!")
                continue
            elif(len(targetFiles) > 2):
                print(person, "has too many similarly named data files!")
                continue

            avgPowerOfFrames, entropy = powerAndEntropy(targetFiles[0], targetFiles[1])

            entropy = np.array(np.reshape(entropy, entropy.shape[0]))

            # zs = stats.zscore(avgPowerOfFrames)
            zs = np.array(customZScore(avgPowerOfFrames, globalMean, globalStandardDeviation))
            # print(zs)
            # print(len(zs))

            combinedArray = np.transpose(np.array((avgPowerOfFrames, entropy)))

            variance = np.var(combinedArray)

            for zsOfFrame in zs:
                if(zsOfFrame > 1):
                    outlierCount += 1
            
            print(person, " - ", emotion)
            print("Variance: ", str(variance))
            print("Z Score Outliers: ", str(outlierCount))

            if(result1 == 0): result1 = outlierCount
            else: result2 = outlierCount 

            # startIndex = 
            
            # endIndex = 30

            # for i in range(0, np.int(np.floor( (len(avgPowerOfFrames) - 30) / 15) ) ):
            #     startIndex += 15
            #     endIndex += 15

            # actualPlot = powerEntropyPlot.add_subplot((numberOfPlots * 100) + 10 + emotions.index(emotion))
            # actualPlot.set_title(emotion)
            # actualPlot.plot(np.abs(avgPowerOfFrames), np.abs(entropy), 'o')

        #plt.show()
        if(result1 != 0) and (result2 != 0): 
            if(result1 > result2):
                print("Normal is higher for", person)
                normal += 1
            elif(result2 > result1):
                print("Angry is higher for", person)
                angry += 1
            else:
                print("\nBoth are equal for", person)
        resul1 = result2 = 0
        print("Normal: ", str(normal), " Angry", str(angry))
        print("\n")

if __name__ == '__main__':
    main()