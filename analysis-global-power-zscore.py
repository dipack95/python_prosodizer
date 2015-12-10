import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
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

    jump = 25
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
        print(startIndex, endIndex)
        tempPower = power[startIndex:endIndex, ]
        avgPower = np.mean(tempPower)
        divFrames = np.append(divFrames, avgPower)
        startIndex += jump
        endIndex += jump

    return divFrames

# names = ['Ajinkya','Ari','Arrow','Arun','Bhargav','Cruise','DadPatil','DC','Dicap_Django', 'Dipack','Doctor','Goswami','Harsh','Harvey','JE','JK','Simmons','Kapish','KL','Leo_Wolf','Liam','Loki','Louis','Malkan','Manas','Mathur','MichaelCera','Nicholson','Nishant','OldGuy','Puneet','RandomMan','Rishi','Robert','Rohit','Sad','Tobymac','Tushar','Tyrion','Vaas','Akansha','Arzoo','Emily','Isha','Lorelai','MomPatil','Olivia','Pallavi','Paris','PDB','Pooja','Pritha','Sayalee','Tanvi','Unsorted','YoungWoman']
names = ['Pallavi']

def main():

    emotions = ["normal", "angry"]

    powerDivFrameFiles = []
    allAvgPower = {}
    avgPowerAll = []
    minDivFrames = []
    startIndexForFiles = {}

    meanNormal = 0
    meanAngry = 0

    count = 0
    totalCount = 0

    startPowerIndex = 0

    for person in names:
        for emotion in emotions:
            targetFiles = []

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

            # print(targetFiles)

            if(len(targetFiles) < 2):
                print(person, "does not have sufficient number of data files!")
                continue
            elif(len(targetFiles) > 2):
                print(person, "has too many similarly named data files!")
                continue

            powerDivFrameFiles = np.append(powerDivFrameFiles, targetFiles[0])

            avgPowerOfFrames, entropy = powerAndEntropy(targetFiles[0], targetFiles[1])

            divFrames = splitSignal(targetFiles[0])

            minDivFrames = np.append(minDivFrames, min(divFrames))
            divFrames = divFrames - min(divFrames)
            
            # if ("normal" in targetFiles[0]) or ("neutral" in targetFiles[0]): 
            #     meanNormal = np.mean(divFrames)
            # elif ("angry" in targetFiles[0]): 
            #     meanAngry = np.mean(divFrames) 
            
            # np.savetxt('Windowed_Power_ZS_Comps/' + targetFiles[0].split('/')[-1].split('.wav')[0] + '.csv', divFrames, fmt='%10.5f')

            # Creates Key Value pairs showing avgPowerOfFrames for each file
            allAvgPower[targetFiles[0]] = avgPowerOfFrames

            # Shows where each file in the amalgamated powerOfFrames array starts from
            startIndexForFiles[targetFiles[0]] = startPowerIndex
            startPowerIndex += len(avgPowerOfFrames)

            # Amalgamates all the avgPowers of each file
            avgPowerAll = np.append(avgPowerAll, avgPowerOfFrames)


        # if(meanNormal != 0) and (meanAngry != 0):
        #     print(person, "Normal:", meanNormal, "Angry:", meanAngry)
        #     totalCount += 1
            
        #     if(meanAngry > meanNormal):
        #         count += 1

        # meanNormal = 0
        # meanAngry = 0

    # totalZS, powerFileMentions, angryCount, normalCount = totalPowerZScoreCalc(avgPowerAll, startIndexForFiles)

    # for person in names:
    #     for emotion in emotions:
    #         targetFiles = []

    #         for filename in powerFiles:
    #             if person in filename:
    #                 if emotion == "normal":
    #                     if ("normal" in filename) or ("neutral" in filename):
    #                         targetFiles.append(filename)
    #                 elif emotion in filename:
    #                     targetFiles.append(filename)

    #         for filename in entropyFiles:
    #             if person in filename:
    #                 if emotion == "normal":
    #                     if ("normal" in filename) or ("neutral" in filename):
    #                         targetFiles.append(filename)
    #                 elif emotion in filename:
    #                     targetFiles.append(filename)

    #         if(len(targetFiles) < 2):
    #             print(person, "does not have sufficient number of data files!")
    #             continue
    #         elif(len(targetFiles) > 2):
    #             print(person, "has too many similarly named data files!")
    #             continue

    #         divFrames = splitSignal(targetFiles[0])

    #         divFrames = divFrames - np.mean(minDivFrames)
            
    #         if ("normal" in targetFiles[0]) or ("neutral" in targetFiles[0]): 
    #             meanNormal = np.mean(divFrames)
    #         elif ("angry" in targetFiles[0]): 
    #             meanAngry = np.mean(divFrames) 
            
    #         # np.savetxt('Windowed_Power_ZS_Comps/' + targetFiles[0].split('/')[-1].split('.wav')[0] + '.csv', divFrames, fmt='%10.5f')


    #     if(meanNormal != 0) and (meanAngry != 0):
    #         print(person, "Normal:", meanNormal, "Angry:", meanAngry)
    #         totalCount += 1
            
    #         if(meanAngry > meanNormal):
    #             count += 1

    #     meanNormal = 0
    #     meanAngry = 0

    print(count, "of", totalCount)


if __name__ == '__main__':
    main()
