import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os
import math
import scipy
from scipy.stats import f
from scipy import stats

np.set_printoptions(threshold=np.inf)

def print_features(localFile, dumpFile):
    mfcc = np.nan_to_num(np.array(pd.read_csv(localFile + '_mfcc.csv', header=None), dtype='float64'))
    energy = np.nan_to_num(np.array(pd.read_csv(localFile + '_energyOfFrames.csv', header=None), dtype='float64'))[0][:-1]
    # labels = np.array(pd.read_csv(localFile + '_labels.csv', header=None))

    avgOfMfcc = np.mean(mfcc, axis = 0)

    for tempMfcc in mfcc:
        for i in range(len(tempMfcc)):
            tempMfcc[i] = (tempMfcc[i] - avgOfMfcc[i])
        # print(tempMfcc[0], tempMfcc[1], tempMfcc[2], tempMfcc[3], tempMfcc[4], tempMfcc[5], tempMfcc[6], tempMfcc[7], tempMfcc[8], tempMfcc[9], (mfcc == tempMfcc)[0][0]]))
        print(tempMfcc[0], tempMfcc[1], tempMfcc[2], tempMfcc[3], tempMfcc[4], tempMfcc[5], tempMfcc[6], tempMfcc[7], tempMfcc[8], tempMfcc[9], tempMfcc[10], tempMfcc[11], tempMfcc[12], file = dumpFile)
    return

def main():

    printToFileNameMen = "men_angry_neutral_mfcc.csv"
    printToFileNameWomen = "women_angry_neutral_mfcc.csv"
    
    if os.path.isfile(printToFileNameMen):
        os.remove(printToFileNameMen)
    if os.path.isfile(printToFileNameWomen):
        os.remove(printToFileNameWomen)

    for path, directories, filenames in os.walk('../sounds/'):
        for filename in filenames: 
            if ".wav" in filename and ".csv" not in filename:
                if "angry" in filename or "neutral" in filename or "anger" in filename or "normal" in filename:
                    if "Men" in path:
                        printToFileName = printToFileNameMen
                        printToFile = open(printToFileName, "a+")
                    elif "Women" in path:
                        printToFileName = printToFileNameWomen
                        printToFile = open(printToFileName, "a+")
                    localFileName = os.path.join(path, filename) 
                    localFileName = localFileName.replace("/sounds/", "/csv/")
                    print_features(localFileName, printToFile)

if __name__ == '__main__':
    main()
