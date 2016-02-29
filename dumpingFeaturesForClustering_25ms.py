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
    labels = np.array(pd.read_csv(localFile + '_labels.csv', header=None))[:, 1]

    avgOfMfcc = np.mean(mfcc, axis = 0)

    # for i in range(len(mfcc)):
    #     for j in range(len(mfcc[i])):
    #         mfcc[i][j] -= avgOfMfcc[j]
    #     print(labels[i], mfcc[i][0], np.log10(energy[i]))

    j = 0
    for tempMfcc in mfcc:
        for i in range(len(tempMfcc)):
            tempMfcc[i] = (tempMfcc[i] - avgOfMfcc[i])
        # print(labels[j], tempMfcc[0], tempMfcc[1], tempMfcc[2], tempMfcc[3], tempMfcc[4], tempMfcc[5], tempMfcc[6], tempMfcc[7], tempMfcc[8], tempMfcc[9], tempMfcc[10], tempMfcc[11], tempMfcc[12], np.log10(energy[j]))
        print(labels[j], tempMfcc[0], tempMfcc[1], tempMfcc[2], tempMfcc[3], tempMfcc[4], tempMfcc[5], tempMfcc[6], tempMfcc[7], tempMfcc[8], tempMfcc[9], tempMfcc[10], tempMfcc[11], tempMfcc[12], np.log10(energy[j]), file=dumpFile)
        j += 1

    return

def main():

    printToFileNameMen = "Docs/men_angry_neutral_mfcc.csv"
    printToFileNameWomen = "Docs/women_angry_neutral_mfcc.csv"
    
    if os.path.isfile(printToFileNameMen):
        os.remove(printToFileNameMen)
    if os.path.isfile(printToFileNameWomen):
        os.remove(printToFileNameWomen)

    menFiles = np.ravel(np.array(pd.read_csv('Docs/filenames/men_filenames.csv', header=None)))
    womenFiles = np.ravel(np.array(pd.read_csv('Docs/filenames/women_filenames.csv', header=None)))
    for filename in menFiles:
        if filename[0] == "#":
            continue
        else:
            printToFile = open(printToFileNameMen, "a+")
            print(filename)
            print_features(filename, printToFile)
    for filename in womenFiles:
        if filename[0] == "#":
            continue
        else:
            printToFile = open(printToFileNameWomen, "a+")
            print(filename)
            print_features(filename, printToFile)


if __name__ == '__main__':
    main()
