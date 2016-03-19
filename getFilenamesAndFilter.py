import pandas as pd
import numpy as np
import os

from scipy import stats
from subprocess import call

np.set_printoptions(threshold=np.inf)

def print_features(localFile):
    mfcc = np.nan_to_num(np.array(pd.read_csv(localFile + '_mfcc.csv', header=None), dtype='float64'))

    avgOfMfcc = np.mean(mfcc, axis = 0)

    for tempMfcc in mfcc:
        for i in range(len(tempMfcc)):
            tempMfcc[i] = (tempMfcc[i] - avgOfMfcc[i])

    return np.mean(mfcc)

def find_zscr(names, means):
    zscr = stats.zscore(means)
    return dict(zip(names, zscr))

def filterFiles(folderPath, printToFileName):
    avgs = {}
    for path, directories, filenames in os.walk(folderPath):
        for filename in filenames: 
            if ".wav" in filename and ".csv" not in filename:
                if "angry" in filename or "neutral" in filename or "sad" in filename:
                    localFileName = os.path.join(path, filename) 
                    localFileName = localFileName.replace("/sounds/", "/csv/")
                    avgs[localFileName] = print_features(localFileName)

    means = np.array(list(avgs.values()))
    names = np.array(list(avgs.keys()))
    scores = find_zscr(names, means)

    for s in scores:
        printToFile = open(printToFileName, "a+")
        if scores[s] > 3 or scores[s] < -3:
            ns = "#" + s
            print (ns, file = printToFile)
        else:
            print (s, file = printToFile)

def main():

    printToFileNameMen = "Docs/filenames/men_filenames.csv"
    printToFileNameWomen = "Docs/filenames/women_filenames.csv"
    
    if os.path.isfile(printToFileNameMen):
        os.remove(printToFileNameMen)
    if os.path.isfile(printToFileNameWomen):
        os.remove(printToFileNameWomen)

    filterFiles('../sounds/Men/', printToFileNameMen)
    filterFiles('../sounds/Women/', printToFileNameWomen)
            
    call(["sort", printToFileNameMen, "-o", printToFileNameMen])
    call(["sort", printToFileNameWomen, "-o", printToFileNameWomen])

if __name__ == '__main__':
    main()
