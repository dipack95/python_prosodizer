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

def main():

    printToFileNameMen = "Docs/filenames/men_filenames.csv"
    printToFileNameWomen = "Docs/filenames/women_filenames.csv"
    
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
                    print(filename)
                    localFileName = os.path.join(path, filename) 
                    localFileName = localFileName.replace("/sounds/", "/csv/")
                    print(localFileName, file=printToFile)

if __name__ == '__main__':
    main()
