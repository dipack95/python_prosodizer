import pandas as pd
import numpy as np
import os

from scipy import stats
from subprocess import call

def dumpLabelsFromSheet(workbook):
	allFiles = [os.path.join(path, name) for path, dirs, files in os.walk("../sounds/") for name in files if ".wav" in name]
	# print(allFiles)
	menFilenames = []
	womenFilenames = []

	menFile = 'Docs/filenames/men_filenames.csv'
	womenFile = 'Docs/filenames/women_filenames.csv'

	if os.path.isfile(menFile):
		os.remove(menFile)
	if os.path.isfile(womenFile):
		os.remove(womenFile)

	# Men 
	labelsWorkbook = pd.read_excel(workbook, header=0, sheetname=1)
	columnNames = labelsWorkbook.columns.values
	j = 0
	for i in range(2, len(columnNames), 3):
		columnData = labelsWorkbook[columnNames[i]].dropna().values
		columnName = columnNames[i - 2]
		# print(columnName)
		labelsFileName = ""
		for tempFile in allFiles:
			if columnName in tempFile:
				labelsFileName = tempFile.replace("/sounds/", "/csv/")
				labelsFileName = labelsFileName + "_labels.csv"
				menFilenames = np.append(menFilenames, tempFile)
				print("Dumping", labelsFileName)
				pd.DataFrame(columnData).to_csv(labelsFileName, header=False)
	pd.DataFrame(menFilenames).to_csv(menFile, header=False, index=False)

	# Women
	labelsWorkbook = pd.read_excel(workbook, header=0, sheetname=0)
	columnNames = labelsWorkbook.columns.valuesudo 
	j = 0
	for i in range(2, len(columnNames), 3):
		columnData = labelsWorkbook[columnNames[i]].dropna().values
		columnName = columnNames[i - 2]
		# print(columnName)
		labelsFileName = ""
		for tempFile in allFiles:
			if columnName in tempFile:
				labelsFileName = tempFile.replace("/sounds/", "/csv/")
				labelsFileName = labelsFileName + "_labels.csv"
				womenFilenames = np.append(womenFilenames, tempFile)
				print("Dumping", labelsFileName)
				pd.DataFrame(columnData).to_csv(labelsFileName, header=False)
	pd.DataFrame(womenFilenames).to_csv(womenFile, header=False, index=False)

	return menFile, womenFile

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

def filterFiles(menFile, womenFile):
    printToFileNameMen = menFile
    printToFileNameWomen = womenFile
    
    if os.path.isfile(printToFileNameMen):
        os.remove(printToFileNameMen)
    if os.path.isfile(printToFileNameWomen):
        os.remove(printToFileNameWomen)

    avgs = {}

    for path, directories, filenames in os.walk('../sounds/Men'):
        for filename in filenames: 
            if ".wav" in filename and ".csv" not in filename:
                if "angry" in filename or "neutral" in filename or "anger" in filename or "normal" in filename:
                    localFileName = os.path.join(path, filename) 
                    localFileName = localFileName.replace("/sounds/", "/csv/")
                    avgs[localFileName] = print_features(localFileName)

    means = np.array(list(avgs.values()))
    names = np.array(list(avgs.keys()))
    scores = find_zscr(names, means)

    for s in scores:
        printToFile = open(printToFileNameMen, "a+")
        if scores[s] > 3 or scores[s] < -3:
            ns = "#" + s
            print (ns, file = printToFile)
        else:
            print (s, file = printToFile)

    avgs = {}

    for path, directories, filenames in os.walk('../sounds/Women'):
        for filename in filenames: 
            if ".wav" in filename and ".csv" not in filename:
                if "angry" in filename or "neutral" in filename or "anger" in filename or "normal" in filename:
                    localFileName = os.path.join(path, filename) 
                    localFileName = localFileName.replace("/sounds/", "/csv/")
                    avgs[localFileName] = print_features(localFileName)

    means = np.array(list(avgs.values()))
    names = np.array(list(avgs.keys()))
    scores = find_zscr(names, means)
    for s in scores:
        printToFile = open(printToFileNameWomen, "a+")
        if scores[s] > 3 or scores[s] < -3:
            ns = "#" + s
            print (ns, file = printToFile)
        else:
            print (s, file = printToFile)
            
    call(["sort", printToFileNameMen, "-o", printToFileNameMen])
    call(["sort", printToFileNameWomen, "-o", printToFileNameWomen])

    return

def main():
	menFile, womenFile = dumpLabelsFromSheet("Docs/Labels_25.xlsx")

	if os.path.isfile(menFile):
		if os.path.isfile(womenFile):
			filterFiles(menFile, womenFile)
			print(menFile, "and", womenFile, "have been filtered for outliers.")
		else:
			print(womenFile, "does not exist.")
	else:
		print(menFile, "does not exist.")

if __name__ == '__main__':
    main()