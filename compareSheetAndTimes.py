import pandas as pd
import numpy as np
import os

timeFiles = [os.path.join(path, name)
             for path, dirs, files in os.walk("25ms_times/")
             for name in files]

def main():
	print("From CSV")
	menFilesLocations = np.ravel(np.array(pd.read_csv('Docs/filenames/men_filenames.csv', header=None), dtype=str))
	for filename in timeFiles:
		targetFile = filename.split('/')[-1].split('_times')[0]
		targetPath = [pathname for pathname in menFilesLocations if targetFile in pathname.split('/')[-1]]
		if targetFile == "angry.wav":
			targetPath = [pathname for pathname in menFilesLocations if "RandomMan" in pathname and "angry" in pathname]
		if targetFile == "neutral.wav":
			targetPath = [pathname for pathname in menFilesLocations if "RandomMan" in pathname and "neutral" in pathname]
		
		fileTime = pd.read_csv(filename, header=0)
		print(targetFile, fileTime.tail(1).values, fileTime.shape)

	print("From Excel")
	countRows = 0
	labelsWorkbook = pd.read_excel("Docs/Labels_25.xlsx", header=0, sheetname=1, parse_cols=range(0, 252, 3))
	for col in labelsWorkbook:
		print(col, labelsWorkbook[col].dropna().tail(1).values, len(labelsWorkbook[col].dropna()))
		countRows += len(labelsWorkbook[col].dropna())

	print("Rows in Excel:", countRows)

if __name__ == '__main__':
    main()