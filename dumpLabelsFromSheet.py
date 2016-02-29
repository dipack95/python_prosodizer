import pandas as pd
import numpy as np
import os


def main():

	labelsWorkbook = pd.read_excel("Docs/Labels_25.xlsx", header=0, sheetname=1)
	namesData = np.ravel(np.array(pd.read_csv('Docs/filenames/men_filenames.csv', header=None)))
	columnNames = labelsWorkbook.columns.values
	j = 0
	for i in range(2, len(columnNames), 3):
		columnData = labelsWorkbook[columnNames[i]].dropna().values
		columnName = columnNames[i - 2]
		if(j < len(namesData)):
			name = namesData[j]
			if(namesData[j][0] == "#"):
				newName = namesData[j][1:len(namesData[j])]
				name = newName
			# Dump to file
			print(name)
			pd.DataFrame(columnData).to_csv(name + '_labels.csv', header=False)
			j += 1

	labelsWorkbook = pd.read_excel("Docs/Labels_25.xlsx", header=0, sheetname=0)
	namesData = np.ravel(np.array(pd.read_csv('Docs/filenames/women_filenames.csv', header=None)))
	columnNames = labelsWorkbook.columns.values
	j = 0
	for i in range(2, len(columnNames), 3):
		columnData = labelsWorkbook[columnNames[i]].dropna().values
		columnName = columnNames[i - 2]
		if(j < len(namesData)):
			name = namesData[j]
			if(namesData[j][0] == "#"):
				newName = namesData[j][1:len(namesData[j])]
				name = newName
			# Dump to file
			print(name)
			pd.DataFrame(columnData).to_csv(name + '_labels.csv', header=False)
			j += 1

if __name__ == '__main__':
    main()