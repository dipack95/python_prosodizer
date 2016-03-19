import pandas as pd
import numpy as np
import sklearn
import os
from subprocess import call
from sklearn.mixture import GMM
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.externals import joblib
from matplotlib import pyplot


def main():

	# Training PDF

	# Angry_1		82
	# Silence_10 	19
	# ... 			...

	clusterPDF = "/Users/kirit/BtechProject/Analysis/PDF/trainingPDF.csv"
	trainingProbabilities = np.array(pd.read_csv(clusterPDF, header=None, sep=' '))


	# INCOMING MATRIX

	# Cluster 			PDF
	# 0					...
	# 1 	 			...
	# ... 				...

	blockPDF = "/Users/kirit/BtechProject/Analysis/testPDF"
	blockProbabilities = np.array(pd.read_csv(blockPDF, header=None, sep=' '))

	# TRAINING MATRIX

	#			0	1	2	3 ... 	11
	# Angry	
	# Neutral
	# Noise	
	# Hybrid
	# Sad
	# Silence

	# Number of rows corresponds to number of emotions 
	# number of columns corresponds to number of clusters
	trainingMatrix = np.zeros(shape=(6,12))	

	for i in trainingProbabilities:
		x = i[0].split('_')
		if "Angry" in x[0]:
			trainingMatrix[0][int(x[1])] = i[1]
		if "Sad" in x[0]:
			trainingMatrix[1][int(x[1])] = i[1]
		if "Neutral" in x[0]:
			trainingMatrix[2][int(x[1])] = i[1]
		if "Noise" in x[0]:
			trainingMatrix[3][int(x[1])] = i[1]
		if "Hybrid" in x[0]:
			trainingMatrix[4][int(x[1])] = i[1]
		if "Silence" in x[0]:
			trainingMatrix[5][int(x[1])] = i[1]

	# Each row contains the probability density for that particular cluster
	incomingProbabilities = np.zeros(shape=(12,1))

	for i in blockProbabilities:
		incomingProbabilities[i[0]] = i[1]

	sentimentProbabilities = np.dot(trainingMatrix, incomingProbabilities)

	sentimentProbabilities = np.divide(sentimentProbabilities, 100) #Divide by 100 to get result in terms of percentages

	print(sentimentProbabilities)
	return

if __name__ == '__main__':
    main()