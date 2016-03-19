import numpy as np
import pandas as pd
import os

from sklearn.mixture import GMM
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import preprocessing

def print_features(localFile, dumpFile):
    mfcc = np.nan_to_num(np.array(pd.read_csv(localFile + '_mfcc.csv', header=None), dtype='float64'))
    energy = np.nan_to_num(np.array(pd.read_csv(localFile + '_energyOfFrames.csv', header=None), dtype='float64'))[0][:-1]
    labels = np.array(pd.read_csv(localFile + '_labels.csv', header=None))[:, 1]

    if mfcc.shape[0] != len(labels):
        print("The number of data frames in the MFCC CSV, and the number of labels for frames, are not equal. MFCC Shape:", mfcc.shape, "Number of labels:", len(labels))
        return

    avgOfMfcc = np.mean(mfcc, axis = 0)

    j = 0
    for tempMfcc in mfcc:
        for i in range(len(tempMfcc)):
            tempMfcc[i] = (tempMfcc[i] - avgOfMfcc[i])
        # print(labels[j], tempMfcc[0], tempMfcc[1], tempMfcc[2], tempMfcc[3], tempMfcc[4], tempMfcc[5], tempMfcc[6], tempMfcc[7], tempMfcc[8], tempMfcc[9], tempMfcc[10], np.log10(energy[j]), file=dumpFile)
        print(labels[j], tempMfcc[0], tempMfcc[1], tempMfcc[2], tempMfcc[3], tempMfcc[4], tempMfcc[5], tempMfcc[6], tempMfcc[7], tempMfcc[8], tempMfcc[9], file=dumpFile)
        j += 1

    return

def showClustersForEmotions(labels, predictedLabels):
	silences = np.unique(predictedLabels[np.where(labels == 'Silence')])
	angry = np.unique(predictedLabels[np.where(labels == 'Angry')])
	neutral = np.unique(predictedLabels[np.where(labels == 'Neutral')])
	hybrid = np.unique(predictedLabels[np.where(labels == 'Hybrid')])
	noise = np.unique(predictedLabels[np.where(labels == 'Noise')])

	print("Silences", silences)
	print("Angry", angry)
	print("Neutral", neutral)
	print("Hybrid", hybrid)
	print("Noise", noise)

	vals = []
	for tempVal in angry:
	  if(tempVal not in hybrid) and (tempVal not in neutral) and (tempVal not in noise) and (tempVal not in silences):
	  	vals.append(tempVal)

	un2angry = vals
	print("Unique to angry", vals)

	vals = []
	for tempVal in neutral:
	  if(tempVal not in hybrid) and (tempVal not in angry) and (tempVal not in noise) and (tempVal not in silences):
	  	vals.append(tempVal)

	un2neutral = vals
	print("Unique to neutral", vals)
	return

def printLabelsAndClusters(labels, predictedLabels, dumpFile):
	if len(labels) != len(predictedLabels):
		print("The number of labels is not equal to the number of labels returned by clustering. Labels:", len(labels), "Predicted Labels:", len(predictedLabels))
		return

	for i in range(len(labels)):
		print(labels[i], predictedLabels[i], file=dumpFile)
	return

def scaleData(data):
		stdScaler = preprocessing.StandardScaler()
		data = np.reshape(data, (len(data), 1))
		scaledData = stdScaler.fit_transform(data)
		return np.ravel(scaledData)

def calculateEmotionalProbabilities(trainingProbabilities, blockProbabilities):

	'''

	Block Probabilities are the probabilities for a block of frames.
	Block Probabilities must be of the form:

		[Cluster]	[PDF]
		0			15.6
		11			18.4
		...

	Training Probabilites are the probabilities for the entire training dataset.
	Training Probabilties must be of the form:

		[Emotional_Cluster]	[PDF]
		Angry_1				82.1234
		Silence_10 			19.5678
		...

	The training matrix being created is in the following order:

		[Row]	[Emotion]	0	1	2	3	...
		0		Angry
		1		Sad
		2		Neutral
		3		Noise
		4		Hybrid
		5		Silence

	'''

	# trainpdf = "/Users/kirit/BtechProject/Analysis/PDF/trainingPDF.csv"
	# trainingProbabilities = np.array(pd.read_csv(trainpdf, header=None, sep=' '))

	if type(trainingProbabilities) is dict:
		trainingProbabilities = np.array(list(trainingProbabilities.items()))

	trainingMatrix = np.zeros(shape=(6, 12))	

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
	incomingProbabilities = np.zeros(shape=(12, 1))

	for i in blockProbabilities:
		incomingProbabilities[i[0]] = i[1]

	sentimentProbabilities = np.dot(trainingMatrix, incomingProbabilities)
	sentimentProbabilities = np.divide(sentimentProbabilities, 100) #Divide by 100 to get result in terms of percentages

	sentimentScores = {'Angry': sentimentProbabilities[0][0],
						'Sad': sentimentProbabilities[1][0],
						'Neutral': sentimentProbabilities[2][0],
						'Noise': sentimentProbabilities[3][0],
						'Hybrid': sentimentProbabilities[4][0],
						'Silence': sentimentProbabilities[5][0]
						}

	print(sentimentScores)
	maximum = max(sentimentScores, key=sentimentScores.get)  
	print("Highest percentage ", maximum, sentimentScores[maximum], "\n")

	return sentimentScores

def pdfFromClusters(localFile):
	data = np.array(pd.read_csv(localFile, header=None, sep=' '))

	frameLength = 25
	blockLength = 1000
	jump = 250

	numOfFrames = blockLength // frameLength

	startIndex = 0
	endIndex = numOfFrames
	jumpIndex = jump // frameLength

	numOfBlocks = np.ceil(data.shape[0] / jumpIndex - (numOfFrames // jumpIndex))
	weightOfEmotions = {}
	clustersForEmotions = {}
	blockProbabilities = []
	for _ in range(np.int(numOfBlocks)):
		'''
		# NOTE: If the frames are not divisible by the numOfFrames,
		# When it reaches the last block, since it is not of the exact size, it just takes how many ever values it can find,
		# creating a block of less than the size that we want
		'''
		dataBlock = data[startIndex : endIndex, :]
		
		dataBlockLabels = dataBlock[:, 0]
		dataBlockClusters = dataBlock[:, 1]
		emotions, emotionsCount = np.lib.arraysetops.unique(dataBlockLabels, return_counts=True)
		clusters, clustersCount = np.lib.arraysetops.unique(dataBlockClusters, return_counts=True)

		percentagesOfEmotions = {}
		for i in range(len(emotionsCount)):
			percentagesOfEmotions[emotions[i]] = (emotionsCount[i] / numOfFrames) * 100
			
			if emotions[i] not in list(weightOfEmotions.keys()):
				weightOfEmotions[emotions[i]] = percentagesOfEmotions[emotions[i]]
			else:
				weightOfEmotions[emotions[i]] += percentagesOfEmotions[emotions[i]]

		percentagesOfClusters = {}
		for i in range(len(clustersCount)):
			percentagesOfClusters[clusters[i]] = (clustersCount[i] / numOfFrames) * 100

		for i in range(len(emotions)):
			for j in range(len(clusters)):
				val = (percentagesOfEmotions[emotions[i]]) * percentagesOfClusters[clusters[j]]
				tempKey = emotions[i] + '_' + str(clusters[j])
				
				if tempKey not in list(clustersForEmotions.keys()):
					clustersForEmotions[tempKey] = val
				else:
					clustersForEmotions[emotions[i] + '_' + str(clusters[j])] += val
				
		blockProbability = np.array(list(percentagesOfClusters.items()))
		blockProbabilities.append(blockProbability)
		#startIndex += numOfFrames
		#endIndex += numOfFrames
		startIndex += jumpIndex
		endIndex = startIndex + numOfFrames
	oldClustersForEmotions = clustersForEmotions

	for tempEmote in list(weightOfEmotions.keys()):
		for tempEmotionCluster in list(clustersForEmotions.keys()):
			if tempEmote in tempEmotionCluster:
				stdWeight = clustersForEmotions[tempEmotionCluster] / weightOfEmotions[tempEmote]
				clustersForEmotions[tempEmotionCluster] = stdWeight

	return weightOfEmotions, clustersForEmotions, blockProbabilities


def main():
	localFileName = "../Kirit_Data/csv/Men/Al_Pacino/pacino_devils_angry.wav"
	localName = localFileName.split('/')[-1]
	# Dumping the features for the new file
	dumpFeaturesFile = localName + "_pdfFeatures.csv"	
	if os.path.isfile(dumpFeaturesFile):
		os.remove(dumpFeaturesFile)
	dumpFeaturesFileObj = open(dumpFeaturesFile, "a+")  
	print_features(localFileName, dumpFeaturesFileObj)
	dumpFeaturesFileObj.close()
	
	if not os.path.isfile(dumpFeaturesFile):
		print("Features for", localFileName.split('/')[-1], "not dumped.\nExiting program.")
		return
	print("Features for", localFileName.split('/')[-1], "dumped.")
	# End dumping of features for the new file

	# Fitting the new data to a Kmeans Object
	numOfClusters = 12
	targetSex ="men"
	dataFile = np.array(pd.read_csv(dumpFeaturesFile, header=None, sep=' '))
	dataLabels = dataFile[:, 0]
	data = dataFile[:, 1:].astype(np.float64)

	trainedClusterFileName = "KMeans_Trained_Clusters/" + str(numOfClusters) + "_" + str(targetSex) + ".pkl" 
	
	if os.path.isfile(trainedClusterFileName):
		km = joblib.load(trainedClusterFileName)
		print("Picked up from:", trainedClusterFileName)
	else:
		print("The following KMeans Object", trainedClusterFileName, "is not available.")
		return

	for i in range(data.shape[1]):
		data[:, i] = scaleData(np.ravel(data[:, i]))
	
	centers = km.cluster_centers_
	predictedLabels = km.predict(data)

	dumpLabelsFile = localName + "_lpl.csv"
	if os.path.isfile(dumpLabelsFile):
		os.remove(dumpLabelsFile)
	dumpLabelsFileObj = open(dumpLabelsFile, "a+")    
	# showClustersForEmotions(dataLabels, predictedLabels)
	printLabelsAndClusters(dataLabels, predictedLabels, dumpLabelsFileObj)
	dumpLabelsFileObj.close()

	if not os.path.isfile(dumpLabelsFile):
		print("Labels for", localFileName.split('/')[-1], "not dumped.\nExiting program. \n")
		return
	print("Labels for", localFileName.split('/')[-1], "dumped.\n")
	# End fitting the data, and dumping of predicted labels for new data


	#Getting the training matrix
	# trainingfile = "/Users/kirit/BtechProject/Analysis/PDF/trainingPDF.csv"

	# trainingMatrix = getTrainingMatrix(trainingfile)

	weightOfEmotionsGlobal, clustersForEmotionsGlobal, blockProbabilitiesGlobal = pdfFromClusters('Docs/labelsAndPredictedLabels.csv')

	# Dumping probability density functions by comparing our labels, 
	# to KMeans predicted labels, for new data.
	weightOfEmotionsLocal, clustersForEmotionsLocal, blockProbabilitiesLocal = pdfFromClusters(dumpLabelsFile)
	for blockProbability in blockProbabilitiesLocal:
		sentimentScores = calculateEmotionalProbabilities(clustersForEmotionsGlobal, blockProbability)

	print("----------Program End----------")

	return

if __name__ == '__main__':
    main()