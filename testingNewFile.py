import numpy as np
import pandas as pd
import os
from sklearn.mixture import GMM
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import preprocessing

def print_features(localFile, dumpFile):
    mfcc = np.nan_to_num(np.array(pd.read_csv(localFile + '_mfcc.csv', header=None), dtype='float64'))
    labelsFile = localFile + '_labels.csv'

    if os.path.isfile(labelsFile):
    	print("Labels exist")
    	labels = np.array(pd.read_csv(labelsFile, header=None))[:, 1]
    else:
    	print("No Labels for this file")

    avgOfMfcc = np.mean(mfcc, axis = 0)

    j = 0
    for tempMfcc in mfcc:
        for i in range(len(tempMfcc)):
            tempMfcc[i] = (tempMfcc[i] - avgOfMfcc[i]) 
        if os.path.isfile(labelsFile):
        	print(labels[j], tempMfcc[0], tempMfcc[1], tempMfcc[2], tempMfcc[3], tempMfcc[4], tempMfcc[5], tempMfcc[6], tempMfcc[7], tempMfcc[8], tempMfcc[9], file=dumpFile)
        	j += 1
        else:
        	print(tempMfcc[0], tempMfcc[1], tempMfcc[2], tempMfcc[3], tempMfcc[4], tempMfcc[5], tempMfcc[6], tempMfcc[7], tempMfcc[8], tempMfcc[9], file=dumpFile)

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

def getTrainingMatrix(trainingfilename, numOfClusters):
	'''
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

	trainingProbabilities = np.array(pd.read_csv(trainingfilename, header=None, sep=' '))

	trainingMatrix = np.zeros(shape=(6, numOfClusters))	

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

	return trainingMatrix

def calculateEmotionalProbabilities(trainingMatrix, blockProbabilities, numOfClusters):

	'''

	Block Probabilities are the probabilities for a block of frames.
	Block Probabilities must be of the form:

		[Cluster]	[PDF]
		0			15.6
		11			18.4
		...

	'''	

	# Each row contains the probability density for that particular cluster
	incomingProbabilities = np.zeros(shape=(numOfClusters, 1))

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


def pdfFromClusters_withLabels(localFile):
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



def pdfFromClusters(predictedLabels):
	data = predictedLabels

	# In Milliseconds
	frameLength = 25
	blockLength = 1000
	jump = 250

	numOfFrames = blockLength // frameLength

	startIndex = 0
	endIndex = numOfFrames
	jumpIndex = jump // frameLength

	numOfBlocks = np.ceil(data.shape[0] / jumpIndex - (numOfFrames // jumpIndex))
	
	blockProbabilities = []
	for _ in range(np.int(numOfBlocks)):
		dataBlock = data[startIndex : endIndex]
		clusters, clustersCount = np.lib.arraysetops.unique(dataBlock, return_counts=True)

		blockClusters = {}
		for i in range(0, len(clusters)):
			blockClusters[clusters[i]] = clustersCount[i] / np.sum(clustersCount) * 100

		
		blockClusters = np.array(list(blockClusters.items()))

		blockProbabilities.append(blockClusters)

		#startIndex += numOfFrames
		#endIndex += numOfFrames
		startIndex += jumpIndex
		endIndex = startIndex + numOfFrames

	return blockProbabilities

def main():
	localFileName = "../csv/Men/Al_Pacino/pacino_devils_neutral.wav"
	localName = localFileName.split('/')[-1]

	labelsFile = localFileName + "_labels.csv"

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
	numOfClusters = 20
	targetSex ="men"

	dataFile = np.array(pd.read_csv(dumpFeaturesFile, header=None, sep=' '))

	if os.path.isfile(labelsFile):
		dataLabels = dataFile[:, 0]
		data = dataFile[:, 1:]
	else:	
		data = dataFile

	data = data.astype(np.float64)

	trainedClusterFileName = "KMeans_Trained_Clusters/" + str(numOfClusters) + "_" + str(targetSex) + ".pkl" 
	
	if os.path.isfile(trainedClusterFileName):
		km = joblib.load(trainedClusterFileName)
		print("Picked up from:", trainedClusterFileName)
	else:
		print("The following KMeans Object", trainedClusterFileName, "is not available.")
		return

	for i in range(data.shape[1]):
		data[:, i] = scaleData(np.ravel(data[:, i]))
	
	predictedLabels = km.predict(data)

	if os.path.isfile(labelsFile):
		dumpLabelsFile = localName + "_lpl.csv"
		if os.path.isfile(dumpLabelsFile):
			os.remove(dumpLabelsFile)
		dumpLabelsFileObj = open(dumpLabelsFile, "a+")    
		printLabelsAndClusters(dataLabels, predictedLabels, dumpLabelsFileObj)
		dumpLabelsFileObj.close()

	#Getting the training matrix
	trainingMatrix = getTrainingMatrix("Docs/trainingPDF.csv", numOfClusters)

	# Dumping probability density functions by comparing our labels, 
	# to KMeans predicted labels, for new data.
	if os.path.isfile(labelsFile):
		weightOfEmotions, clustersForEmotions, blockProbabilities = pdfFromClusters_withLabels(dumpLabelsFile)
	else:
		blockProbabilities = pdfFromClusters(predictedLabels)
	for blockProbability in blockProbabilities:
		sentimentScores = calculateEmotionalProbabilities(trainingMatrix, blockProbability, numOfClusters)

	
	print("----------Program End----------")

	return

if __name__ == '__main__':
    main()