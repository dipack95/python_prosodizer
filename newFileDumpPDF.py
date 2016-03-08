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

def pdfFromClusters(localFile):
	data = np.array(pd.read_csv(localFile, header=None, sep=' '))

	# In Milliseconds
	frameLength = 25
	blockLength = 1000

	numOfFrames = blockLength // frameLength

	startIndex = 0
	endIndex = numOfFrames

	numOfBlocks = np.ceil(data.shape[0] / numOfFrames)
	weightOfEmotions = {}
	clustersForEmotions = {}
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
				# print(emotions[i], clusters[j], val)

		# print(percentagesOfEmotions, percentagesOfClusters)
		# print("------------NEXT BLOCK---------")
		startIndex += numOfFrames
		endIndex += numOfFrames

	print(weightOfEmotions, clustersForEmotions)

	for tempEmote in list(weightOfEmotions.keys()):
		for tempEmotionCluster in list(clustersForEmotions.keys()):
			if tempEmote in tempEmotionCluster:
				stdWeight = clustersForEmotions[tempEmotionCluster] / weightOfEmotions[tempEmote]
				print(tempEmotionCluster, stdWeight)

	return


def main():
	localFileName = "../csv/Men/JK/jk_angry.wav"
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
	data = dataFile[:, 1:]

	trainedClusterFileName = "KMeans_Trained_Clusters/" + str(numOfClusters) + "_" + str(targetSex) + ".pkl" 

	km = KMeans(n_clusters = numOfClusters, n_jobs = -1)
	
	if os.path.isfile(trainedClusterFileName):
		km = joblib.load(trainedClusterFileName)
		print("Picked up from:", trainedClusterFileName)
	else:
		print("The following KMeans Object", trainedClusterFileName, "is not available.")
		return

	for i in range(data.shape[1]):
		data[:, i] = scaleData(data[:, i])
	centers = km.cluster_centers_
	predictedLabels = km.predict(data)

	dumpLabelsFile = localName + "_lpl.csv"
	if os.path.isfile(dumpLabelsFile):
		os.remove(dumpLabelsFile)
	dumpLabelsFileObj = open(dumpLabelsFile, "a+")    
	# showClustersForEmotions(labels, predictedLabels)
	printLabelsAndClusters(dataLabels, predictedLabels, dumpLabelsFileObj)
	dumpLabelsFileObj.close()

	if not os.path.isfile(dumpLabelsFile):
		print("Labels for", localFileName.split('/')[-1], "not dumped.\nExiting program.")
		return
	print("Labels for", localFileName.split('/')[-1], "dumped.")
	# End fitting the data, and dumping of predicted labels for new data

	# Dumping probability density functions by comparing our labels, 
	# to KMeans predicted labels, for new data.
	pdfFromClusters(dumpLabelsFile)

	print("----------Program End----------")

	return

if __name__ == '__main__':
    main()