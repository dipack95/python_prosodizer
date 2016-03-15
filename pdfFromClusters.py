import numpy as np
import pandas as pd
import os

def main():
	data = np.array(pd.read_csv('Docs/labelsAndPredictedLabels.csv', header=None, sep=' '))
	# In Milliseconds
	frameLength = 25
	blockLength = 1000
	jump = 250

	numOfFrames = blockLength // frameLength

	startIndex = 0
	endIndex = numOfFrames
	jumpIndex = jump // frameLength

	numOfBlocks = np.ceil(data.shape[0] / jumpIndex - (numOfFrames // jumpIndex))
	#numOfBlocks = np.ceil(data.shape[0] / numOfFrames)
	weightOfEmotions = {}
	clustersForEmotions = {}
	for _ in range(np.int(numOfBlocks)):
		'''
		# NOTE: If the frames are not divisible by the numOfFrames,
		# When it reaches the last block, since it is not of the exact size, it just takes how many ever values it can find,
		# creating a block of less than the size that we want
		'''
		#print (startIndex, endIndex)

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

		#startIndex += numOfFrames
		#endIndex += numOfFrames

		startIndex += jumpIndex
		endIndex = startIndex + numOfFrames


	# print(weightOfEmotions, clustersForEmotions)

	for tempEmote in list(weightOfEmotions.keys()):
		for tempEmotionCluster in list(clustersForEmotions.keys()):
			if tempEmote in tempEmotionCluster:
				stdWeight = clustersForEmotions[tempEmotionCluster] / weightOfEmotions[tempEmote]
				print(tempEmotionCluster, stdWeight)

if __name__ == '__main__':
    main()