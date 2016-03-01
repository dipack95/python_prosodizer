import numpy as np
import pandas as pd
import os

def main():
	data = np.array(pd.read_csv('Docs/labelsAndPredictedLabels.csv', header=None, sep=' '))

	# In Milliseconds
	frameLength = 25
	blockLength = 1000

	numOfFrames = blockLength // frameLength

	startIndex = 0
	endIndex = numOfFrames

	numOfBlocks = np.ceil(data.shape[0] / numOfFrames)

	for _ in range(np.int(numOfBlocks)):
		'''
		# NOTE: If the frames are not divisible by the numOfFrames,
		# When it reaches the last block, since it is not of the exact size, it just takes how many ever values it can find,
		# creating a block of less than the size that we want
		'''
		dataBlock = data[startIndex : endIndex, :]
		
		dataBlockLabels = dataBlock[:, 0]
		dataBlockClusters = dataBlock[:, 1]
		values, counts = np.lib.arraysetops.unique(dataBlockLabels, return_counts=True)
		clusters, clustersCount = np.lib.arraysetops.unique(dataBlockClusters, return_counts=True)

		percentagesOfEmotions = {}
		for i in range(len(counts)):
			percentagesOfEmotions[values[i]] = (counts[i] / numOfFrames) * 100 

		percentagesOfClusters = {}
		for i in range(len(clustersCount)):
			percentagesOfClusters[clusters[i]] = (clustersCount[i] / numOfFrames) * 100

		print(percentagesOfEmotions, percentagesOfClusters)
		startIndex += numOfFrames
		endIndex += numOfFrames

if __name__ == '__main__':
    main()