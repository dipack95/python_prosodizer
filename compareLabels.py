import pandas as pd
import numpy as np
import sklearn
import os
from sklearn.mixture import GMM
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.externals import joblib
from matplotlib import pyplot

# np.set_printoptions(threshold=np.inf)

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

def main():
	men = np.array(pd.read_csv('Docs/men_angry_neutral_mfcc.csv', header=None, sep=' '))
	women = np.array(pd.read_csv('Docs/women_angry_neutral_mfcc.csv', header=None, sep=' '))

	menData = men[:, 1:]
	menLabels = men[:, 0]

	womenData = women[:, 1:]
	womenLabels = women[:, 0]

	numOfClusters = 12
	targetSex = "men"
	data = menData
	labels = menLabels
	trainedClusterFileName = "KMeans_Trained_Clusters/" + str(numOfClusters) + "_" + str(targetSex) + ".pkl" 

	km = KMeans(n_clusters = numOfClusters, n_jobs = -1)
	
	for i in range(data.shape[1]):
		data[:, i] = scaleData(data[:, i])
		
	km.fit(data)
	joblib.dump(km, trainedClusterFileName)
	print("Saved to:", trainedClusterFileName)

	centers = km.cluster_centers_
	predictedLabels = km.predict(data)

	dumpFile = 'Docs/labelsAndPredictedLabels.csv'
	if os.path.isfile(dumpFile):
		os.remove(dumpFile)

	dumpFile = open(dumpFile, "a+")    
	# showClustersForEmotions(labels, predictedLabels)
	printLabelsAndClusters(labels, predictedLabels, dumpFile)

if __name__ == '__main__':
    main()