import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.externals import joblib
from sklearn.mixture import GMM
from sklearn.cluster import KMeans
from sklearn import preprocessing
from matplotlib import pyplot

np.set_printoptions(threshold=np.inf)

def scaleData(stdScaler, data):
		scaledData = stdScaler.fit_transform(data)
		return scaledData

def main():

	data = np.array(pd.read_csv("Docs/men_angry_neutral_mfcc.csv", sep = ' ', header=None))[:,1:]
	labels = np.array(pd.read_csv("Docs/men_angry_neutral_mfcc.csv", sep = ' ', header=None))[:,0]

	data = data.astype(np.float64)

	numOfClusters = 20
	targetSex = "men"

	stdScaler = preprocessing.StandardScaler()
	for i in range(data.shape[1]):
		data = scaleData(stdScaler, data)

	km = KMeans(n_clusters = numOfClusters, n_jobs = -1)
	km.fit(data)

	dumpToFile = "KMeans_Trained_Clusters/" + str(numOfClusters) + "_" + str(targetSex) + ".pkl"

	if os.path.isfile(dumpToFile):
		os.remove(dumpToFile)
	        
	joblib.dump(km, dumpToFile)

	return

if __name__ == '__main__':
	main()