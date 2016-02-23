import pandas as pd
import numpy as np
import sklearn
import os
from sklearn.mixture import GMM
from sklearn.cluster import KMeans
from sklearn import preprocessing
from matplotlib import pyplot       

def scaleData(data):
		stdScaler = preprocessing.StandardScaler()
		scaledData = stdScaler.fit_transform(data)
		return scaledData

men = np.array(pd.read_csv('Docs/men_angry_neutral_mfcc.csv', header=None, sep = ' '), dtype='float64')
women = np.array(pd.read_csv('Docs/women_angry_neutral_mfcc.csv', header=None, sep = ' '), dtype='float64')

menData = men
womenData = women

# print(menData, menData.shape)

for i in range(menData.shape[1]):
	menData[:, i] = scaleData(menData[:, i])

for i in range(2, 256):
	km = KMeans(n_clusters=i, n_jobs=-1)
	km.fit(menData)
	print("Kmeans Index:", i, "Inertia:", km.inertia_)