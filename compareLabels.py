import pandas as pd
import numpy as np
import sklearn
from sklearn.mixture import GMM
from sklearn.cluster import KMeans
from sklearn import preprocessing
from matplotlib import pyplot

# np.set_printoptions(threshold=np.inf)

def scaleData(stdScaler, data):
		scaledData = stdScaler.fit_transform(data)
		return scaledData

men = np.array(pd.read_csv('Docs/men_angry_neutral_mfcc.csv', header=None, sep=' '))
women = np.array(pd.read_csv('Docs/women_angry_neutral_mfcc.csv', header=None, sep=' '))

menData = men[:, 1:]
menLabels = men[:, 0]

womenData = women[:, 1:]
womenLabels = women[:, 0]

data = menData
labels = menLabels

kmeans = []


stdScaler = preprocessing.StandardScaler()
for i in range(data.shape[1]):
	data[:, i] = scaleData(stdScaler, data[:, i])

km = KMeans(n_clusters = 13, n_jobs = -1)
km.fit(data)

centers = km.cluster_centers_

predictedLabels = km.predict(data)

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