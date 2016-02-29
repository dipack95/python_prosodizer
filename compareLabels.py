import pandas as pd
import numpy as np
import sklearn
from sklearn.mixture import GMM
from sklearn.cluster import KMeans
from sklearn import preprocessing
from matplotlib import pyplot

np.set_printoptions(threshold=np.inf)

def scaleData(stdScaler, data):
		scaledData = stdScaler.fit_transform(data)
		return scaledData

men = np.array(pd.read_csv('/Users/kirit/BtechProject/Analysis/Men_angry-neutral_features', header=None))
women = np.array(pd.read_csv('/Users/kirit/BtechProject/Analysis/Women_angry-neutral_features', header=None))

mdata = men[:,1:]
mlabels = men[:,0]

wdata = women[:,1:]
wlabels = women[:,0]

kmeans = []

stdScaler = preprocessing.StandardScaler()
for i in range(mdata.shape[1]):
	mdata[:, i] = scaleData(stdScaler, mdata[:, i])

km = KMeans(n_clusters = 84, n_jobs = -1)
km.fit(mdata)

centers = km.cluster_centers_

labels = km.predict(mdata)

silences = np.unique(labels[np.where(mlabels == 'Silence')])
angry = np.unique(labels[np.where(mlabels == 'Angry')])
neutral = np.unique(labels[np.where(mlabels == 'Neutral')])
hybrid = np.unique(labels[np.where(mlabels == 'Hybrid')])
noise = np.unique(labels[np.where(mlabels == 'Noise')])

print "Silences"
print silences

print "Angry"
print angry

print "Neutral"
print neutral

print "Hybrid"
print hybrid

print "Noise"
print noise


vals = []
for tempVal in angry:
  if(tempVal not in hybrid) and (tempVal not in neutral) and (tempVal not in noise) and (tempVal not in silences):
	vals.append(tempVal)

un2angry = vals
print "Unique to angry", vals

vals = []
for tempVal in neutral:
  if(tempVal not in hybrid) and (tempVal not in angry) and (tempVal not in noise) and (tempVal not in silences):
	vals.append(tempVal)

un2neutral = vals
print "Unique to neutral", vals

'''
others = []
for i in range(1, 106):
	if i not in un2angry and i not in un2neutral:
		others.append(i)

new_clusters = 32
n = 32 - len(un2angry) - len(un2neutral)

if len(others) > n:
	new_others = others[0:n]
else:
	new_others = 32 - (len(others) - n)

newcenters = np.vstack((centers[un2angry], centers[un2neutral], centers[new_others]))

kmn = KMeans(n_clusters = new_clusters, init = newcenters)
kmn.fit(wdata)
newlabels = kmn.predict(wdata)

nsilences = np.unique(newlabels[np.where(wlabels == 'Silence')])
nangry = np.unique(newlabels[np.where(wlabels == 'Angry')])
nneutral = np.unique(newlabels[np.where(wlabels == 'Neutral')])
nhybrid = np.unique(newlabels[np.where(wlabels == 'Hybrid')])
nnoise = np.unique(newlabels[np.where(wlabels == 'Noise')])

print "Silences"
print nsilences

print "Angry"
print nangry

print "Neutral"
print nneutral

print "Hybrid"
print nhybrid

print "Noise"
print nnoise
'''