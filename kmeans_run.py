import numpy as np
import pandas as pd
import sklearn.datasets
import matplotlib.pyplot as plt

from sklearn import cluster
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler

np.set_printoptions(threshold=np.inf)

def compute_bic(kmeans,X):
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) - n[i] * np.log(N) - ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) - ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)

dataFile = 'Docs/women_angry_neutral_mfcc.csv'
data = np.array(pd.read_csv(dataFile, header=None, sep=' '))

km = cluster.KMeans(n_clusters = 27, n_jobs = -1).fit(data)
labels = km.predict(data)
uniques, counts = np.unique(labels, True)
print(uniques, counts)
# print(km.labels_)