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



BIC = []
dataFile = 'Docs/men_angry_neutral_mfcc.csv'
data = np.array(pd.read_csv(dataFile, header=None, sep=' '))
dataLabels = data[:, 0]
data = data[:, 1:]
kmRange = range(1, 256)

print(dataFile, data.shape)

for i in kmRange:
    km = cluster.KMeans(n_clusters = i, n_jobs = -1).fit(data)
    kmBIC = compute_bic(km, data)
    print("KMeans Index:", i, "BIC:", kmBIC)
    BIC = np.append(BIC, kmBIC)
    if(i > kmRange[0]):
        if(BIC[i - kmRange[0]] < BIC[i - kmRange[0] - 1]):
            print("-------KNEE-------")

print("BIC Array:", BIC)

plt.plot(kmRange, BIC, 'r-o')
plt.title("iris data  (cluster vs BIC)")
plt.xlabel("# clusters")
plt.ylabel("# BIC")
plt.show()