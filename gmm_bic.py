import pandas as pd
from sklearn import cluster
from sklearn import mixture
from scipy.spatial import distance
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

X = np.array(pd.read_csv('Docs/men_angry_neutral_mfcc.csv', header=None, sep=' '))
BIC = []

for i in range(2, 256):
	g = mixture.GMM(n_components=i)
	g.fit(X)
	bicVal = g.bic(X)
	BIC.append(bicVal)
	print("Number of components = ", i, " BIC = ", bicVal)

gmmRange = range(2, 20)
plt.plot(gmmRange, BIC, 'r-o')
plt.title("iris data  (cluster vs BIC)")
plt.xlabel("# clusters")
plt.ylabel("# BIC")
plt.show()