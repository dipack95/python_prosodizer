import pandas as pd
import numpy as np
import sklearn
from sklearn.mixture import GMM
from matplotlib import pyplot

def main():
	men = np.array(pd.read_csv('/Users/kirit/BtechProject/Analysis/Men_angry-neutral_features', header=None))
	women = np.array(pd.read_csv('/Users/kirit/BtechProject/Analysis/Women_angry-neutral_features', header=None))

	mdata = men[:,1:]
	mlabels = men[:,0]

	wdata = women[:,1:]
	wlabels = women[:,0]

	gmm = []

	for i in range(4, 12):
		gmm.append(GMM(n_components=i, n_init=250, covariance_type = 'tied'))

	for g in gmm:
		print gmm.index(g)
		while not g.converged_:
			print "Fitting the gmm"
			g.fit(mdata)
			print "Has the gmm converged?", g.converged_

if __name__ == '__main__':
	main()
