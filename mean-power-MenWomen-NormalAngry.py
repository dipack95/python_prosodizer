import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os

from scipy.stats import f
from scipy import stats

targetPath = os.path.abspath("../csv")

powerFiles = [os.path.join(path, name)
             for path, dirs, files in os.walk(targetPath)
             for name in files if (name.endswith(("-powerSpectrum.csv")))]

def averagePower(filename):
	power = np.nan_to_num(np.array(pd.read_csv(filename, header=None), dtype='float64'))
	avgPower = np.array(np.mean(power))
	return avgPower

def normalisePower(filename):
	power = np.nan_to_num(np.array(pd.read_csv(filename, header=None), dtype='float64'))
	avgPowerOfBins = np.array(np.mean(power, axis = 0))

	for frame in power:
		for i in range(len(frame)):
			frame[i] -= avgPowerOfBins[i]
	
	sumOfPower = np.sum(power)
	# print(filename.split('/')[-1].split('.wav')[0], sumOfPower)
	return sumOfPower

def main():
	menNeutral = []
	womenNeutral = []
	menAngry = []
	womenAngry = []
	
	for filename in powerFiles:
		if "Men" in filename:
			if("neutral" in filename) or ("normal" in filename):
				menNeutral = np.append(menNeutral, normalisePower(filename))
			elif("angry" in filename):
				menAngry = np.append(menAngry, normalisePower(filename))
		elif "Women" in filename:
			if("neutral" in filename) or ("normal" in filename):
				womenNeutral = np.append(womenNeutral, normalisePower(filename))
			elif("angry" in filename):
				womenAngry = np.append(womenAngry, normalisePower(filename))
		else:
			continue

	print("Average Power Men Angry:", np.mean(menAngry), "Men Neutral:", np.mean(menNeutral), "Women Angry:", np.mean(womenAngry), "Women Neutral:", np.mean(womenNeutral))

if __name__ == '__main__':
    main()
