import numpy as np
import math
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

from scipy import fft
from scipy.fftpack import dct
from scikits.audiolab import wavread
from scipy.signal import blackmanharris, fftconvolve
from numpy import argmax, sqrt, mean, diff, log
from numpy.fft import rfft, irfft
from matplotlib.mlab import find
from parabolic import parabolic
from sklearn import preprocessing

def freq_from_autocorr(sig, fs):
    """
    Estimate frequency using autocorrelation
    """
    # Calculate autocorrelation (same thing as convolution, but with 
    # one input reversed in time), and throw away the negative lags
    corr = fftconvolve(sig, sig[::-1], mode='full')
    corr = corr[len(corr)/2:]
    
    # Find the first low point
    d = diff(corr)
    start = find(d > 0)[0]
    
    # Find the next peak after the low point (other than 0 lag).  This bit is 
    # not reliable for long signals, due to the desired peak occurring between 
    # samples, and other peaks appearing higher.
    # Should use a weighting function to de-emphasize the peaks at longer lags.
    peak = argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)
    
    return fs / px 

def freq_from_HPS(sig, fs):
    """
    Estimate frequency using harmonic product spectrum (HPS)
    
    """
    windowed = sig * blackmanharris(len(sig))

    from pylab import subplot, plot, log, copy, show

    #harmonic product spectrum:
    c = abs(rfft(windowed))
    maxharms = 8
    subplot(maxharms,1,1)
    plot(log(c))
    for x in range(2,maxharms):
        a = copy(c[::x]) #Should average or maximum instead of decimating
        # max(c[::x],c[1::x],c[2::x],...)
        c = c[:len(a)]
        i = argmax(abs(c))
        true_i = parabolic(abs(c), i)[0]
        print 'Pass %d: %f Hz' % (x, fs * true_i / len(windowed))
        c *= a
        subplot(maxharms,1,x)
        plot(log(c))
    show()

def normalizeSignal(signal):
    minMax = preprocessing.MinMaxScaler()
    normalizedSignal = minMax.fit_transform(signal)
    return normalizedSignal

def main():
    inputFile = []
    #inputFile += ["sounds/Men/Bhargav/bhargav-normal.wav", "sounds/Men/Bhargav/bhargav-sad.wav", "sounds/Men/Bhargav/bhargav-angry.wav"]
    inputFile += ["sounds/Women/Pallavi/pallavi-normal.wav", "sounds/Women/Pallavi/pallavi-angry.wav"]
    #inputFile += ["sounds/Men/Cruise/tomcruise-angry.wav", "sounds/Men/Cruise/tomcruise-normal.wav", "sounds/Men/Cruise/tomcruise-crying.wav"]
    #inputFile += ["sounds/Men/Bale/bale-angry.wav", "sounds/Men/Bale/bale-normal.wav"]
    #inputFile += ["sounds/Men/Vaas/vaas-angry.wav", "sounds/Men/Vaas/vaas-calm.wav"]
    #HinputFile += ["sounds/Men/Leonardo/leo_angry.wav", "sounds/Men/Leonardo/leo_normal.wav"]
    #inputFile += ["sounds/Men/Dipack/dipack-normal.wav", "sounds/Men/Dipack/dipack-angry.wav"]
    #inputFile += ["sounds/Men/Manas/manas-angry.wav", "sounds/Men/Manas/manas-normal.wav"]
    #inputFile += ["sounds/Men/Nishant/nishant-angry.wav", "sounds/Men/Nishant/nishant-normal.wav"]

    for tempInFile in inputFile:
        signal, fs, enc = wavread(tempInFile)
        #signal = np.round(signal, decimals=3)
        #signal = normalizeSignal(signal)

        # Getting Pitch
        print("Frequency from AutoCorr is " + str(freq_from_autocorr(signal, fs)) + " Hz for " + tempInFile)
        print("Frequency from Harmonic Product Spectrum")
        freq_from_HPS(signal, fs)

if __name__ == "__main__":
    main()