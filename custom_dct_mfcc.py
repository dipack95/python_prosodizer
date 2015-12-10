import numpy as np
import math
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

from scipy import fft
from scipy.fftpack import dct
from sklearn import preprocessing

def frame_sig(sig, framelen, framestep, winfunc=lambda x:np.ones((1,x))):
    """
    Frame a signal into overlapping frames.
    :param sig: the audio signal to frame.
    :param framelen: length of each frame measured in samples.
    :param framestep: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.    
    :returns: an array of frames. Size is NUMFRAMES by framelen.
    """
    slen = len(sig)
    framelen = int(round(framelen))
    framestep = int(round(framestep))

    if slen <= framelen:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - framelen) / framestep))
    
    padlen = int((numframes - 1) * framestep + framelen)
    padding = np.zeros(padlen - slen)
    padded_sig = np.concatenate((sig, padding))
    
    indices = np.tile(np.arange(0, framelen), (numframes, 1)) + np.tile(np.arange(0, numframes * framestep, framestep), (framelen, 1)).T
    indices = np.array(indices, dtype = np.int32)

    frames = padded_sig[indices]
    #print("indices: " + str(indices) + "\nFrames: " + str(frames))
    #print("Frames shape: " + str(frames.shape) + "\nPadded signal shape: " + str(padded_sig.shape))
    win = np.tile(winfunc(framelen), (numframes,1))
    
    return frames * win

def compPower(frames, NFFT):
    """
    Convert a value in Hertz to Mels
    :param frames: array of frames.
    :returns: Power of frequency bins in every frame.
    """
    fsig = fft(frames, NFFT)
    fsig = abs(fsig)
    x = np.linspace(0, 8000, 512)
    #print(fsig.shape)
    return fsig[:, :NFFT / 2 + 1]

def hz2mel(hz):
    """
    Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1 + hz / 700.0)
    
def mel2hz(mel):
    """
    Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700 * (10 ** (mel / 2595.0) - 1)

def get_filterbanks(nfilt=20,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"
    
    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    # from Hz to fft bin number
    bin = np.floor((nfft + 1) * mel2hz(melpoints) / samplerate)

    fbank = np.zeros([nfilt, nfft / 2 + 1])
    for j in range(0, nfilt):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j,i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j,i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
        
    return fbank

def deltas(mfcc, N):
    '''
    Finds the derivative of the mfcc passed to it
    '''
    S = mfcc.shape
    d = np.zeros((S[0], S[1]))
    for n in range(0, S[1]):
        d[:, n] = mfcc[:, (n + N / 2) % S[1]] - mfcc[:, n - N / 2] #negative indices wrap around
    d = d / N
    return d

def normalizeSignal(signal):
    minMax = preprocessing.MinMaxScaler()
    normalizedSignal = minMax.fit_transform(signal)
    return normalizedSignal

def customDctCombo(energies):
    '''
    Mimics Aquila DSP function
    Combination of Type 2, and Type 3 DCT formulae
    '''
    
    numberOfRows, numberOfCols = energies.shape
    cosines = np.zeros([numberOfCols, numberOfCols])
    output = np.zeros([numberOfRows, numberOfCols])

    c0 = np.sqrt(1 / numberOfCols)
    cn = np.sqrt(2 / numberOfCols)

    for k in range(0, numberOfRows):
        for i in range(0, numberOfCols):
            for j in range(0, numberOfCols):
                cosines[i, j] = np.cos((np.pi * (2 * j + 1) * i) / (2 * numberOfCols))
                output[k, i] += energies[k, j] * cosines[i, j]
            output[k, i] *= c0 if (0 == j) else cn

    return output

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

    energyPlot = plt.figure()
    energyPlot.suptitle('Energy')
    velocityPlot = plt.figure()
    velocityPlot.suptitle('Velocity')
    accPlot = plt.figure()
    accPlot.suptitle('Acceleration')
    powerPlot = plt.figure()
    powerPlot.suptitle('Power Signal Plot')
    numberOfPlots = len(inputFile)
    plotIndex = 0

    # testSineSignalArray = np.arange(0, 20000)
    # testSineSignal = [np.sin(1000 * (2 * 3.14) * i / 44100) for i in testSineSignalArray]
    # framedTestSineSignal = frame_sig(testSineSignal, (25 * (44100/1000)), (10 * (44100/1000)))
    # powerComputedSineSignal = compPower(framedTestSineSignal, 512);
    # print("framedTestSineSignal shape: " + str(framedTestSineSignal.shape))
    # print("framedTestSineSignal: " + str(framedTestSineSignal))
    # print("powerComputedSineSignal shape: " + str(powerComputedSineSignal.shape))
    # print("powerComputedSineSignal: " + str(powerComputedSineSignal))
    
    for tempInFile in inputFile:
        (rate, sig) = wav.read(tempInFile)
        signal = np.float64(sig / 2 ** 15)

        signal = np.round(signal, decimals=3)
        signal = normalizeSignal(signal)
        
        frames = frame_sig(signal, 25 * rate / 1000, 10 * rate / 1000)
        signalPower = compPower(frames, 512)
        filterBanks = get_filterbanks(nfilt = 26)
        filterBankEnergies = np.dot(signalPower, filterBanks.T)
        filterBankEnergies += 1
        filterBankEnergies = np.log10(filterBankEnergies)
        mfccs = customDctCombo(filterBankEnergies)
        deltaOne = deltas(mfccs[:, 3:15], 2)
        deltaTwo = deltas(deltaOne, 2)

        # Plotting Signal Power
        sigPowerPlot = powerPlot.add_subplot((numberOfPlots * 100)+ 10 + plotIndex)
        sigPowerPlot.set_title(tempInFile)
        sigPowerPlot.plot(np.arange(0, signalPower.shape[0]), signalPower[:, 0])
        #print("Signal Power for " + tempInFile + " is " + str(signalPower))

        # Getting moments
        average = np.mean(mfccs[:mfccs.shape[0], 0])
        totalMoment = 0
        for tempMfcc in mfccs[:mfccs.shape[0], 0]:
            tempMoment = tempMfcc * ((tempMfcc - average) ** 2)
            totalMoment += tempMoment
        #print("totalMoment: " + str(totalMoment) + " for " + tempInFile)
        
        # Plotting Filter Bank Energies
        filterBankPlot = energyPlot.add_subplot((numberOfPlots * 100)+ 10 + plotIndex)
        filterBankPlot.set_title(tempInFile)
        filterBankPlot.plot(np.arange(0, mfccs.shape[0]), mfccs[:mfccs.shape[0], 0])
    
        # Plotting Velocities
        flat_del = deltaOne
        flat_del.shape = (1, flat_del.shape[0] * flat_del.shape[1])
        velocitiesPlot = velocityPlot.add_subplot((numberOfPlots * 100)+ 10 + plotIndex)
        velocitiesPlot.set_title(tempInFile)
        velocitiesPlot.plot(np.arange(0, len(flat_del[0][:4800])), flat_del[0][:4800])

        # Plotting Accelerations
        flat_del2 = deltaTwo
        flat_del2.shape = (1, flat_del2.shape[0] * flat_del2.shape[1])
        accPlots = accPlot.add_subplot((numberOfPlots * 100)+ 10 + plotIndex)
        accPlots.set_title(tempInFile)
        accPlots.plot(np.arange(0, len(flat_del2[0][:4800])), flat_del2[0][:4800])

        plotIndex = plotIndex + 1

    plt.show()

if __name__ == "__main__":
    main()