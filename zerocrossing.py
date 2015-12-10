import numpy as np
import math
import scipy.io.wavfile as wav
from scipy import fft
from scipy import fftpack
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from scipy.stats import entropy

def main():
    """
    Read the wav file. 
    Change type to float64 for precision.
    Frame the signal.
    Compute the power.
    """

    inputFile = ["sounds/Women/normal_woman.wav", "sounds/Women/angry_woman.wav", "sounds/Women/happy_woman.wav" ]
    #inputFile = ["sounds/Men/Normal/calmwalt.wav", "sounds//Men/Normal/filemono.wav"]
    #inputFile = ["sounds/Men/Other/intense-man.wav", "sounds/Men/Angry/angry-fred.wav", "sounds//Men/Sad/sad-man.wav"]
    #inputFile = ["sounds/Men/Angry/angry-niko.wav", "sounds/Men/Normal/soft-male.wav", "sounds/Men/Angry/angry-jap.wav", "sounds/Men/Normal/calmwalt.wav", "sounds/Men/Angry/angrywalt.wav", "sounds/Men/Other/laughing-people.wav", "sounds/Men/Other/hysterical-man.wav"]

    '''
    energyPlot = plt.figure()
    energyPlot.suptitle('Energy')
    velocityPlot = plt.figure()
    velocityPlot.suptitle('Velocity')
    accPlot = plt.figure()
    accPlot.suptitle('Acceleration')
    '''
    
    for tempInFile in inputFile:    
        (rate, sig) = wav.read(tempInFile)
        signal = np.float64(sig / 2 ** 15)

        zero_crossings = np.where(np.diff(np.sign(sig)))[0]  
        print (len(zero_crossings), tempInFile)
        


if __name__ == "__main__":
    main()