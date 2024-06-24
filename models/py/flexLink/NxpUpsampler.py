# File:   FullUpsampler.py
# Author: Andreas Schwarzinger
# Notes:  This script show the complete upsampling process from 20MHz (the main signal processing rate)
#         to 80MHz (the DAC rate)
#         The process consists of two separate steps:
#         1. An interpolator step (which is a zero stuffing operation combined with a 16 taps filter operating at 80MHz)
#         2. An FIR filter of either 32 taps or 64 taps operating at 80MHz


import os
import sys          
OriginalWorkingDirectory = os.getcwd()
DirectoryOfThisFile      = os.path.dirname(__file__)   # FileName = os.path.basename
if DirectoryOfThisFile != '':
    os.chdir(DirectoryOfThisFile)

# There are modules that we will want to access and they sit two directories above DirectoryOfThisFile
sys.path.append("..\..\DspComm")

import numpy as np
import matplotlib.pyplot as plt
from   FilterDesigner2   import CFilterDesigner 
from   SignalProcessing  import SpectrumAnalyzer

# -----------------------------------------------------------------------------
# > 1a. Define the impulse response of the 16 Tap FIR filter that runs inside the interpolation step.
# ----------------------------------------------------------------------------
# Note, the 16 tap filter nowhere near powerful enough to suppress the aliasing images. It needs a
# Frequency response that is flat with the +/- 10MHz band that the signal occupies.
FilterDesigner  = CFilterDesigner()
GainListLinear  = [.0001, .0001,     1,    1,  .0001, .0001 ]
NormalizedFreq  = [-0.5,   -0.28, -0.20, 0.20,    0.28,    0.5]
N               = 16
h_Interpolator  = (1/N)*FilterDesigner.FrequencySampling(GainListLinear, NormalizedFreq, N,  False).real

if (False):
    FilterDesigner.ShowFrequencyResponse(ImpulseResponse    = h_Interpolator
                                        , SampleRate        = 80.0e6
                                        , OversamplingRatio = 32       
                                        , bShowInDb         = True)


# --------------------------------------------------------------
# > 1b. Define the FIR filter that will do the image rejection caused by the zero-stuffing process
# --------------------------------------------------------------
GainListLinear  = [.0001, .0001,     1,    1,  .0001, .0001 ]
N               = 32
if N == 64:
    NormalizedFreq  = [-0.5,   -0.14, -0.12, 0.12,    0.14,    0.5]
elif N == 32:
    NormalizedFreq  = [-0.5,   -0.20, -0.13, 0.13,    0.20,    0.5]
else:
    assert False, 'The filter length N is invalid.'


h_ImageReject   = (1/N)*FilterDesigner.FrequencySampling(GainListLinear, NormalizedFreq, N,  False).real

if (True):
    FilterDesigner.ShowFrequencyResponse(ImpulseResponse   = h_ImageReject
                                       , SampleRate        = 80.0e6
                                       , OversamplingRatio = 32       
                                       , bShowInDb         = True)


# ----------------------------------------------------------------------------
# > 2. Define an input signal at 20MHz
# ----------------------------------------------------------------------------
SampleRate1     = 20e6         # 20 MHz
Freq1           = 1.0e6        # 1  Mhz
Freq2           = 8.0e6        # 8  Mhz
NumSamples      = 600
n               = np.arange(0, NumSamples, 1, np.float32)
Input20MHz      = np.sin(2*np.pi*n*Freq1/SampleRate1) +  np.sin(2*np.pi*n*Freq2/SampleRate1)

# -----------------------------------------------------------------------
# Ideal output signal - This would be the ideal interpolated signal if we had
# a perfect interpolator. Too bad that we don't. But we can get close
SampleRate2     = 80e6
NumSamples      = 2400
n               = np.arange(0, NumSamples, 1, np.float32)
Ideal80MHz      = np.sin(2*np.pi*n*Freq1/SampleRate2) +  np.sin(2*np.pi*n*Freq2/SampleRate2)


# ----------------------------------------------------------------------------
# > 3a. Use the Zero-Stuffing method (push 3 zeros in between all samples)
# ----------------------------------------------------------------------------
ZeroStuffed       = np.zeros(len(Input20MHz) * 4, np.float32)
ZeroStuffed[0::4] = Input20MHz 

# Push the ZeroStuffed signal through the FIR filter
Interpolated1     = np.convolve(h_Interpolator, ZeroStuffed)


# ----------------------------------------------------------------------------
# > 3b. Use the NPX method (Same mathematics as 3a, but different implementation)
# ----------------------------------------------------------------------------
#
# The NXP method is exactly the same as the zero-stuffing method. It does the same calculations
# but arranges them differently. If you draw some pictures on a piece of paper and look hard enough,
# you can see it in your head. Admittedly, it took me a while.
#
Coeff_Set0      = h_Interpolator[0::4]   # Start at element 0 with steps = 4 until the end of array
Coeff_Set1      = h_Interpolator[1::4] 
Coeff_Set2      = h_Interpolator[2::4]
Coeff_Set3      = h_Interpolator[3::4]

OutputA         = np.convolve(Coeff_Set0, Input20MHz)  # They likely run these convolutions 
OutputB         = np.convolve(Coeff_Set1, Input20MHz)  # in parallel, which is a lot faster
OutputC         = np.convolve(Coeff_Set2, Input20MHz)  # that convolving the way it is
OutputD         = np.convolve(Coeff_Set3, Input20MHz)  # done with the single convolution of the zero stuffed signal.

# Let's stick these individual arrays into a matrix. Each individual sequence is now one row in the matrix
Interpolated2   = np.vstack([OutputA, OutputB, OutputC, OutputD])

# With the debugger look inside Output2 and realize that we just need to read out the values one column at a time.
# Meaning, we read it out [Output2[0,0], Output2[0,1], Output2[0,2], Output2[0,3], Output2[1,0], Output2[1,1], ... etc]
Interpolated2   = Interpolated2.transpose().flatten()

print('Interpolator Coeff Set0: ' + str(Coeff_Set0))
print('Interpolator Coeff Set1: ' + str(Coeff_Set1))
print('Interpolator Coeff Set2: ' + str(Coeff_Set2))
print('Interpolator Coeff Set3: ' + str(Coeff_Set3))
print(' ---------------------------------------------------------')
for Index, Coeff in enumerate(h_ImageReject):
    print('h_ImageReject[' + str(Index) + '] = ' + str(h_ImageReject[Index]))

# ----------------------------------------------------------
# > 4. Run the Image Rejection Filter
# ----------------------------------------------------------
OutputFinal    = np.convolve(h_ImageReject, Interpolated1)


plt.figure(1)
plt.subplot(4,1,1)
plt.plot(range(0, len(Input20MHz)), Input20MHz, 'k-o')
plt.plot(np.arange(0, len(Input20MHz), 0.25), Ideal80MHz, 'b')
plt.legend(['Actual 20MHz Input', 'Perfect Output'])
plt.grid('#cccccc')
plt.title('Input Signal Sampled at 20MHz')
plt.tight_layout()
plt.subplot(4,1,2)
plt.plot(ZeroStuffed, 'k-o')
plt.grid('#cccccc')
plt.title('Zero-Stuffed Input signal at 80MHz')
plt.tight_layout()
plt.subplot(4,1,3)
plt.plot(Interpolated1, 'r-o')
plt.plot(Interpolated2, 'b-d')
plt.grid('#cccccc')
plt.title('Interpolated Signal at 80MHz')
plt.legend(['Using zero Stuffing', 'NXP Method'])
plt.tight_layout()
plt.subplot(4,1,4)
plt.plot(OutputFinal, 'k')
plt.grid('#cccccc')
plt.title('Final Signal After Image Rejection at 80MHz')
plt.tight_layout()
plt.show()


 

SpectrumAnalyzer( OutputFinal 
                , SampleRate2     
                , 64
                , True)