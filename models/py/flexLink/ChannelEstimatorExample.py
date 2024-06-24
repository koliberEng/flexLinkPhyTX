# File:     ChannelEstimatorExample.py
# Notes:    In this file I will experiment with least squares to estimate a multipath channel


__title__     = "ChannelEstimatorExample"
__author__    = "Andreas Schwarzinger"
__status__    = "preliminary"
__date__      = "Oct, 28rd, 2022"
__copyright__ = 'Andreas Schwarzinger'

import numpy              as np
import matplotlib.pyplot  as plt
from   ChannelProcessing  import CMultipathChannel, CreateGaussianNoise
from   FlexLinkParameters import *
 

# --------------------------------------------------
# 1. Create and initialize a CMultipath Model
# --------------------------------------------------
SampleRate       = 20.48e6
MultipathChannel = CMultipathChannel()
Amplitudes       = [      -1j,     0+0j,  0.0 - .0j,     -0.0 + .0j]
DelaysInSec      = [  -200e-9,   000e-9,     600e-9,        1000e-9]
DopplersInHz     = [       200,     100,        400,           -500]

assert len(Amplitudes) == len(DelaysInSec) and len(Amplitudes) == len(DopplersInHz)
for Index in range(0, len(Amplitudes)):
    MultipathChannel.AddPath(Amplitudes[Index]
                           , DelaysInSec[Index]
                           , DopplersInHz[Index]   
                           , False   # Round to nearest sample
                           , SampleRate)    

# -------------------------------------
# 2. Create the list of resource elements at which DMRS are located
# -------------------------------------
ReType        = EReType.DmrsPort0
Sc            = ESubcarrierSpacing.Sc20KHz
Cp            = ECyclicPrefix.Ec4MicroSec
FreqUnitArray = list(range(0, 300, 3))
TimeUnitArray = [0]
DmrsList      = CResourceElement.CreateReArray(FreqUnitArray
                                             , TimeUnitArray
                                             , ReType
                                             , Sc
                                             , Cp)

# ----------------------------------------------
# 3. Compute the ideal frequency response and set the TX / RX Values of the RE
# ----------------------------------------------
MultipathChannel.ComputeFreqResponse(DmrsList)

# --------------------------
# Program the DMRS list with some fake TX values. Then compute the RX values 
# based on the ideal frequency response and then add noise to the received values
MeanSquare = 0
for RE in DmrsList:
    RE.TxValue = 1 + 0j
    RE.RxValue = RE.TxValue * RE.IdealFreqResponse
    MeanSquare += (1/len(DmrsList)) * (RE.RxValue * RE.RxValue.conj()).real 


# -------------------------------------------------
# 4. Add noise to the received values and compute the raw frequency response
# -------------------------------------------------
CinrdB           = 50
CinrLinear       = 10**(CinrdB/10)
LinearNoisePower = MeanSquare / CinrLinear
Noise            = CreateGaussianNoise(fLinearNoisePower = LinearNoisePower 
                                     , iNumberOfSamples = len(DmrsList) 
                                     , iSeed = -1
                                     , bComplexNoise = True)    

for Index, RE in enumerate(DmrsList):
    RE.RxValue += Noise[Index]
    RE.RawFreqResponse = RE.RxValue / RE.TxValue


IdealFreqResponse = [Dmrs.IdealFreqResponse for Dmrs in DmrsList]
RawFreqResponse   = [Dmrs.RawFreqResponse   for Dmrs in DmrsList]
EstFreqResponse   = [Dmrs.EstFreqResponse   for Dmrs in DmrsList]
FrequencyList     = [Dmrs.FrequencyHz       for Dmrs in DmrsList]


plt.figure(1)
plt.subplot(2,1,1)
plt.plot(FrequencyList, np.array(IdealFreqResponse).real, 'r')
plt.plot(FrequencyList, np.array(RawFreqResponse).real, 'r.')
plt.title('Frequency Response')
plt.legend(['Ideal', 'Raw'])
plt.xlabel('Hz')
plt.grid(True)
plt.tight_layout()
plt.subplot(2,1,2)
plt.plot(FrequencyList, np.array(IdealFreqResponse).imag, 'b')
plt.plot(FrequencyList, np.array(RawFreqResponse).imag, 'b.')
plt.title('Frequency Response')
plt.legend(['Ideal', 'Raw'])
plt.xlabel('Hz')       
plt.grid(True)
plt.tight_layout()




# ---------------------------------------------------------------
# 5. We will take the Fourier transform to find out at which frequency the sinusoid exists
# ---------------------------------------------------------------
FreqRespNpArray  = np.array(RawFreqResponse, np.complex64)
N                = len(FreqRespNpArray)
n                = np.arange(0, N, 1, np.uint16)
Hanning          = 0.5 - 0.5*np.cos(2*np.pi*(n+1)/(N+1))
FreqRespNpArrayH = Hanning * FreqRespNpArray 

FStep           = 1/N
FFT_Out         = np.fft.fft(FreqRespNpArrayH)
Frequencies     = np.arange(-0.5, 0.5, FStep, np.float32)
Frequencies     = np.hstack([Frequencies[-int(N/2):], Frequencies[:int(N/2)]])

plt.figure(2)
plt.stem(Frequencies, np.abs(FFT_Out), 'k')
plt.grid(True)
plt.tight_layout()


MaxIndex = np.argmax(np.abs(FFT_Out))
MaxFreq  = Frequencies[MaxIndex]
print('Max Frequency: ' + str(MaxFreq))

# ----------------------------------------------------------------
# 6. Let's try to recreate the sinusoid using the least squares method
# ----------------------------------------------------------------

H = np.reshape(FreqRespNpArray, (len(FreqRespNpArray),1))

# We will pick three frequencies close to the max
F1 = MaxFreq - 1/(2*N)
F2 = MaxFreq - 1/(4*N)
F3 = MaxFreq
F4 = MaxFreq + 1/(4*N)
F5 = MaxFreq + 1/(2*N)
h  = np.array([ F2, F3, F4], np.complex64)

h = np.reshape(h, (len(h), 1))
F = np.zeros([N, len(h)], np.complex64)

for Column in range(0, len(h)):
    F[:,Column] = np.exp(1j*2*np.pi*n*h[Column])

Fp    = F.conj().transpose()
Temp1 = Fp.dot(F)
Temp2 = np.linalg.inv(Temp1).dot(Fp)
A     = Temp2.dot(H)

Estimate = np.zeros(100, np.complex64)
for Index in range(0, len(A)):
    Estimate += A[Index] * np.exp(1j*2*np.pi*n*h[Column])



plt.figure(3)
plt.subplot(2,1,1)
plt.plot(n, FreqRespNpArray.real, 'r')
plt.plot(n, Estimate.real, 'r.')
plt.title('Frequency Response')
plt.legend(['Raw', 'Estimage'])
plt.grid(True)
plt.tight_layout()
plt.subplot(2,1,2)
plt.plot(n, FreqRespNpArray.imag, 'b')
plt.plot(n, Estimate.imag, 'b.')
plt.title('Frequency Response')
plt.legend(['Raw', 'Estimate'])     
plt.grid(True)
plt.tight_layout()
plt.show()
