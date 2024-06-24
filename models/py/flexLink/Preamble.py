# File:       Preamble.py
# Notes:      This file generates all sequences of the preamble including the AgcBurst, PreambleA, and PreambleB
# Goals:      AgcBurst: The goal of the AgcBurst is to provide an ideal signal for the AGC to lock.
#                       The signal is ideal if it has a wide bandwidth and a low PAP allowing the AGC to detect 
#                       magnitude as fast as possible.
#                       Length = 5usec = math.floor(5e-6 * SampleRate)            
# 
#             PreambleA: The goal of the PreambleA is to allow the receiver to detect the packet and resolve the frequency
#                        offset. For the ~20KHz subcarrier spacing, a frequency offset of 200Hz will likely create a lower limit
#                        on the EVM of around -35dB as loss of orthogonality starts to insert interference into the bins.
#                        We will try to develope the frequency offset detection algorithm such that the frequency offset can 
#                        be resolved as follows:
#                        -> High SNR cases (>20dB)         --> +/- 200Hz
#                        -> Mid1 SNR case  (10dB - 20dB)   --> +/- 300Hz
#                        -> Mid2 SNR case  (0dB  - 10dB)   --> +/- 400Hz
#                        -> Low SNR  case  (<0dB)          --> +/- 500Hz 
#                        Length ~ 250usec = math.floor(250e-6 * SampleRate) for long PreambleA for packet and frequency offset detection
#                        Length ~ 50usec  = math.floor(50e-6  * SampleRate) for short PreambleB for packet detection only
#
#              PreambleB: The goal of the PreambleB is to allow timing acquisition in the receiver.
#                         The ideal waveform is one with excellent auto-correlation properties. The Zadoff-Chu sequence
#                         is perfect as I can set the bandwidth. The low PAP property is not that important here.
#                         Length = FFT_Size / SampleRate (An OFDM symbol without the CP) 
#                         
# Nots PreambleA:  For the SampleRate of Fs = 20MHz, the PreambleA shall be defined as
#                  PreambleA = cos(2pi * F1 * n/Fs + pi/4) + cos(2pi * F2 * n/Fs + pi/4)
#                  F1 = 32*SubcarrierSpacing ~ 625KHz    
#                  F2 = 96*SubcarrierSpacing ~ 1.875MHz 
#                  This provides 4 complex sinusoid, each of which or all can be used to estimate the frequency offset.
#                  -> Four sinusoids are provided in case any of them are attenuated due to frequency selective fading.
#                  -> The frequencies are somewhat low as to allow for lowpass filtering ahead of the detection process.


__title__     = "Preamble"
__author__    = "Andreas Schwarzinger"
__status__    = "preliminary"
__date__      = "April, 24th, 2024"
__copyright__ = 'Andreas Schwarzinger'

import os
import sys                               # We use sys to include new search paths of modules
OriginalWorkingDirectory = os.getcwd()   # Get the current directory
DirectoryOfThisFile      = os.path.dirname(__file__)   # FileName = os.path.basename
if DirectoryOfThisFile != '':
    os.chdir(DirectoryOfThisFile)        # Restore the current directory

# There are modules that we will want to access and they sit two directories above DirectoryOfThisFile
sys.path.append(DirectoryOfThisFile + "\\..\\..\\DspComm")

from   SignalProcessing  import *
import numpy             as np
import math
import matplotlib.pyplot as plt
from   FilterDesigner2   import CFilterDesigner   




# ---------------------------------------------------------------------------------------------------- #
# > Generate the AGC Burst                                                                             #
# ---------------------------------------------------------------------------------------------------- #
def GenerateAgcBurst(SampleRate: float = 20.00e6
                   , bPlot:      bool  = False) -> np.ndarray:
    '''
    This function generates the AGC burst. The Zadoff-Chu sequence is great for this application. 
    I can set a high bandwidth and maintain a low PAP. Exactly what an AGC wants to see.
    '''
    # Type checking
    assert isinstance(SampleRate, float) or isinstance(SampleRate, int)
    assert SampleRate == 20e6 or SampleRate == 40e6

    # The AGC burst is 5 microseconds in length.
    NumberOfSamplesToRetain = math.floor(5e-6 * SampleRate)

    # AgcBurst Generation
    Nzc        = 887 
    ScPositive = math.ceil(Nzc/2)
    ScNegative = math.floor(Nzc/2)
    u1         = 34
    n          = np.arange(0, Nzc, 1, np.int16)
 
    # Definition of the Zadoff-Chu Sequence
    zc         = np.exp(-1j*np.pi*u1*n*(n+1)/Nzc)

    if bPlot == True:
        plt.figure(1)
        plt.plot(zc.real, 'r', zc.imag, 'b')
        plt.grid(True)
        plt.title('Original')

    # The Fourier Transform of the Zadoff-Chu Sequence
    AgcBurst_FFT = np.fft.fft(zc); 

    # ------------------------------------
    # Mapping into N = 1024 or 2048 IFFT buffer and render into time domain
    # ------------------------------------
    if SampleRate == 20e6:
         IFFT_Buffer = np.zeros(1024, np.complex128)
    else:
         IFFT_Buffer = np.zeros(2048, np.complex128)

    IFFT_Buffer[0:ScPositive]  = AgcBurst_FFT[0:ScPositive]          
    IFFT_Buffer[-ScNegative:]  = AgcBurst_FFT[-ScNegative:]  
    IFFT_Buffer[0]             = 0
    AgcBurstFull               = (len(IFFT_Buffer)/Nzc)*np.fft.ifft(IFFT_Buffer)

    AgcBurst  = AgcBurstFull[0:NumberOfSamplesToRetain]

    if bPlot == True:
        plt.figure(2)
        plt.plot(AgcBurst.real, 'r', AgcBurst.imag, 'b')
        plt.grid(True)
        plt.title('Rendered')
        plt.show()

    #InstPower = AgcBurst * np.conj(AgcBurst) 
    #Pap       = 10 * np.log10(np.max(InstPower) / np.mean(InstPower)).real

    return AgcBurst





# ----------------------------------------------------------------------------------------------------------------- #
# > GeneratePreambleA()                                                                                             #
# ----------------------------------------------------------------------------------------------------------------- #
def GeneratePreambleA(SampleRate: float = 20.0e6
                    , strMode:    str = 'long') -> np.ndarray:
    """
    brief:  This function generates the PreambleA Waveform. At the receiver, just multiply the incoming 
            : preambleA sequence by a Hamming window and take the 4096 point FFT. Now reconstruct the four
            : different complex sinusoids and detect their frequency offst.
    param:  SampleRate - an integer or floating point number with value 19.2MHz/20MHz/20.48MHz
    param:  strMode    - The strMode determines the length of the PreambleA
    """

    # Type and error checking
    assert isinstance(SampleRate, int) or isinstance(SampleRate, float)
    assert SampleRate == 20e6 or SampleRate == 40e6
    assert isinstance(strMode, str)
    assert strMode.lower() == 'long' or strMode.lower() == 'short'

    # Select the mode (Length of the preambleA)
    if strMode == 'long':
        DurationSec = 250e-6    # 250 microseconds 
    else:
        DurationSec = 50e-6     # 50 microseconds (enough time for packet detection)

    SubcarrierSpacing = 20e6/1024 
    CosineFrequencyA  = 32*SubcarrierSpacing # Approximately  625000 Hz 
    CosineFrequencyB  = 96*SubcarrierSpacing # Approximately 1875000 Hz  

    Ts           = 1/SampleRate
    NumSamples   = math.floor(DurationSec / Ts)

    Time  = np.arange(0, NumSamples*Ts, Ts, dtype = np.float64)
    Angle = np.pi / 4
    Tone1 = np.exp( 1j*(2*np.pi*CosineFrequencyA*Time + Angle), dtype = np.complex64) 
    Tone2 = np.exp( 1j*(2*np.pi*CosineFrequencyB*Time + 3*Angle), dtype = np.complex64) 
    Tone3 = np.exp(-1j*(2*np.pi*CosineFrequencyA*Time + Angle), dtype = np.complex64) 
    Tone4 = np.exp(-1j*(2*np.pi*CosineFrequencyB*Time + 3*Angle), dtype = np.complex64) 

    PreambleA = (1/2) * (Tone1 + Tone2 + Tone3 + Tone4)

    return PreambleA





# ----------------------------------------------------------------------------------------------------------- #
# > GeneratePreambleB()                                                                                       #
# ----------------------------------------------------------------------------------------------------------- #
def GeneratePreambleB(SampleRate: float = 20.0e6) -> np.ndarray:

    assert isinstance(SampleRate, int) or isinstance(SampleRate, float)
    assert SampleRate == 20e6 or SampleRate == 40e6

    # PreambleB Generation
    Nzc        = 331 
    ScPositive = math.ceil(Nzc/2)
    ScNegative = math.floor(Nzc/2)
    u1         = 34
    n          = np.arange(0, Nzc, 1, np.int16)
 
    # Definition of the Zadoff-Chu Sequence
    zc         = np.exp(-1j*np.pi*u1*n*(n+1)/Nzc)

    # The Fourier Transform of the Zadoff-Chu Sequence
    PreambleB_FFT = np.fft.fft(zc); 

    # ------------------------------------
    # Mapping into N = 1024 or 2048 IFFT buffer and render into time domain
    # ------------------------------------
    if SampleRate == 20e6:
         IFFT_Buffer = np.zeros(1024, np.complex128)
    else:
         IFFT_Buffer = np.zeros(2048, np.complex128)

    IFFT_Buffer[0:ScPositive]  = PreambleB_FFT[0:ScPositive]          
    IFFT_Buffer[-ScNegative:]  = PreambleB_FFT[-ScNegative:]  
    IFFT_Buffer[0]             = 0
    PreambleB                  = (len(IFFT_Buffer)/Nzc)*np.fft.ifft(IFFT_Buffer)

    return PreambleB























# --------------------------------------------------------------
# > DetectPreambleA()
# --------------------------------------------------------------
def DetectPreambleA(RxPreambleA:       np.ndarray
                  , SampleRate:        float = 20.48e6
                  , bShowPlot:         bool = False) -> bool:
    
    # Error checking
    assert isinstance(RxPreambleA, np.ndarray)
    assert isinstance(SampleRate, float) or isinstance(SampleRate, int) 
    assert isinstance(bShowPlot, bool)

    # Generate Halfband filter
    #N    = 15;                                                   # Number of taps
    #n    = np.arange(0, N, 1, dtype = np.int32) 
    #Arg  = n/2 - (N-1)/4;                                        # Argument inside sinc function
    #Hann = np.ones(N, np.float32) - np.cos(2*np.pi*(n+1)/(N+1)); # The Hanning window  
                                                                  
    # Half Band Filter impulse response
    #h   = np.sinc(Arg)*Hann; 
    #h   = h/sum(h);                                              # normalize to unity DC gain

    #CFilterDesigner.ShowFrequencyResponse(h
    #                                    , SampleRate = 20e6
    #                                    , OversamplingRatio = 4
    #                                    , bShowInDb  = True
    #                                    , bShowIqVersion  = False)   
    

    # Test the frequency sampling method
    GainListLinear = [.0001, .0001,     1,    1,  .0001, .0001 ]
    NormalizedFreq = [-0.4,   -0.16, -0.13, 0.13,    0.16,    0.5]
    N              = 31
    ImpulseResponse = CFilterDesigner.FrequencySampling(GainListLinear, NormalizedFreq, N,  False)

    #CFilterDesigner.ShowFrequencyResponse(ImpulseResponse   = ImpulseResponse
    #                                     , SampleRate        = 20.0e6
    #                                     , OversamplingRatio = 32
    #                                     , bShowInDb         = False)

    

    # Filter the RxPreambleA
    FilteredPreambleA          = np.convolve(RxPreambleA, ImpulseResponse)
    # FilteredPreambleA          = np.convolve(RxPreambleA, h)

    # Some definitions
    FftSize                    = 1024
    SubcarrierSpacing          = SampleRate / FftSize
    PeriodicityHz              = 32*SubcarrierSpacing  
    PeriodSamples              = int(SampleRate / PeriodicityHz)  # = int(32.0)
    RxLength                   = len(FilteredPreambleA)  
    IntegrationLengthInSamples = 512
    
    CurrentCovariance          = np.zeros(RxLength - PeriodSamples, FilteredPreambleA.dtype)
    CurrentVariance            = np.ones (RxLength - PeriodSamples, FilteredPreambleA.dtype)
    Ratio                      = np.zeros(RxLength - PeriodSamples, np.float64)

    # Computing the covariance, variance and their ratio
    for Index in range(0, RxLength - PeriodSamples):
        A = FilteredPreambleA[Index + PeriodSamples]
        B = FilteredPreambleA[Index]
        CurrentCovariance[Index]     = CurrentCovariance[Index - 1] + A * np.conj(B)
        CurrentVariance[Index]       = CurrentVariance[Index - 1]   + A * np.conj(A)

        if Index >= IntegrationLengthInSamples:
            A = FilteredPreambleA[Index + PeriodSamples - IntegrationLengthInSamples]
            B = FilteredPreambleA[Index - IntegrationLengthInSamples]
            CurrentCovariance[Index]    -= A * np.conj(B)
            CurrentVariance[Index]      -= B * np.conj(B)

        Ratio[Index] = np.abs(CurrentCovariance[Index]) / np.abs(CurrentVariance[Index])
        if Index < IntegrationLengthInSamples:
            Ratio[Index] = 0

    # ---------------------------------------------
    # Did we see the packet?
    # ---------------------------------------------
    # We declare a detected PreambleA if we see 200 consecutive samples larger than 0.2
    DetectionSuccess   = False
    DetectionThreshold = 0.18   
    Range              = 200
    Count              = 0
    DetectionStrength  = 0
    for Index, CurrentRatio in enumerate(Ratio):
        if CurrentRatio > DetectionThreshold:
            Count             += 1
            DetectionStrength += CurrentRatio/Range
            if Count == Range:
                 DetectionSuccess = True
                 break
        else:
             Count             = 0
             DetectionStrength = 0
                               

    if bShowPlot == True:
        plt.figure(5)
        plt.subplot(3, 1, 1)
        plt.plot(np.arange(0, len(CurrentCovariance)), np.abs(CurrentCovariance), 'k')
        plt.title('The absolute value of the Covariance')
        plt.tight_layout()
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(np.arange(0, len(CurrentVariance)), np.abs(CurrentVariance), 'k')
        plt.title('The absolute value of the Variance')
        plt.tight_layout()
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(np.arange(0, len(Ratio)), Ratio, 'k')
        plt.title('The Ratio of absolute values of the Covariance and Variance')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return (DetectionSuccess, DetectionStrength)


# --------------------------------------------------------------
# > ProcessPreambleA()
# --------------------------------------------------------------
def ProcessPreambleA(RxPreambleA:       np.ndarray
                   , SampleRate:        float = 20.48e6
                   , bHighCinr:         bool  = False
                   , bShowPlots:        bool  = False) -> float:
    """
    This function will attempt to guess at the frequency offset in the PreambleA Waveform
    """
    # F1 = 4*160KHz  * (Fs / 20.48MHz) = 640KHz  * (Fs / 20.48MHz)
    # F2 = 12*160KHz * (Fs / 20.48MHz) = 1.92MHz * (Fs / 20.48MHz) 
    
    # F1/F2 = 640KHz/1.92MHz  if Fs = 20.48MHz
    # F1/F2 = 625KHz/1.875MHz if Fs = 20.00MHz


    # ------------------------
    # Error checking
    # ------------------------
    FFT_Size    = 4096  
    
    assert np.issubdtype(RxPreambleA.dtype, np.complexfloating), 'The RxWaveform input argument must be a complex numpy array.'
    assert len(RxPreambleA.shape) == 1,                          'The RxWaveform must be a simple np array.'
    assert len(RxPreambleA) >= FFT_Size,                         'The RxWavefom must feature at least FFT_Size entries.'
    assert isinstance(bHighCinr, bool),                          'The bHighCinr input argument must be of type bool.'
    assert isinstance(bShowPlots, bool),                         'The bShowPlots input arguement must be of type bool.'

    # ------------------------
    # Overlay the Hanning Window to induce more FFT leakage close by and suppress it further away
    # ------------------------
    N              = FFT_Size
    n              = np.arange(0, N, 1, dtype = np.int32)
    Hanning        = 0.5 - 0.5 * np.cos(2*np.pi * (n + 1) / (N + 1))
    RxWaveform     = RxPreambleA[0:N].copy() * Hanning

    # ------------------------------
    # Take the FFT of the first FFT_Size samples and rearrange it such that negative frequency bins are first
    # ------------------------------
    ZeroIndex      = int(FFT_Size/2)
    FFT_Output     = np.fft.fft(RxWaveform[0:FFT_Size])
    FFT_Rearranged = np.hstack([FFT_Output[ZeroIndex : ], FFT_Output[:ZeroIndex]]) 

    if(False):
        plt.figure()
        xindices = np.arange(-2048, 2048, 1, np.int32) * SampleRate / 4096
        plt.plot(xindices, np.abs(FFT_Rearranged), 'k')
        plt.grid(True)
        plt.title('FFT Rearranged')
        plt.show()

    # ------------------------------
    # Find all peak bin indices
    # ------------------------------
    MaxIndex        = np.argmax(abs(FFT_Rearranged))
    OffsetIndex     = MaxIndex - ZeroIndex
    # PeakIndexDeltas is the distance in bins of all peaks relative to the most negative peak index
    PeakIndexDeltas = np.array([0, 256, 512, 768])   

    # Now that we have the peak, we need to figure out the indices of the other peaks
    if   OffsetIndex <  -320:                        # Then the maximum peak is the one belonging to -1920MHz
            PeakIndices = MaxIndex + PeakIndexDeltas 
    elif OffsetIndex >= -320 and OffsetIndex  < 0:   # Then the maximum peak is the one belonging to  -640MHz
            PeakIndices = MaxIndex + PeakIndexDeltas - PeakIndexDeltas[1]
    elif OffsetIndex <=  320 and OffsetIndex >= 0:   # Then the maximum peak is the one belonging to  +640MHz
            PeakIndices = MaxIndex + PeakIndexDeltas - PeakIndexDeltas[2]
    elif OffsetIndex >   320:                        # Then the maximum peak is the one belonging to +1920MHz
            PeakIndices = MaxIndex + PeakIndexDeltas - PeakIndexDeltas[3]


    # ------------------------------
    # Prior to the maximum combining, we need to twist the peaks and its immediate neighboring bins 
    # ------------------------------        
    # 1. Find the rotation to twist each of the four peaks back to 1.0 + 0j. This gives us four values.
    # 2. Use each value to derotate the corresponding peak and its immediate neighboring bins.
    PeakAngles = np.angle(FFT_Rearranged[PeakIndices])
    Derotation = np.exp(-1j*PeakAngles)

    for Count, PeakIndex in enumerate(PeakIndices):
         Range = np.arange(PeakIndex - 5, PeakIndex + 5, 1, np.int32)
         FFT_Rearranged[Range] *= Derotation[Count]


    # ------------------------------
    # Use maximum ratio combining to compute the frequency offset
    # ------------------------------
    MRC_Scaling = (FFT_Rearranged[PeakIndices] * np.conj(FFT_Rearranged[PeakIndices])).real

    Sum0 = np.sum(FFT_Rearranged[PeakIndices - 3] * MRC_Scaling)
    Sum1 = np.sum(FFT_Rearranged[PeakIndices - 2] * MRC_Scaling)
    Sum2 = np.sum(FFT_Rearranged[PeakIndices - 1] * MRC_Scaling)
    Sum3 = np.sum(FFT_Rearranged[PeakIndices + 0] * MRC_Scaling)
    Sum4 = np.sum(FFT_Rearranged[PeakIndices + 1] * MRC_Scaling)
    Sum5 = np.sum(FFT_Rearranged[PeakIndices + 2] * MRC_Scaling)
    Sum6 = np.sum(FFT_Rearranged[PeakIndices + 3] * MRC_Scaling)

    # Note that the sample rate of 20.48MHz / 4096 results in tone spacing of 5KHz. If the frequency offset
    # is beyond 5 KHz, then we must adjust the recreating frequencies below. If the frequency offset is
    # less than 2.5KHz away, then the peaks will be at [1664, 1920, 2176, 2432]
    SubcarrierOffsetA = PeakIndices[0] - 1664
    SubcarrierOffset  = 0

    n = np.array([(2048 - 256), (2048 + 256)])
    N = FFT_Size
    if bHighCinr == False:
        Sum1 = 0
        Sum5 = 0
        Tone = np.exp( 1j*2*np.pi*n*(SubcarrierOffset-2)/N)   * Sum1 + \
               np.exp( 1j*2*np.pi*n*(SubcarrierOffset-1)/N)   * Sum2 + \
               np.exp( 1j*2*np.pi*n*(SubcarrierOffset  )/N)   * Sum3 + \
               np.exp( 1j*2*np.pi*n*(SubcarrierOffset+1)/N)   * Sum4 + \
               np.exp( 1j*2*np.pi*n*(SubcarrierOffset+2)/N)   * Sum5
    else:
        Tone = np.exp( 1j*2*np.pi*n*(SubcarrierOffset-3)/N)   * Sum0 + \
               np.exp( 1j*2*np.pi*n*(SubcarrierOffset-2)/N)   * Sum1 + \
               np.exp( 1j*2*np.pi*n*(SubcarrierOffset-1)/N)   * Sum2 + \
               np.exp( 1j*2*np.pi*n*(SubcarrierOffset  )/N)   * Sum3 + \
               np.exp( 1j*2*np.pi*n*(SubcarrierOffset+1)/N)   * Sum4 + \
               np.exp( 1j*2*np.pi*n*(SubcarrierOffset+2)/N)   * Sum5 + \
               np.exp( 1j*2*np.pi*n*(SubcarrierOffset+3)/N)   * Sum6

    Rotation1   = Tone[1] * np.conj(Tone[0])
    
    AngleRad    = np.angle(Rotation1)
    AngleCycles = AngleRad / (2*np.pi)
    FreqOffset  = AngleCycles * SampleRate / 512

    FreqOffset  += SubcarrierOffsetA * SampleRate / 4096

    if bShowPlots:
        print('Frequency Offset = ' + str(FreqOffset) + ' Hz')
        print('MaxIndex =    ' + str(MaxIndex))
        print('OffsetIndex = ' + str(OffsetIndex)) 
        print(PeakIndices)

        plt.figure(1)
        plt.stem(np.arange(0, len(FFT_Rearranged)), np.abs(FFT_Rearranged))
        plt.grid(True)
        plt.show()

    return FreqOffset
    
    
  



































# ------------------------------------------------------------
# The test bench
# ------------------------------------------------------------
if __name__ == '__main__':
      

    Test = 1          # 0 - Test the AGC Burst performance
                      # 1 - Spot test of the        PreambleA detection performance 
                      # 2 - Monte Carlo test of the PreambleA detection performance
                      # 3 - Spot test of the        PreambleA frequency detector 
                      # 4 - Monte Carlo test of the PreambleA frequency detector
    
    # -----------------------------------------------------------------------------------------------
    if Test == 0:

        D = GenerateAgcBurst()
        MaxD = max(np.abs(D))
        Pap = 10*np.log10( (MaxD**2) / np.var(D) )
        MeanSquare = np.mean(D * np.conj(D)).real

        print('Pap (dB):   ' + str(Pap))
        print('MeanSquare: ' + str(MeanSquare))

        plt.figure(1)
        plt.plot(np.abs(D), c = 'red')
        plt.grid(True)
        plt.show()



    # -----------------------------------------------------------------------------------------------
    if Test == 1:
        # -------------------------------------
        # Spot Checking Analysis
        # -------------------------------------

        # -------------------------------------
        # Generate the PreambleA sequence 
        # -------------------------------------
        ReferenceFrequency = 20e6
        ReferencePeriod    = 1/ReferenceFrequency
        TxPreambleA        = GeneratePreambleA(SampleRate = ReferenceFrequency
                                              , strMode    = 'long')
        
        MeanSquare         = np.mean(TxPreambleA * np.conj(TxPreambleA)).real
        Time               = np.arange(0, len(TxPreambleA), 1) * ReferencePeriod 

        plt.figure()
        plt.plot(TxPreambleA.real, 'r', TxPreambleA.imag, 'b')
        plt.title('Ideal PreambleA')
        plt.grid(True)
        plt.tight_layout()   
        plt.show()     

        # ------------------------
        # Add Multipath
        TxPreambleMp =  AddMultipath(InputSequence = TxPreambleA
                                    , SampleRate    = ReferenceFrequency
                                    , Delays        = [int(20e6*0), int(20e6*1.25e-6), int(20e6*2.25e-6), int(20e6*3.25e-6)]  
                                    , Constants     = [.7+0.72j, -0.50, 0.2j, 0.1j]   
                                    , Dopplers      = [10000000, 0, -0, 0])[0]
        SignalPower = np.mean(TxPreambleMp * np.conj(TxPreambleMp)).real 

        plt.figure(1)
        plt.subplot(2,1,1)
        plt.plot(TxPreambleA.real, 'r', TxPreambleA.imag, 'b')
        plt.title('Ideal PreambleA')
        plt.grid(True)
        plt.tight_layout()


        # ------------------------
        # Add noise to the waveform
        SNRdB            = 0
        SNR_Linear       = 10**(SNRdB/10)
        NoisePower       = SignalPower / SNR_Linear
        Noise            = GenerateAwgn(NumberOfSamples = len(TxPreambleMp) + 500  
                                      , Type            = 'complex64'
                                      , Variance        = NoisePower) 
        
        RxPreambleANoisy        = Noise.copy()
        RxPreambleANoisy[500:] += TxPreambleMp      

        #plt.subplot(2,1,2)
        #plt.plot(RxPreambleANoisy.real, 'r', RxPreambleANoisy.imag, 'b')
        #plt.grid(True)
        #plt.title('Distorted and Noisy PreambleA')
        #plt.tight_layout()
        #plt.show()
    
        # ------------------------
        # Run the packet detector 
        DetectionSuccess, DetectionStrength = DetectPreambleA(RxPreambleANoisy, ReferenceFrequency, True)
        
        if DetectionSuccess == True:
             print('PreambleA detected with strength: ' + str(DetectionStrength))
        else:
             print('PreambleA not detected')



    # -----------------------------------------------------------------------------------------------
    if Test == 2:
        # -------------------------------------
        # Monte Carlo Analysis
        # -------------------------------------

        # -------------------------------------
        # Generate the PreambleA sequence 
        # -------------------------------------
        ReferenceFrequency = 20e6
        ReferencePeriod    = 1/ReferenceFrequency
        TxPreambleA        = GeneratePreambleA(SampleRate = ReferenceFrequency
                                             , FftSize    = 1024
                                             , strMode    = 'long') 
        
        Time        = np.arange(0, len(TxPreambleA), 1) * ReferencePeriod 
        
        # -------------------------------------
        # Set up the simulation
        # -------------------------------------
        # Defining the SNR list over which we will sweep the outer loop of the simulation
        List_SnrDb     = [  -20, -10, -5, 0, ]  

        # Defining the Mean Doppler frequencies over which to run the inner loop of the simulation
        NumIterations  = 20
        MainFreqOffset = np.random.randint(low=-20000, high=20000, size= NumIterations, dtype = np.int32)
        for SnrDb in List_SnrDb:
            
            for Iteration in range(0, len(MainFreqOffset)):
                # The sequence of 5 Doppler frequency, which we generate by assigning via 
                # a mean (MainFreqOffset[Iteration]) and a range which is set below.
                LocalFreqOffsets = MainFreqOffset[Iteration] + np.random.randint(low=-150, high=150, size = 5, dtype = np.int32)

                # We set the delay to between 0 and 5 microseconds
                Delays           = np.random.randint(low=0, high=500, size = 5, dtype = np.int32) * 1e-8
                DelayArray       = (Delays * ReferenceFrequency).astype(np.int32).tolist()
                

                # ------------------------
                # Add Multipath
                TxPreambleMp, MinDelay =  AddMultipath(InputSequence = TxPreambleA
                                                    , SampleRate    = ReferenceFrequency
                                                    , Delays        = DelayArray 
                                                    , Constants     = [1+0j, -0.5, 0.3j, 0.2j, -0.1]   
                                                    , Dopplers      = LocalFreqOffsets.tolist())
                
                SignalPower = np.mean(TxPreambleMp * np.conj(TxPreambleMp)).real 


                # ------------------------
                # Add noise to the waveform
                SNR_Linear       = 10**(SnrDb/10)
                NoisePower       = SignalPower / SNR_Linear
                Noise            = GenerateAwgn(NumberOfSamples = len(TxPreambleMp) + 500  
                                              , Type            = 'complex64'
                                              , Variance        = NoisePower) 
                
                RxPreambleANoisy        = Noise.copy()
                RxPreambleANoisy[500:] += TxPreambleMp      

            
                # ------------------------
                # Run the packet detector (We want to rerun the detector several times at different frequency offsets)
                DetectionSuccess, DetectionStrength = DetectPreambleA(RxPreambleANoisy, ReferenceFrequency, False)
                if DetectionSuccess == True:
                    Message = 'SNR (dB) = ' + str(SnrDb) + '  Passed:  Detection Strength = ' + str(DetectionStrength) + ' FreqOffset: ' + str(MainFreqOffset[Iteration])
                else:
                    Message = 'SNR (dB) = ' + str(SnrDb) + '  Failed.'         
                
                print(Message)




    # -----------------------------------------------------------------------------------------------
    if Test == 3:
        # -------------------------------------
        # Spot Checking Analysis
        # -------------------------------------

        # -------------------------------------
        # Generate the PreambleA sequence 
        # -------------------------------------
        SampleRate      = 20e6
        SamplePeriod    = 1/SampleRate
        TxPreambleA     = GeneratePreambleA(SampleRate = SampleRate
                                             , FftSize = 1024
                                             , strMode = 'long')
        
        Time        = np.arange(0, len(TxPreambleA), 1) * SamplePeriod     

        TxPreambleA_Offset =  AddMultipath(InputSequence = TxPreambleA
                                         , SampleRate    = SampleRate
                                         , Delays        = [int(20e6*0), int(20e6*1.25e-6), int(20e6*2.25e-6), int(20e6*3.25e-6)]  
                                         , Constants     = [.7+0.72j, -0.50, 0.2j, 0.1j]   
                                         , Dopplers      = [-1000, -1050, -950, -800])[0]

        if(False):
            # -------------------------------------
            # Introduce Frequency Error
            # -------------------------------------
            n                   = np.arange(0, len(TxPreambleA), 1, np.int32)
            FrequencyError      = 40000
            Sinusoid            = np.exp(1j*2*np.pi*n*FrequencyError/SampleRate) 
            TxPreambleA_Offset  = TxPreambleA * Sinusoid
        
        SignalPower         = np.mean(TxPreambleA_Offset * np.conj(TxPreambleA_Offset)).real

        # --------------------------------------
        # Introduce Gaussian Noise
        # --------------------------------------
        SnrDb            = -10
        SNR_Linear       = 10**(SnrDb/10)
        NoisePower       = SignalPower / SNR_Linear
        Noise            = GenerateAwgn(NumberOfSamples = len(TxPreambleA_Offset)  
                                      , Type            = 'complex64'
                                      , Variance        = NoisePower
                                      , Seed            = -1) 
        
        TxPreampleA_Noisy   = TxPreambleA_Offset + Noise

        HighSnrdB      = True
        FreqOffset     = ProcessPreambleA(TxPreampleA_Noisy, SampleRate, HighSnrdB, False)
        print(FreqOffset)





    # -----------------------------------------------------------------------------------------------
    if Test == 4:
        # -------------------------------------
        # Monte Carlo Analysis
        # -------------------------------------

        # -------------------------------------
        # Generate the PreambleA sequence 
        # -------------------------------------
        SampleRate      = 20e6
        SamplePeriod    = 1/SampleRate
        TxPreambleA     = GeneratePreambleA(SampleRate = SampleRate
                                             , FftSize = 1024
                                             , strMode = 'long')
        
        Time        = np.arange(0, len(TxPreambleA), 1) * SamplePeriod     

        TxPreambleA_Offset =  AddMultipath(InputSequence = TxPreambleA
                                         , SampleRate    = SampleRate
                                         , Delays        = [int(20e6*0), int(20e6*1.25e-6), int(20e6*2.25e-6), int(20e6*3.25e-6)]  
                                         , Constants     = [.7+0.72j, -0.50, 0.2j, 0.1j]   
                                         , Dopplers      = [4900, 4850, 4900, 4900])[0]

        if(False):
            # -------------------------------------
            # Introduce Frequency Error
            # -------------------------------------
            n                   = np.arange(0, len(TxPreambleA), 1, np.int32)
            FrequencyError      = 40000
            Sinusoid            = np.exp(1j*2*np.pi*n*FrequencyError/SampleRate) 
            TxPreambleA_Offset  = TxPreambleA * Sinusoid
        
        SignalPower         = np.mean(TxPreambleA_Offset * np.conj(TxPreambleA_Offset)).real


        # --------------------------------------------------
        # Set up the simulation
        # --------------------------------------------------

        List_SnrDb   = [ -10, -5, 0, 5, 10, 40]                 # Sweep the simulation over these SNR     
        NumRuns      = 1000                                     # The number of runs in this monte carlo simulation
        Output       = np.zeros([len(List_SnrDb), NumRuns], np.float32)

        for SnrIndex, SnrDb in enumerate(List_SnrDb):
            print('Snr: ' + str(SnrDb))
            for Run in range(0, NumRuns):
                # Add noise to the TxPreambleA
                RxPreambleANoisy = AddAwgn(SnrDb, TxPreambleA_Offset, -1)    
               
                # Run the frequency estimator
                if SnrDb >= 10:
                     HighSnrdB = True
                else:
                     HighSnrdB = False     
                Output[SnrIndex, Run] = ProcessPreambleA(RxPreambleANoisy, SampleRate, HighSnrdB, False)            

                Stop = 1        

        # Plot a 3x2 figure showing histograms indicating the frequency estimator results
        plt.subplot(3, 2, 1)
        Results = Output[0,:]
        n, bins, patches = plt.hist(Results, 10, (np.min(Results), np.max(Results)), density=True, facecolor='g', alpha=0.5)
        plt.title('Histogram at SNR (dB) = ' + str(List_SnrDb[0]) )
        plt.xlabel('Frequency Offset Estimate in Hz')
        plt.tight_layout()

        plt.subplot(3, 2, 2)
        Results = Output[1,:]
        n, bins, patches = plt.hist(Results, 10, (np.min(Results), np.max(Results)), density=True, facecolor='g', alpha=0.5)
        plt.title('Histogram at SNR (dB) = ' + str(List_SnrDb[1]) )
        plt.xlabel('Frequency Offset Estimate in Hz')
        plt.tight_layout()

        plt.subplot(3, 2, 3)
        Results = Output[2,:]
        n, bins, patches = plt.hist(Results, 10, (np.min(Results), np.max(Results)), density=True, facecolor='g', alpha=0.5)
        plt.title('Histogram at SNR (dB) = ' + str(List_SnrDb[2]))
        plt.xlabel('Frequency Offset Estimate in Hz')
        plt.tight_layout()

        plt.subplot(3, 2, 4)
        Results = Output[3,:]
        n, bins, patches = plt.hist(Results, 10, (np.min(Results), np.max(Results)), density=True, facecolor='g', alpha=0.5)
        plt.title('Histogram at SNR (dB) = ' + str(List_SnrDb[3])  )
        plt.xlabel('Frequency Offset Estimate in Hz')
        plt.tight_layout()

        plt.subplot(3, 2, 5)
        Results = Output[4,:]
        n, bins, patches = plt.hist(Results, 10, (np.min(Results), np.max(Results)), density=True, facecolor='g', alpha=0.5)
        plt.title('Histogram at SNR (dB) = ' + str(List_SnrDb[4]) )
        plt.xlabel('Frequency Offset Estimate in Hz')
        plt.tight_layout()

        plt.subplot(3, 2, 6)
        Results = Output[5,:]
        n, bins, patches = plt.hist(Results, 10, (np.min(Results), np.max(Results)), density=True, facecolor='g', alpha=0.5)
        plt.title('Histogram at SNR (dB) = ' + str(List_SnrDb[5])  )
        plt.xlabel('Frequency Offset Estimate in Hz')
        plt.tight_layout()
        plt.show()