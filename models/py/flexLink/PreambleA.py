# File:       PreambleA.py
# Notes:      This file provides the code to generate and process the preamble A section of the FlexLink Waveform
# Goal:       The goal of the PreambleA is to allow the receiver to acquire the AGC lock and resolve the frequency
#             offset. For the 20KHz subcarrier spacing, a frequency offset of 200Hz will likely create an lower limit
#             on the EVM of around -35dB as loss of orthogonality starts to insert interference into the bins.
#             We will try to develope the frequency offset detection algorithm such that the frequency offset can 
#             be resolved as follows:
#             -> High SNR cases (>20dB)         --> +/- 200Hz
#             -> Mid1 SNR case  (10dB - 20dB)   --> +/- 300Hz
#             -> Mid2 SNR case  (0dB  - 10dB)   --> +/- 400Hz
#             -> Low SNR  case  (<0dB)          --> +/- 500Hz 
# Definition: Depending on the SampleRate = Fs (20MHz / 20.48MHz) available, the PreambleA shall be defines as
#             PreambleA = cos(2pi * F1 * t) + cos(2pi * F2 *t)
#             F1 = 4*160KHz  * (Fs / 20.48MHz) = 640KHz  * (Fs / 20.48MHz)
#             F2 = 12*160KHz * (Fs / 20.48MHz) = 1.92MHz * (Fs / 20.48MHz) 
#             This provides 4 complex sinusoid, each of which or all can be used to estimate the frequency offset.
#             Four sinusoids are provided in case any of them are attenuated due to frequency selective fading.
#
#             The PreambleA shall have a length of 225 microseconds, where
#             -> Less than 25 microseconds are used for AGC acquisition by the analog receiver.
#             -> 200 microseconds are used for frequency offset aquisition.

__title__     = "PreambleA"
__author__    = "Andreas Schwarzinger"
__status__    = "preliminary"
__date__      = "Sept, 16rd, 2022"
__copyright__ = 'Andreas Schwarzinger'

from   SignalProcessing  import *
import numpy             as np
import math
import matplotlib.pyplot as plt



# --------------------------------------------------------------
# > GeneratePreambleA()
# --------------------------------------------------------------
def GeneratePreambleA(SampleRate: float = 20.48e6) -> np.ndarray:
    """
    This function generates the PreambleA Waveform
    """
    CosineFrequencyA = 4*160e3
    CosineFrequencyB = 12*160e3

    Ts           = 1/SampleRate
    NumSamples   = math.floor(220e-6 / Ts)

    Time  = np.arange(0, NumSamples*Ts, Ts, dtype = np.float64)
    Tone1 = np.exp( 1j*2*np.pi*CosineFrequencyA*Time, dtype = np.complex64) 
    Tone2 = np.exp( 1j*2*np.pi*CosineFrequencyB*Time, dtype = np.complex64) 
    Tone3 = np.exp(-1j*2*np.pi*CosineFrequencyA*Time, dtype = np.complex64) 
    Tone4 = np.exp(-1j*2*np.pi*CosineFrequencyB*Time, dtype = np.complex64) 

    PreambleA = (1/4) * (Tone1 + Tone2 + Tone3 + Tone4)

    return PreambleA, Time


# --------------------------------------------------------------
# > DetectPreambleA()
# --------------------------------------------------------------
def DetectPreambleA(RxPreambleA:       np.ndarray
                  , SampleRate:        float = 20.48e6
                  , bShowPlot:         bool = False) -> float:
    
    PeriodicityHz              = 4*160e3
    PeriodSamples              = int(SampleRate / PeriodicityHz)
    RxLength                   = len(RxPreambleA)  
    IntegrationLengthInSamples = 400
    
    CurrentCovariance          = np.zeros(RxLength - PeriodSamples, RxPreambleA.dtype)
    CurrentVariance            = np.ones(RxLength - PeriodSamples, RxPreambleA.dtype)
    Ratio                      = np.zeros(RxLength - PeriodSamples, RxPreambleA.dtype)

    
    for Index in range(0, RxLength - PeriodSamples):
        CurrentCovariance[Index]     = CurrentCovariance[Index - 1] + RxPreambleA[Index + PeriodSamples] * np.conj(RxPreambleA[Index])
        CurrentVariance[Index]       = CurrentVariance[Index - 1]   + RxPreambleA[Index]                 * np.conj(RxPreambleA[Index])

        if Index >= IntegrationLengthInSamples:
            CurrentCovariance[Index]    -= RxPreambleA[Index + PeriodSamples - IntegrationLengthInSamples] * \
                                            np.conj(RxPreambleA[Index - IntegrationLengthInSamples])
            CurrentVariance[Index]      -= RxPreambleA[Index - IntegrationLengthInSamples]                 * \
                                            np.conj(RxPreambleA[Index - IntegrationLengthInSamples])

        Ratio[Index] = CurrentCovariance[Index] / CurrentVariance[Index]
        if Index < 100:
            Ratio[Index] = 0
            
    if bShowPlot == True:
        plt.figure(5)
        plt.subplot(3, 1, 1)
        plt.plot(np.arange(0, len(CurrentCovariance)), np.abs(CurrentCovariance))
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(np.arange(0, len(CurrentVariance)), np.abs(CurrentVariance))
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(np.arange(0, len(Ratio)), np.abs(Ratio))
        plt.grid(True)
        plt.show()


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
    # Use maximum ratio combining to compute the frequency offset
    # ------------------------------
    MRC_Scaling = FFT_Rearranged[PeakIndices] * np.conj(FFT_Rearranged[PeakIndices])

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
    SubcarrierOffset = PeakIndices[0] - 1664

    n = np.array([(2048 - 256), (2048 + 256)])
    N = FFT_Size
    if bHighCinr == False:
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
    TxPreambleA, Time = GeneratePreambleA()

    Test = 2

    if Test == 1:    
        # ------------------------
        # Add noise to the waveform
        SNRdB            = 50
        RxPreambleANoisy = AddAwgn(SNRdB, TxPreambleA)              
   
        # ------------------------
        # Add frequency Offset
        FreqOffsetHz = -250    
        Offset       = np.exp( 1j*2*np.pi*FreqOffsetHz*Time) 
        RxPreambleA  = RxPreambleANoisy * Offset

        # ------------------------
        # Run the frequency estimator
        FreqOffset = ProcessPreambleA(RxPreambleA, 20.48e6, True)

        plt.figure(2)
        plt.plot(Time, TxPreambleA.real, c = 'red')
        plt.plot(Time, TxPreambleA.imag, c = 'blue')
        plt.grid(True)
    

        plt.figure(3)
        plt.plot(Time, RxPreambleA.real, c = 'red')
        plt.plot(Time, RxPreambleA.imag, c = 'blue')
        plt.grid(True)
        plt.show()


    if Test == 2:
        # -------------------------------------
        # Monte Carlo Analysis
        # -------------------------------------
        List_SnrDb   = [ -5, 0, 5, 10, 20, 30]                   # Sweep the simulation over these SNR
        FreqOffsetHz = -00                                      # Run the simulation at this frequency offset
        Offset       = np.exp( 1j*2*np.pi*FreqOffsetHz*Time) 
        Offset       = np.hstack([Offset[0:500], Offset])

        # Multipath Channel conditions are static
        Delays        = [0, 0, 0]
        Constants     = [1 + 0j, 0, -10]  
        Dopplers      = [000, 0, -000]
        SampleRate    = 20.48e6
        
        NumRuns      = 100                                      # The number of runs in this monte carlo simulation
        FreqOffsets  = np.zeros([len(List_SnrDb), NumRuns], dtype = np.float32) 
        
        TxPreambleA = np.hstack([np.zeros(500, TxPreambleA.dtype), TxPreambleA])

        for SnrIndex, SnrDb in enumerate(List_SnrDb):
            print('Snr: ' + str(SnrDb))
            for Run in range(0, NumRuns):
                # Add noise to the TxPreambleA
                RxPreambleANoisy = AddAwgn(SnrDb, TxPreambleA)    
                # Add frequency offset  
                RxPreambleA      = RxPreambleANoisy * Offset
                # Add Multipath to the noise and frequency offset PreambleA
                Output, MinDelay = AddMultipath(RxPreambleA, SampleRate, Delays, Constants, Dopplers) 

                # Run the frequency estimator
                if SnrDb >= 10:
                     HighSnrdB = True
                else:
                     HighSnrdB = False     
                FreqOffsets[SnrIndex, Run] = ProcessPreambleA(Output, SampleRate, HighSnrdB, False)            

                DetectPreambleA(Output, SampleRate, True)    
                Stop = 1        

        # Plot a 3x2 figure showing histograms indicating the frequency estimator results
        plt.subplot(3, 2, 1)
        Results = FreqOffsets[0,:]
        n, bins, patches = plt.hist(Results, 10, (np.min(Results), np.max(Results)), density=True, facecolor='g', alpha=0.5)
        plt.title('Histogram at SNR (dB) = ' + str(List_SnrDb[0]) + ' at ' + str(FreqOffsetHz) + 'Hz' )
        plt.xlabel('Frequency Offset Estimate in Hz')
        plt.tight_layout()

        plt.subplot(3, 2, 2)
        Results = FreqOffsets[1,:]
        n, bins, patches = plt.hist(Results, 10, (np.min(Results), np.max(Results)), density=True, facecolor='g', alpha=0.5)
        plt.title('Histogram at SNR (dB) = ' + str(List_SnrDb[1]) + ' at ' + str(FreqOffsetHz) + 'Hz' )
        plt.xlabel('Frequency Offset Estimate in Hz')
        plt.tight_layout()

        plt.subplot(3, 2, 3)
        Results = FreqOffsets[2,:]
        n, bins, patches = plt.hist(Results, 10, (np.min(Results), np.max(Results)), density=True, facecolor='g', alpha=0.5)
        plt.title('Histogram at SNR (dB) = ' + str(List_SnrDb[2])+ ' at ' + str(FreqOffsetHz) + 'Hz' )
        plt.xlabel('Frequency Offset Estimate in Hz')
        plt.tight_layout()

        plt.subplot(3, 2, 4)
        Results = FreqOffsets[3,:]
        n, bins, patches = plt.hist(Results, 10, (np.min(Results), np.max(Results)), density=True, facecolor='g', alpha=0.5)
        plt.title('Histogram at SNR (dB) = ' + str(List_SnrDb[3]) + ' at ' + str(FreqOffsetHz) + 'Hz' )
        plt.xlabel('Frequency Offset Estimate in Hz')
        plt.tight_layout()

        plt.subplot(3, 2, 5)
        Results = FreqOffsets[4,:]
        n, bins, patches = plt.hist(Results, 10, (np.min(Results), np.max(Results)), density=True, facecolor='g', alpha=0.5)
        plt.title('Histogram at SNR (dB) = ' + str(List_SnrDb[4]) + ' at ' + str(FreqOffsetHz) + 'Hz' )
        plt.xlabel('Frequency Offset Estimate in Hz')
        plt.tight_layout()

        plt.subplot(3, 2, 6)
        Results = FreqOffsets[5,:]
        n, bins, patches = plt.hist(Results, 10, (np.min(Results), np.max(Results)), density=True, facecolor='g', alpha=0.5)
        plt.title('Histogram at SNR (dB) = ' + str(List_SnrDb[5]) + ' at ' + str(FreqOffsetHz) + 'Hz' )
        plt.xlabel('Frequency Offset Estimate in Hz')
        plt.tight_layout()
        plt.show()