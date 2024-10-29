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
import path_include     #local file to adjust library paths

# OriginalWorkingDirectory = os.getcwd()   # Get the current directory
# DirectoryOfThisFile      = os.path.dirname(__file__)   # FileName = os.path.basename
# if DirectoryOfThisFile != '':
#     os.chdir(DirectoryOfThisFile)        # Restore the current directory

# # There are modules that we will want to access and they sit two directories above DirectoryOfThisFile
# sys.path.append(DirectoryOfThisFile + "\\..\\..\\DspComm")
# sys.path.append(DirectoryOfThisFile + "\\..\\..\\..\\python\\KoliberEng")

import Visualization as vis
from   SignalProcessing  import *
import numpy             as np
import math
import matplotlib.pyplot as plt
from   FilterDesigner2   import CFilterDesigner   


vis = vis.Visualization()


# ---------------------------------------------------------------------------------------------------- #
#                                                                                                      #
#                                                                                                      #
# > Generating the Preamble                                                                            #
#                                                                                                      #
#                                                                                                      #
# ---------------------------------------------------------------------------------------------------- #

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
        plt.show()
        #plt.figure(2)
        vis.plot_constellation(zc)
        #vis.plot_psd(zc, 20.0e6, "Zadoff-Chu Sequence PSD")


    # The Fourier Transform of the Zadoff-Chu Sequence
    AgcBurst_FFT = np.fft.fft(zc); # in freq domain

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
    AgcBurstFull               = (len(IFFT_Buffer)/Nzc)*np.fft.ifft(IFFT_Buffer) # time domain, FFT length

    AgcBurst  = AgcBurstFull[0:NumberOfSamplesToRetain] # time domain 

    if bPlot == True:
        plt.figure(2)
        plt.plot(AgcBurst.real, 'r', AgcBurst.imag, 'b')
        plt.grid(True)
        plt.title('Rendered')
        plt.show()

        #InstPower = AgcBurst * np.conj(AgcBurst) 
        #Pap       = 10 * np.log10(np.max(InstPower) / np.mean(InstPower)).real
        vis.plot_data([list(IFFT_Buffer.real)], ["FFT w Freq shift"])
        vis.plot_data([list(AgcBurst)], ["Agc Burst"])
        vis.plot_psd(AgcBurstFull, SampleRate, True, "Agc Burst Full, time domain")
        vis.plot_psd(AgcBurst, SampleRate, True, "Agc Burst, Nsamp time domain")
        vis.plot_constellation(AgcBurst, name="AgcBurst time domain")

    return AgcBurst



# ---------------------------------------------------------------------------------------------------- #
# > AGC Burst tests
# ---------------------------------------------------------------------------------------------------- #
def GenerateAgcBurstTest(SampleRate: float = 20.00e6
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
    k          = 913   # total number of tones in RG, 0 to 912 inclusive
    #k          = 101
    
 
    # Definition of the Zadoff-Chu Sequence
    zc         = np.exp(-1j*np.pi*u1*n*(n+1)/Nzc)  # time domain 
    zc_buffer = np.zeros(1024, np.complex128)
    zc_buffer[0:len(zc)]  = zc          



    if bPlot == True:
        plt.figure(1)
        plt.plot(zc.real, 'r', zc.imag, 'b')
        plt.grid(True)
        plt.title('Original')
        
        plt.figure(2)
        plt.plot(zc.real, zc.imag, 'r.')
        plt.plot(zc.real, zc.imag, 'b')
        plt.grid(True)
        plt.title('IQ constellation')
        
        vis.plot_data([zc_buffer], ["zc sequence, zc buffer, time domain"],0,True)
        vis.plot_psd(zc_buffer, SampleRate, True, "zc buffer psd")
        

    # The Fourier Transform of the Zadoff-Chu Sequence
    #AgcBurst_FFT = np.fft.fft(zc)
    #AgcBurst_FFT = np.fft.fft(zc[0:NumberOfSamplesToRetain],1024)
    # FFT_Buffer2  = np.zeros(1024, np.complex128)
    # FFT_Buffer2[0:NumberOfSamplesToRetain]  = zc[0:NumberOfSamplesToRetain]

    
    FFT_Buffer = np.zeros(1024, np.complex128)
    FFT_Buffer[0:math.ceil(NumberOfSamplesToRetain/2)]  = zc[0:math.ceil(NumberOfSamplesToRetain/2)]          
    FFT_Buffer[-math.floor(NumberOfSamplesToRetain/2):]  = zc[-math.floor(NumberOfSamplesToRetain/2):]  
    FFT_Buffer[0]             = 0
    # FFT_Buffer[1]  = zc[0]          

    AgcBurst_FFT = np.fft.fft((FFT_Buffer))   # freq domain
    if bPlot == True: 
        vis.plot_data([np.fft.fftshift(FFT_Buffer)], ["zc sequence, FFT_Buffer, time domain"],0,True)
        vis.plot_psd(FFT_Buffer, SampleRate, True, "FFT buffer (td)")
        vis.plot_psd(np.fft.ifft(np.fft.fftshift(AgcBurst_FFT)), SampleRate, True, "AgcBurst FFT, ifft")

    
    
    # zero out the edge bins to limit bandwidth
    AgcBurst_FFT[0:math.ceil((1024-k)/2)] = 0
    AgcBurst_FFT[-(math.floor((1024-k)/2)):] = 0
    if bPlot == True: 
        vis.plot_data([AgcBurst_FFT], ["AgcBurst_FFT"],0,True)    
        vis.plot_psd(np.fft.ifft(np.fft.fftshift(AgcBurst_FFT)), 20.0e6, "AgcBurst_FFT, ifft")

    AgcBurst_FFT_IFFT = np.fft.ifft(np.fft.fftshift(AgcBurst_FFT))
    if bPlot == True: 
        vis.plot_data([AgcBurst_FFT_IFFT], ["AgcBurst_FFT_IFFT"],0,True)
        vis.plot_data([np.fft.fftshift(AgcBurst_FFT_IFFT)], ["AgcBurst_FFT_IFFT shift"],0,True)
        vis.plot_psd((AgcBurst_FFT_IFFT), 20.0e6, "AgcBurst_FFT_IFFT")

        
    
    
    # AgcBurst_FFTshift = np.fft.fftshift(AgcBurst_FFT) # freq domain 
    
    # AgcBurst_FFT2 = np.fft.fft(FFT_Buffer2)   # freq domain
    # AgcBurst_FFT2shift = np.fft.fftshift(AgcBurst_FFT2) # freq domain 



    
    if bPlot == True:
        plt.figure()
        plt.plot(FFT_Buffer.real, 'r', FFT_Buffer.imag, 'b')
        plt.grid(True)
        plt.title('ZC N samples, time domain')

        plt.figure()
        plt.plot(np.fft.fftshift(FFT_Buffer).real, 'r', np.fft.fftshift(FFT_Buffer).imag, 'b')
        plt.grid(True)
        plt.title('ZC N samples, time domain')
        
        plt.figure()
        plt.plot(AgcBurst_FFT.real, 'r', AgcBurst_FFT.imag, 'b')
        plt.grid(True)
        plt.title('AGCBurst FFT, Freq Domain')
        
        # plt.figure()
        # plt.plot(AgcBurst_FFTshift.real, 'r', AgcBurst_FFTshift.imag, 'b')
        # plt.grid(True)
        # plt.title('AGCBurst FFT shift, Freq Domain')
        
        # plt.figure()
        # plt.plot(AgcBurst_FFT2.real, 'r', AgcBurst_FFT2.imag, 'b')
        # plt.grid(True)
        # plt.title('AGCBurst FFT2, Freq Domain')

        # plt.figure()
        # plt.plot(AgcBurst_FFT2shift.real, 'r', AgcBurst_FFT2shift.imag, 'b')
        # plt.grid(True)
        # plt.title('AGCBurst FFT2 shift, Freq Domain')

        # AgcBurst_FFT_IFFT = np.fft.ifft(AgcBurst_FFTshift)
        # plt.figure()
        # plt.plot(AgcBurst_FFT_IFFT.real, AgcBurst_FFT_IFFT.imag, 'r.')
        # plt.plot(AgcBurst_FFT_IFFT.real, AgcBurst_FFT_IFFT.imag, 'b')
        # plt.plot(zc[0:NumberOfSamplesToRetain].real, zc[0:NumberOfSamplesToRetain].imag, 'go')
        # plt.grid(True)
        # plt.title('IQ constellation , time domain')
        
        plt.figure()
        plt.plot(AgcBurst_FFT_IFFT.real,'b', AgcBurst_FFT_IFFT.imag, 'r')
        plt.plot(zc[0:NumberOfSamplesToRetain].real, 'go', zc[0:NumberOfSamplesToRetain].imag, 'gx')
        plt.grid(True)
        plt.title('I, Q in time domain')
        

    # ------------------------------------
    # Mapping into N = 1024 or 2048 IFFT buffer and render into time domain
    # ------------------------------------
    if SampleRate == 20e6:
         IFFT_Buffer = np.zeros(1024, np.complex128)
    else:
         IFFT_Buffer = np.zeros(2048, np.complex128)

    # was:
    # IFFT_Buffer[0:ScPositive]  = AgcBurst_FFT[0:ScPositive]          
    # IFFT_Buffer[-ScNegative:]  = AgcBurst_FFT[-ScNegative:]  
    # IFFT_Buffer[0]             = 0
    
    IFFT_Buffer[0:ScPositive]  = AgcBurst_FFT[0:ScPositive]          
    IFFT_Buffer[-ScNegative:]  = AgcBurst_FFT[-ScNegative:]  
    IFFT_Buffer[0]             = 0
    
    if bPlot == True:
        plt.figure(3)
        plt.plot(IFFT_Buffer.real, 'r', IFFT_Buffer.imag, 'b')
        plt.grid(True)
        plt.title('Original')
        
        plt.figure(4)
        plt.plot(IFFT_Buffer.real, IFFT_Buffer.imag, 'r.')
        plt.plot(IFFT_Buffer.real, IFFT_Buffer.imag, 'b')
        plt.grid(True)
        plt.title('IQ constellation')
    
    AgcBurstFull               = (len(IFFT_Buffer)/Nzc)*np.fft.ifft(IFFT_Buffer)
    # AgcBurstFull               = np.fft.ifft(IFFT_Buffer)
    
    if bPlot == True:
        AgcBurstFull_FFT        = np.fft.fft(AgcBurstFull)
        plt.figure(6)
        plt.plot(AgcBurstFull_FFT.real, AgcBurstFull_FFT.imag, 'g.')
        # plt.plot(AgcBurst.real, AgcBurst.imag, 'r.')
        # plt.plot(AgcBurst.real, AgcBurst.imag, 'b')
        plt.grid(True)
        plt.title('Rendered')
        plt.show()


    AgcBurst  = AgcBurstFull[0:NumberOfSamplesToRetain]

    if bPlot == True:
        AgcBurstRetain_FFT = np.fft.fft(AgcBurst,1024)
        plt.figure(5)
        # plt.plot(AgcBurstRetain_FFT.real, AgcBurstRetain_FFT.imag, 'b.')
        plt.plot(zc[0:NumberOfSamplesToRetain].real, zc[0:NumberOfSamplesToRetain].imag, 'b.')
        plt.grid(True)
        plt.title('Rendered')
        plt.show()

        plt.figure(6)
        plt.plot(AgcBurstFull.real, AgcBurstFull.imag, 'g.')
        plt.plot(AgcBurst.real, AgcBurst.imag, 'r.')
        plt.plot(AgcBurst.real, AgcBurst.imag, 'b')
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
                    , FftSize:    int = 1024
                    , strMode:    str = 'long'
                    , bPlot:      bool = False) -> np.ndarray:
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

    SubcarrierSpacing = 20e6/float(FftSize)
    CosineFrequencyA  = 32*SubcarrierSpacing # Approximately  625000 Hz 
    CosineFrequencyB  = 96*SubcarrierSpacing # Approximately 1875000 Hz 
    
    IFFT_Buffer = np.zeros(1024, np.complex128)
    IFFT_Buffer[32] = 0 + 1.0j
    IFFT_Buffer[96] = 0 - 1.0j
    IFFT_Buffer[-32] = 0 - 1.0j
    IFFT_Buffer[-96] = 0 + 1.0j
    IFFT_preambleA = np.fft.ifft(IFFT_Buffer)
    
    IFFT_duration = FftSize / SampleRate
    print("One IFFT duration : ", IFFT_duration )
    
    if bPlot == True:
        plt.figure()
        plt.plot(IFFT_preambleA.real, 'r')
        plt.plot(IFFT_preambleA.imag, 'b')
        plt.grid(True)
        plt.title('IFFT of PreambleA')
        plt.show()
    
        binSpacing = 20.00e6 / FftSize
        print('bin spacing ({0}, {1}FFT): {2}'.format('20.0Msps', FftSize, binSpacing))             
        binSpacing = 20.48e6 / FftSize
        print('bin spacing ({0}Msps, {1}FFT): {2}'.format(20.48e6/1e6, FftSize, binSpacing))        

    
    
    
    
    Ts           = 1/SampleRate
    NumSamples   = math.floor(DurationSec / Ts)

    Time  = np.arange(0, NumSamples*Ts, Ts, dtype = np.float64)
    Angle = np.pi / 4
    Tone1 = np.exp( 1j*(2*np.pi*CosineFrequencyA*Time + Angle), dtype = np.complex64) 
    Tone2 = np.exp( 1j*(2*np.pi*CosineFrequencyB*Time + 3*Angle), dtype = np.complex64) 
    Tone3 = np.exp(-1j*(2*np.pi*CosineFrequencyA*Time + Angle), dtype = np.complex64) 
    Tone4 = np.exp(-1j*(2*np.pi*CosineFrequencyB*Time + 3*Angle), dtype = np.complex64) 

    PreambleA = (1/2) * (Tone1 + Tone2 + Tone3 + Tone4)

    return PreambleA.astype(np.complex64)





# ----------------------------------------------------------------------------------------------------------- #
# > GeneratePreambleB()                                                                                       #
# ----------------------------------------------------------------------------------------------------------- #
def GeneratePreambleB(SampleRate: float = 20.0e6, bPlot: bool = False) -> np.ndarray:

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

    if bPlot == True:
        vis.plot_constellation(zc,0,-1,"zc constelation")
        vis.plot_psd(zc, SampleRate, True, "zc psd")
        vis.plot_data([zc.real, zc.imag], ["zc seq real", "zc seq imag"],0,True)

    # The Fourier Transform of the Zadoff-Chu Sequence
    PreambleB_FFT = np.fft.fft(zc); 
    if bPlot == True:
        vis.plot_data([PreambleB_FFT.real, PreambleB_FFT.imag], ["PreambleB_FFT real", "PreambleB_FFT imag"],0,True)
        # vis.plot_constellation(zc,0,-1,"zc constelation")
        # vis.plot_psd(zc, 20.0e6, "zc psd")
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
    PreambleB_FFT1024       = IFFT_Buffer
    
    if bPlot == True:
        vis.plot_data([IFFT_Buffer.real, IFFT_Buffer.imag], ["PreambleB_FFT(mapped) real ", "PreambleB_FFT(mapped) imag"],0,True)
        
    PreambleB                  = (len(IFFT_Buffer)/Nzc)*np.fft.ifft(IFFT_Buffer)
    if bPlot == True:
        vis.plot_data([PreambleB.real, PreambleB.imag], ["PreambleB real", "PreambleB imag"],0,True)
        vis.plot_constellation(PreambleB,0,-1,"PreambleB constelation")
        vis.plot_psd(PreambleB, SampleRate, True, "PreambleB psd")
    

    return PreambleB.astype(np.complex64), PreambleB_FFT1024.astype(np.complex64)












# ---------------------------------------------------------------------------------------------------- #
#                                                                                                      #
#                                                                                                      #
# > Processing the Preamble                                                                            #
#                                                                                                      #
#                                                                                                      #
# ---------------------------------------------------------------------------------------------------- #


# --------------------------------------------------------------
# > DetectPreambleA()
# --------------------------------------------------------------
def DetectPreambleA(InputSequence:     np.ndarray
                  , SampleRate:        float = 20e6
                  , bShowPlot:         bool = False) -> tuple:
    
    # Error checking
    assert isinstance(InputSequence, np.ndarray)
    assert np.issubdtype(InputSequence.dtype, np.complexfloating) 
    assert isinstance(SampleRate, float) or isinstance(SampleRate, int) 
    assert isinstance(bShowPlot, bool)
    assert SampleRate == 20e6,                                              'The SampleRate must be 20MHz'

    # Use a filter to remove noise from the signal 
    # Remember that the Nyquist range is between -10Mhz to +10MHz. 
    # The tones are at 625000 Hz and 1875000 Hz
    GainListLinear = [.0001, .0001,     1,    1,  .0001, .0001 ]
    NormalizedFreq = [-0.4,   -0.16, -0.13, 0.13,    0.16,    0.5]
    N              = 31
    ImpulseResponse = CFilterDesigner.FrequencySampling(GainListLinear, NormalizedFreq, N,  False)

    if bShowPlot == True:
        vis.plot_data([ImpulseResponse], ["ImpulseResponse"])

    #CFilterDesigner.ShowFrequencyResponse(ImpulseResponse   = ImpulseResponse
    #                                     , SampleRate        = 20.0e6
    #                                     , OversamplingRatio = 32
    #                                     , bShowInDb         = False)
   
    # Filter the RxPreambleA
    FilteredPreambleA          = np.convolve(InputSequence, ImpulseResponse)
    
    if bShowPlot == True:
        vis.plot_data([FilteredPreambleA], ["FilteredPreambleA"])


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

    if bShowPlot == True:
        FilteredPreambleA_FFT = np.fft.fftshift(np.fft.fft(FilteredPreambleA[-1024:]))
        Nfft    = FftSize
        #df = ReferenceFrequency/Nfft
        freqs = np.fft.fftshift(np.fft.fftfreq(Nfft, d=1/SampleRate))
        freqs = freqs/1000
        FilteredPreambleA_FFT_absdBm = 20*np.log10(abs(FilteredPreambleA_FFT))
        #vis.plot_data([FilteredPreambleA_FFT_absdBm], ["FilteredPreambleA_FFT_absdBm"])
        vis.plot_data_xy(freqs, FilteredPreambleA_FFT_absdBm,"FilteredPreambleA_FFT_absdBm" )



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
    DetectionIndex     = 0
    for Index, CurrentRatio in enumerate(Ratio):
        if CurrentRatio > DetectionThreshold:
            Count             += 1
            DetectionStrength += CurrentRatio/Range
            if Count == Range:
                 DetectionSuccess = True
                 DetectionIndex   = Index
                 break
        else:
             Count             = 0
             DetectionStrength = 0
                               

    if bShowPlot == True:
        plt.figure(5)
        plt.subplot(3, 1, 1)
        plt.plot(np.arange(0, len(CurrentCovariance)), np.abs(CurrentCovariance), 'b')
        plt.title('The absolute value of the Covariance')
        plt.tight_layout()
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(np.arange(0, len(CurrentVariance)), np.abs(CurrentVariance), 'b')
        plt.title('The absolute value of the Variance')
        plt.tight_layout()
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(np.arange(0, len(Ratio)), Ratio, 'b')
        plt.title('The Ratio of absolute values of the Covariance and Variance')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return (DetectionSuccess, DetectionStrength, DetectionIndex, Ratio)




# --------------------------------------------------------------
# > ProcessPreambleA()
# --------------------------------------------------------------
def ProcessPreambleA(IqInputSequence:   np.ndarray
                   , SampleRate:        float = 20e6
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
    
    assert np.issubdtype(IqInputSequence.dtype, np.complexfloating), 'The RxWaveform input argument must be a complex numpy array.'
    assert len(IqInputSequence.shape) == 1,                          'The RxWaveform must be a simple np array.'
    assert len(IqInputSequence) >= FFT_Size,                         'The RxWavefom must feature at least FFT_Size entries.'
    assert isinstance(bHighCinr, bool),                              'The bHighCinr input argument must be of type bool.'
    assert isinstance(bShowPlots, bool),                             'The bShowPlots input arguement must be of type bool.'
    assert SampleRate == 20e6,                                       'The sample rate must be 20MHz'

    # ------------------------
    # Overlay the Hanning Window to induce more FFT leakage close by and suppress it further away
    # ------------------------
    N              = FFT_Size
    n              = np.arange(0, N, 1, dtype = np.int32)
    Hanning        = 0.5 - 0.5 * np.cos(2*np.pi * (n + 1) / (N + 1))
    RxWaveform     = IqInputSequence[0:N].copy() * Hanning

    # ------------------------------
    # Take the FFT of the first FFT_Size samples and rearrange it such that negative frequency bins are first
    # ------------------------------
    ZeroIndex      = int(FFT_Size/2)
    FFT_Output     = np.fft.fft(RxWaveform[0:FFT_Size])
    FFT_Rearranged = np.hstack([FFT_Output[ZeroIndex : ], FFT_Output[:ZeroIndex]]) 

    #if(False):
    if(bShowPlots):
        plt.figure()
        xindices = np.arange(-2048, 2048, 1, np.int32) * SampleRate / 4096
        # plt.plot(xindices, np.abs(FFT_Rearranged), 'k')
        plt.stem(xindices, np.abs(FFT_Rearranged))
        plt.grid(True)
        plt.title('FFT Rearranged')
        plt.show()

        binSpacing = 20.00e6 / FFT_Size
        print('bin spacing ({0}, {1}FFT): {2}'.format('20.0Msps', FFT_Size, binSpacing))
        
        binSpacing = 20.00e6 / 1024
        print('bin spacing ({0}, {1}FFT): {2}'.format('20.0Msps', 1024, binSpacing))
        
        binSpacing = 20.48e6 / FFT_Size
        print('bin spacing ({0}Msps, {1}FFT): {2}'.format(20.48e6/1e6, FFT_Size, binSpacing))
        
        binSpacing = 20.48e6 / 1024
        print('bin spacing ({0}Msps, {1}FFT): {2}'.format(20.48e6/1e6, 1024, binSpacing))
        

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






# --------------------------------------------------------------
# > TestPreambleA()
# --------------------------------------------------------------
def TestPreambleA(IqInputSequence:   np.ndarray
                   , SampleRate:        float = 20e6
                   , bHighCinr:         bool  = False
                   , bShowPlots:        bool  = False) -> float:
    """
    This function will attempt to guess at the frequency offset in the PreambleA Waveform
    """
    # F1 = 4*160KHz  * (Fs / 20.48MHz) = 640KHz  * (Fs / 20.48MHz)
    # F2 = 12*160KHz * (Fs / 20.48MHz) = 1.92MHz * (Fs / 20.48MHz) 
    F1 = 4*160000   * (SampleRate / 20.48e6) 
    F2 = 12*160000  * (SampleRate / 20.48e6) 
    
    # F1/F2 = 640KHz/1.92MHz  if Fs = 20.48MHz
    # F1/F2 = 625KHz/1.875MHz if Fs = 20.00MHz
    
    # trying to detect Approximately  625000 Hz
    # trying to detect Approximately 1875000 Hz


    # ------------------------
    # Error checking
    # ------------------------
    FFT_Size    = 4096 #64
    
    assert np.issubdtype(IqInputSequence.dtype, np.complexfloating), 'The RxWaveform input argument must be a complex numpy array.'
    assert len(IqInputSequence.shape) == 1,                          'The RxWaveform must be a simple np array.'
    assert len(IqInputSequence) >= FFT_Size,                         'The RxWavefom must feature at least FFT_Size entries.'
    assert isinstance(bHighCinr, bool),                              'The bHighCinr input argument must be of type bool.'
    assert isinstance(bShowPlots, bool),                             'The bShowPlots input arguement must be of type bool.'
    assert SampleRate == 20e6,                                       'The sample rate must be 20MHz'

    # ------------------------
    # Overlay the Hanning Window to induce more FFT leakage close by and suppress it further away
    # ------------------------
    N              = FFT_Size
    n              = np.arange(0, N, 1, dtype = np.int32)
    Hanning        = 0.5 - 0.5 * np.cos(2*np.pi * (n + 1) / (N + 1))
    RxWaveform     = IqInputSequence[0:N].copy() * Hanning

    # ------------------------------
    # Take the FFT of the first FFT_Size samples and rearrange it such that negative frequency bins are first
    # ------------------------------
    ZeroIndex      = int(FFT_Size/2)
    FFT_Output     = np.fft.fft(RxWaveform[0:FFT_Size])
    FFT_Rearranged = np.hstack([FFT_Output[ZeroIndex : ], FFT_Output[:ZeroIndex]]) 

    #if(False):
    if(bShowPlots):
        plt.figure()
        xindices = np.arange(-(int(FFT_Size/2)), int(FFT_Size/2), 1, np.int32) * SampleRate / FFT_Size
        # plt.plot(xindices, np.abs(FFT_Rearranged), 'k')
        plt.stem(xindices, FFT_Rearranged.real, 'r')
        plt.stem(xindices, FFT_Rearranged.imag, 'b')
        plt.grid(True)
        plt.title('FFT Rearranged')
        plt.show()

        binSpacing = 20.00e6 / FFT_Size
        print('bin spacing ({0}, {1}FFT): {2}'.format('20.0Msps', FFT_Size, binSpacing))
        
        binSpacing = 20.00e6 / 1024
        print('bin spacing ({0}, {1}FFT): {2}'.format('20.0Msps', 1024, binSpacing))
        
        binSpacing = 20.48e6 / FFT_Size
        print('bin spacing ({0}Msps, {1}FFT): {2}'.format(20.48e6/1e6, FFT_Size, binSpacing))
        
        binSpacing = 20.48e6 / 1024
        print('bin spacing ({0}Msps, {1}FFT): {2}'.format(20.48e6/1e6, 1024, binSpacing))
    


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





    #------------------------------------------------------------
    peak_indices = PeakIndices #  used for fft of 64 [26, 30, 34, 38]
    fft_result = FFT_Rearranged
    Fs = SampleRate
    
    frequencies = []
    delta_list = []
    N = FFT_Size # 64  # FFT size
    
    for k in peak_indices:
        if 0 < k < N - 1:
            Xk = fft_result[k]
            Xk_minus = fft_result[k - 1]
            Xk_plus = fft_result[k + 1]
            
            numerator = (Xk_plus.real * Xk.real + Xk_plus.imag * Xk.imag) - \
                        (Xk_minus.real * Xk.real + Xk_minus.imag * Xk.imag)
            denominator = (2 * (Xk.real**2 + Xk.imag**2)) - \
                        (Xk_plus.real * Xk.real + Xk_plus.imag * Xk.imag) - \
                        (Xk_minus.real * Xk.real + Xk_minus.imag * Xk.imag)
            
            delta = numerator / denominator if denominator != 0 else 0
            delta_list.append(delta)
            k_interp = k + delta
            freq = ((k_interp - int(FFT_Size/2)) * Fs / N) 
            frequencies.append(freq)
        else:
            freq = k * Fs / N
            frequencies.append(freq)

    
    #------------------------------------------------------------
    # Assuming 'signal' is your time-domain signal and 'n' is the FFT size (64 in your case)
    fft_result = np.fft.fft(IqInputSequence[:-FFT_Size],n=FFT_Size) #(FFT_Rearranged, n=FFT_Size)
    I = fft_result.real
    Q = fft_result.imag
    magnitude_spectrum = np.sqrt(I**2 + Q**2)
    phase_spectrum = np.arctan2(Q, I)

    # Since the FFT is symmetric, we only need to consider the first half
    half_spectrum = magnitude_spectrum[:int(FFT_Size/2)]
    peak_indices = np.argpartition(half_spectrum, -2)[-2:]

    # magnitude_spectrum = np.abs(fft_result)
    peak_indices = np.argpartition(magnitude_spectrum[:int(FFT_Size/2)], -2)[-2:]  # Use only half due to symmetry

    
    frequencies = []
    N = FFT_Size # 64  # FFT size
    Fs = SampleRate  # Replace with your actual sampling rate

    for k in peak_indices:
        if 0 < k < N - 1:
            # Phases of the bins around the peak
            phi_k_minus = phase_spectrum[k - 1]
            phi_k = phase_spectrum[k]
            phi_k_plus = phase_spectrum[k + 1]
            
            # Unwrap phases to prevent discontinuities
            phi_k_minus = np.unwrap([phi_k_minus, phi_k])[0]
            phi_k_plus = np.unwrap([phi_k, phi_k_plus])[1]
            
            # Estimate frequency deviation
            delta = (phi_k_plus - phi_k_minus) / (4 * np.pi)
            
            k_interp = k + delta
            freq = k_interp * Fs / N
            frequencies.append(freq)
        else:
            # Edge bins where interpolation is not possible
            freq = k * Fs / N
            frequencies.append(freq)
    #------------------------------------------------------------
    





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
      

    #Test = 0 #<<         # 0 - Test the AGC Burst performance
    #Test = 1 #<<         # 1 - Spot test of the        PreambleA detection performance 
    # Test = 2          # 2 - Monte Carlo test of the PreambleA detection performance
    # Test = 3          # 3 - Spot test of the        PreambleA frequency detector 
    # Test = 4          # 4 - Monte Carlo test of the PreambleA frequency detector
    Test = 5 #<<         # 5 - generate and visualize PreambleB
    
    # -----------------------------------------------------------------------------------------------
    if Test == 0:

        TxAgcBurst = GenerateAgcBurst( SampleRate= 20.00e6, bPlot = True)
        MaxD = max(np.abs(TxAgcBurst))
        Pap = 10*np.log10( (MaxD**2) / np.var(TxAgcBurst) )
        MeanSquare = np.mean(TxAgcBurst * np.conj(TxAgcBurst)).real
        
        vis.plot_data([TxAgcBurst.real, TxAgcBurst.imag], ["TxAgcBurst.real", "TxAgcBurst.imag"])
        #vis.plot_data([np.abs(TxAgcBurst)], ["abs TxAgcBurst"])

        print('Pap (dB):   ' + str(Pap))
        print('MeanSquare: ' + str(MeanSquare))

        # plt.figure(1)
        # plt.plot(np.abs(TxAgcBurst), c = 'red')
        # plt.grid(True)
        # plt.show()
        print('end')
        input("press enter key to close and exit: ") 




    # -----------------------------------------------------------------------------------------------
    if Test == 1:
        # -------------------------------------
        # Spot Checking Analysis
        # -------------------------------------

        # -------------------------------------
        # Generate the PreambleA sequence 
        # -------------------------------------
        bShowPlot   = True
        
        ReferenceFrequency = 20e6
        ReferencePeriod    = 1/ReferenceFrequency
        TxPreambleA        = GeneratePreambleA(SampleRate  = ReferenceFrequency
                                              , FftSize    = 1024
                                              , strMode    = 'long'
                                              , bPlot      = bShowPlot)
        
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
                                    , Constants     = [0.7+0.72j, -0.50, 0.2j, 0.1j]   
                                    , Dopplers      = [5000, 4000, -2000, 3000])[0]
        SignalPower = np.mean(TxPreambleMp * np.conj(TxPreambleMp)).real 


        if bShowPlot == True:
            signal_FFT = np.fft.fftshift(np.fft.fft(TxPreambleA[500:500+1024]))
            Nfft    = 1024
            #df = ReferenceFrequency/Nfft
            freqs = np.fft.fftshift(np.fft.fftfreq(Nfft, d=1/ReferenceFrequency))
            freqs = freqs/1000
            signal_FFT_absdBm = 20*np.log10(abs(signal_FFT))
            vis.plot_data_xy(freqs, signal_FFT_absdBm,"TxPreambleA_FFT_absdBm" )
            
            
            TxPreambleMp_FFT = np.fft.fftshift(np.fft.fft(TxPreambleMp[500:500+1024]))
            Nfft    = 1024
            #df = ReferenceFrequency/Nfft
            freqs = np.fft.fftshift(np.fft.fftfreq(Nfft, d=1/ReferenceFrequency))
            freqs = freqs/1000
            TxPreambleMp_FFT_absdBm = 20*np.log10(abs(TxPreambleMp_FFT))
            vis.plot_data_xy(freqs, TxPreambleMp_FFT_absdBm,"TxPreambleMp_FFT_absdBm" )


        # ------------------------
        # Add noise to the waveform
        SNRdB            = -10
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
        vis.plot_data([TxPreambleA[500:].real, RxPreambleANoisy[500:].imag], ["TxPreambleA.real", "RxPreambleANoisy.imag"])

    
        # ------------------------
        # Run the packet detector 
        #
        # set last argument to be True to display plots.
        DetectionSuccess, DetectionStrength, DetectionIndex, Ratio = DetectPreambleA(RxPreambleANoisy, ReferenceFrequency, True)
        
        if DetectionSuccess == True:
             print('PreambleA detected with strength: ' + str(DetectionStrength))
        else:
             print('PreambleA not detected')
             
        print('End of test 1')
        input("press enter key to close and exit: ") 




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

        print('End of test 1')
        input("press enter key to close and exit: ") 


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
                                         #, Constants     = [.7+0.72j, -0.50, 0.2j, 0.1j]
                                         #, Dopplers      = [-1000, -1050, -950, -800])[0]
                                         , Constants     = [1.0+0.0j, 0.0001j, 0.0001j, 0.0001j] #   
                                         , Dopplers      = [-5000   , 0.0,  0.0,  0.0 ])[0] #

        # lwd testing 
        vis.plot_data([TxPreambleA_Offset.real, TxPreambleA_Offset.imag], 
                      ["TxPreambleA_Offset real", "TxPreambleA_Offset imag"], 0,True)

        FreqOffset     = TestPreambleA(TxPreambleA_Offset, SampleRate, bHighCinr=True, bShowPlots=True)
        
        
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
        SnrDb            = 10 #-10
        SNR_Linear       = 10**(SnrDb/10)
        NoisePower       = SignalPower / SNR_Linear
        Noise            = GenerateAwgn(NumberOfSamples = len(TxPreambleA_Offset)  
                                      , Type            = 'complex64'
                                      , Variance        = NoisePower
                                      , Seed            = -1) 
        
        TxPreampleA_Noisy   = TxPreambleA_Offset + Noise

        HighSnrdB      = True
        FreqOffset     = ProcessPreambleA(TxPreampleA_Noisy, SampleRate, HighSnrdB, False)
        print("Frequency offset: ", FreqOffset)

        print('End of test')
        input("press enter key to close and exit: ") 



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
        
        print('End of test 1')
        input("press enter key to close and exit: ") 
        
    #-----------------------------------------------------------------------------------------------
    if Test == 5:
        # -------------------------------------
        # Generate the PreambleB sequence 
        # -------------------------------------
        bPlot   = False
        dspConfPlot = True
        SNR     =  -10.0 # in dB
        ReferenceFrequency = 20e6   # also sampling frequency, Fs
        ReferencePeriod    = 1/ReferenceFrequency
        # TxPreambleB        = GeneratePreambleB(SampleRate = ReferenceFrequency, bPlot=True)
        TxPreambleB, PreambleB_FFT1024 = GeneratePreambleB(SampleRate = ReferenceFrequency, bPlot=False)
        
        if (bPlot or dspConfPlot) == True:
            vis.plot_data([TxPreambleB.real, PreambleB_FFT1024.real], ["TxPreambleB", "PreambleB_FFT1024"], 0,True)
            print("Showing Tx Preamble with  no impairments")

        
        # MeanSquare         = np.mean(TxPreambleA * np.conj(TxPreambleA)).real
        PreambleBTime               = np.arange(0, len(TxPreambleB), 1) * ReferencePeriod 
        print("Length in samples of Preamble B: ", len(TxPreambleB))
        # print("Time for Preamble B :" , PreambleBTime)
        
        
        NumberOfSamples = len(TxPreambleB)  # should be 1024
        
        #----------------------------------------------------------------------
        TxPreambleB_Offset =  AddMultipath(InputSequence = TxPreambleB
                                         , SampleRate    = ReferenceFrequency
                                         , Delays        = [int(20e6*0), int(20e6*1.25e-6), int(20e6*2.25e-6), int(20e6*3.25e-6)]  
                                         , Constants     = [0.17+0.172j, -0.7, 0.2j, 0.1j]   #[0.7+0.72j, -0.50, 0.2j, 0.1j] 
                                         , Dopplers      = [-1000, -1050, -950, -200])[0]

        if(False):
            # -------------------------------------
            # Introduce Frequency Error
            # -------------------------------------
            n                   = np.arange(0, len(TxPreambleA), 1, np.int32)
            FrequencyError      = 40000
            Sinusoid            = np.exp(1j*2*np.pi*n*FrequencyError/SampleRate) 
            TxPreambleA_Offset  = TxPreambleA * Sinusoid
        
        SignalPower         = np.mean(TxPreambleB_Offset * np.conj(TxPreambleB_Offset)).real

        # --------------------------------------
        # Introduce Gaussian Noise
        # --------------------------------------
        SnrDb            = SNR
        SNR_Linear       = 10**(SnrDb/10)
        NoisePower       = SignalPower / SNR_Linear
        AdditionalSamples = 0 # number of samples added to the start and end of the preamble
        Noise            = GenerateAwgn(NumberOfSamples = len(TxPreambleB_Offset)  
                                      , Type            = 'complex64'
                                      , Variance        = NoisePower
                                      , Seed            = -1) 
        
        # RxPreambleB_Noisy   = TxPreambleB_Offset + Noise
        RxPreambleB_Noisy   = (np.hstack([[np.zeros(AdditionalSamples, np.complex64)], 
                                          [TxPreambleB_Offset], 
                                          [np.zeros(AdditionalSamples, np.complex64)]])  + Noise).flatten()
        #----------------------------------------------------------------------
        if (bPlot or dspConfPlot) == True:
            vis.plot_data([RxPreambleB_Noisy.real, RxPreambleB_Noisy.imag], ["RxPreambleB_Noisy I", "RxPreambleB_Noisy Q"], 0,True)
            vis.plot_psd(RxPreambleB_Noisy, ReferenceFrequency, True, "RxPreambleB_Noisy PSD")
            print("Showing Rx Preamble is Tx Preamble with impairments added")
        
        
        #----------------------------------------------------------------------
        # precalculated complex conjagate value for correlation function
        PreambleB_FFT1024cc = PreambleB_FFT1024.conj()
        PreambleB_FFT1024cc = np.fft.fftshift(PreambleB_FFT1024cc)
        RxPreambleB_FFT_early = np.fft.fftshift(np.fft.fft(RxPreambleB_Noisy[0:1024])) #TxPreambleB))
        RxPreambleB_FFT_late  = np.fft.fftshift(np.fft.fft(RxPreambleB_Noisy[-1024:])) #TxPreambleB))

        RxPreambleB_FFT = RxPreambleB_FFT_early
        
        if bPlot == True:
            vis.plot_data([PreambleB_FFT1024cc.real, PreambleB_FFT1024cc.real], ["Ideal PreambleB_FFT1024 \nComplex Conj I", "PreambleB_FFT1024 \nComplex Conj I"], 0,True)
            vis.plot_data([RxPreambleB_FFT.real, RxPreambleB_FFT.imag], ["RxPreambleB_FFT I", "RxPreambleB_FFT Q"], 0,True)
            # vis.plot_psd(RxPreambleB_FFT, ReferenceFrequency, "RxPreambleB_FFT \n1024 samples")


        #% calculate frequency spacing
        #df = Fs/nfft = 1/(t*nfft)
        Nfft    = len(PreambleB_FFT1024)
        df = ReferenceFrequency/Nfft

        # df = Fs_d/nfft

        # nfft = len(sig_fft)  # or
        RxPreambleB_Nfft = RxPreambleB_FFT.size
        freqs = np.fft.fftshift(np.fft.fftfreq(Nfft, d=1/ReferenceFrequency))
        freqs = freqs/1000
        RxPreambleB_FFT_absdBm = 20*np.log10(abs(RxPreambleB_FFT))

        # plot of the TX signal in freq domain dBm
        # vis.plot_data_xy_complex(freqs, RxPreambleB_FFT_absdBm )
        if bPlot == True:
            vis.plot_data([RxPreambleB_FFT_absdBm], ["RX PreambleB FFT abs dBm"], 0,True)

        
        #------------------------------------------------------------------------------
        # perform an element wise mulitiplication of the FFT of the signal
        # and the FFT conjugate for correlation function 
        # -----------------------------------------------------------------------------
        TxPreBFFT_x_PreBFFT1024 = RxPreambleB_FFT * PreambleB_FFT1024cc
        if bPlot == True:
            vis.plot_data([RxPreambleB_FFT.real, PreambleB_FFT1024cc.real], ["RxPreambleB_FFT", "PreambleB_FFT1024cc"], 0,True)

        
        PreambleB_corr = abs(np.fft.ifft(TxPreBFFT_x_PreBFFT1024))
        # plot of the TX signal in freq domain dBm
        if (bPlot or dspConfPlot) == True:
            #vis.plot_data_xy(freqs, PreambleB_corr, "PreambleB_corr" )
            vis.plot_data([PreambleB_corr], ["PreambleB_correlation"], 0,True)
            
        
        print('end')
        input("press enter key to close and exit: ") 
        
        
        
        