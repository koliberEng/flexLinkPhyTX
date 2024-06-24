# File:     SignalProcessing.py
# Notes:    This file provides basic signal processing functions
#           1. GenerateAwgn()                - This function creates real or complex Additive White Gaussian noise  
#           2. AddAdwgn()                    - This function adds    real or complex Additive White Gaussian Noise to a input sequence
#           3. AddMultipath()                - This function adds multiple time domain paths of a complex transmit sequence.
#                                              The delays are in terms of samples rather than time.
#           4. AddMultipath2()               - This function adds multiple time domain paths of a complex transmit sequence.
#                                              The delays are in terms of time and interpolation is non-linear
#           5. AddMultipath3()               - This function is identical to AddMultipath2 excepts that it allows you to choose a
#                                              an offset where in the source waveform to execute the first interpolation
#           6. SpectrumAnalyzer()            - This function computes the power spectrum of a real or complex sequence
#           7. ComputeCorrelation()          - Computes the correlation between two sequences using both time and frequency domain algorithms.
#           8. FilterFrequency()             - Frequency domain filtering algorithm
#           9. ComputeFirFrequencyResponse() - Computes and shows the frequency response given a sample rate and an array of FIR filters.
#          10. Resample()                    - Resamples the waveform using spline interpolation from one sample rate to another


__title__     = "SignalProcessing"
__author__    = "Andreas Schwarzinger"
__version__   = "1.0.0"
__status__    = "released"
__date__      = "May, 23rd, 2023"
__copyright__ = 'Andreas Schwarzinger'

import os
import sys
import numpy             as np
import matplotlib.pyplot as plt
import math
from   scipy             import interpolate    
import time

ThisDirectory = os.path.dirname(__file__)
sys.path.append(ThisDirectory)







# ------------------------------------------------------------------------------------------------------------ #
#                                                                                                              #
# > 1. GenerateAwgn()                                                                                          #
#                                                                                                              #
# ------------------------------------------------------------------------------------------------------------ #
def GenerateAwgn(NumberOfSamples: int 
               , Type:            str   = 'complex64'
               , Variance:        float = 1.0
               , Seed:            int   = -1):
    """
    brief:  This function will generate white Gaussian noise for both real or complex types.
    param:  NumberOfSamples - The number of noise samples to generate (an integer) 
    param:  NumberType      - 'complex64', 'complex128', 'float32', 'float64' 
    param:  Variance        - The power of the noise sequence (a float)
    param:  Seed            -  We want noise with both random seeds (Seed = -1) and constant seed (thus repeatable noise sequences)
    return: An array is returned containing the AWGN.
    """
    # ----------------------------------------- #
    # Error and type checking                   #
    # ----------------------------------------- #
    assert isinstance(NumberOfSamples, int)
    assert isinstance(Type, str)
    assert Type.lower() == 'complex64' or Type.lower() == 'complex128' or \
           Type.lower() == 'float32'   or Type.lower() == 'float64'
    assert isinstance(Variance, int) or isinstance(Variance, float)
    assert isinstance(Seed, int), 'The seed for the random number generator. -1 for a random seed.'

    # ----------------------------------------- #
    # Generate and return the noise sequence    #
    # ----------------------------------------- #
    match Type.lower():
        case 'complex64': 
            Dtype = np.complex64
        case 'complex128':
            Dtype = np.complex128
        case 'float32':
            Dtype = np.float32
        case 'float64':
            Dtype = np.float64
        case _:
            assert False


    # Set up the random number generator object
    if Seed < 0:
        Count = int(time.perf_counter() % 20000)
    else:
        Count = Seed

    r    = np.random.RandomState(Count) 

    N = NumberOfSamples
    if Type.lower() == 'complex64' or Type.lower() == 'complex128':
        Scalar        = np.sqrt(Variance) * np.sqrt(1/2)
        NoiseSequence = Scalar * np.array(r.randn(N) + 1j*r.randn(N), dtype = Dtype)  
    else:
        NoiseSequence = np.sqrt(Variance) * np.array(r.randn(N), dtype = Dtype)  

    return NoiseSequence















# ------------------------------------------------------------------------------------------------------------ #
#                                                                                                              #
# > 2. AddAwgn()                                                                                               #
#                                                                                                              #
# ------------------------------------------------------------------------------------------------------------ #
def AddAwgn(SnrdB:          float 
          , InputSequence:  np.ndarray
          , Seed:           int = -1) -> np.ndarray:
    """
    brief:  This function will add white Gaussian noise to a real or complex sequence in a one dimensional numpy array.
    param:  SnrdB         - The signal to noise ratio of the noise sequence to be generated
    param:  InputSequence - A numpy ndarray of real or complex sequence of numbers 
    param:  Seed          - We want noise with both random seeds (Seed = -1) and constant seed (thus repeatable noise sequences)
    return: A new array is returned that contains the sum of noise and the input sequence.
    """

    # -----------------------------------------
    # Error checking
    assert isinstance(SnrdB, float) or isinstance(SnrdB, int), 'The SnrdB input argument must be a numeric type.'
    assert isinstance(InputSequence, np.ndarray),              'The InputSequence must be a numpy array.'
    assert np.issubdtype(InputSequence.dtype, np.number),      'The InputSequence entries must be real or complex numbers.'
    assert len(InputSequence.shape) == 1,                      'The InputSequence must be a simple one dimensional array.'
    assert isinstance(Seed, int),                              'The seed for the random number generator. -1 for a random seed.'

    # -----------------------------------------
    # We need slighlty different code for real and complexfloat types
    IsComplex = np.issubdtype(InputSequence.dtype, np.complexfloating)

    # Convert SNR in dB to linear
    SnrLinear = 10 ** (SnrdB/10)

    # Compute the signal power (mean square of the input sequence)
    N = len(InputSequence)
    if IsComplex == True:
        MeanSquareSignal = (1/N) * np.sum(InputSequence * np.conj(InputSequence))
    else:
        MeanSquareSignal = (1/N) * np.sum(InputSequence * InputSequence)

    # Compute the required noise power (mean square of the noise sequence)
    MeanSquareNoise = MeanSquareSignal / SnrLinear

    # Set up the random number generator object
    if Seed < 0:
        Count = int( (time.perf_counter() * 100) % 20000)
    else:
        Count = Seed

    r    = np.random.RandomState(Count) 

    if IsComplex == True:
        Scalar        = np.sqrt(MeanSquareNoise) * np.sqrt(1/2)
        NoiseSequence = Scalar * np.array(r.randn(N) + 1j*r.randn(N), dtype = InputSequence.dtype)  
    else:
        NoiseSequence = np.sqrt(MeanSquareNoise) * np.array(r.randn(N), dtype = np.float32)  

    # Return noisy input sequency
    return InputSequence + NoiseSequence




















# ------------------------------------------------------------------------------------------------------------ #
#                                                                                                              #
# > 3. AddMultipath()                                                                                          #
#                                                                                                              #
# ------------------------------------------------------------------------------------------------------------ #
def AddMultipath(InputSequence:  np.ndarray
               , SampleRate:     float
               , Delays:         list
               , Constants:      list
               , Dopplers:       list) -> tuple:
    """
    brief: This function will add multipath distortion to the input signal. 
           Note that the entries in the Delays list are integers representing sample delay.
    param: InputSequence  I  A numpy array with complex floating point entries
    param: SampleRate     I  An integer or floating point scalar
    param: Delays         I  A list of integer sample delays
    param: Constants      I  A list of complex values that scale the waveform
    param: Dopplers       I  A list of integer or floating point Doppler frequencies
    """

    # -----------------------------------------
    # Error checking
    assert isinstance(InputSequence, np.ndarray),                             'The InputSequence must be a numpy array.'
    assert np.issubdtype(InputSequence.dtype, np.complexfloating),            'The InputSequence entries must be complex floating point numbers.'
    assert len(InputSequence.shape) == 1,                                     'The InputSequence must be a simple one dimensional array.'
    assert isinstance(SampleRate, float) or isinstance(SampleRate, int),      'The SampleRate input argument must be a numeric type.'
    assert isinstance(Delays, list) and isinstance(Delays[0], int),           'The Delays argument must be a list of integers.'
    assert isinstance(Constants, list) and isinstance(Constants[0], complex), 'The Constants argument must be a list of complex value.'
    assert isinstance(Dopplers, list),                                        'The Doppler argument must be a list of floating point values.' 
    assert isinstance(Dopplers[0], int) or isinstance(Dopplers[0], float),    'The list of doppler values must be either integer or floating point based.'
    assert len(Delays) == len(Constants) and len(Delays) == len(Dopplers),    'The length of Delays, Constants, and Doppler lists must be the same.'

    # -----------------------------------------
    # Determine the delay range and length of the input sequence
    N             = len(InputSequence)
    DType         = InputSequence.dtype
    n             = np.arange(0, N)
    DelayDistance = max(Delays) - min(Delays) 
    M             = N + DelayDistance    # The length of the output waveform

    # Allocate memory for the output waveform
    OutputSequence = np.zeros(M, dtype = DType)

    # Generate each path and add it to the output sequence
    NumberOfPaths = len(Delays)
    for PathIndex in range(0, NumberOfPaths):
        Delay      = Delays[PathIndex]
        Constant   = Constants[PathIndex]
        Doppler    = Dopplers[PathIndex] 

        # Compute the start index in the output sequence
        StartIndex      = Delay - min(Delays)
        OutputSequence[StartIndex:StartIndex + N] += \
                Constant * np.exp(1j*2*np.pi*Doppler*n/SampleRate, dtype = DType) * InputSequence

    # Return the output sequence
    # We return the min(Delays) so that the user can truncate the OutputSequence to start at Delay = 0.
    return OutputSequence, min(Delays)















# ------------------------------------------------------------------------------------------------------------ #
#                                                                                                              #
# > 4. AddMultipath2()                                                                                          #
#                                                                                                              #
# ------------------------------------------------------------------------------------------------------------ #
def AddMultipath2(InputSequence:  np.ndarray
                , SampleRate:     float
                , Delays:         np.ndarray
                , Constants:      np.ndarray
                , Dopplers:       np.ndarray) -> np.ndarray:
    """
    brief: This function will add multipath distortion to the input signal. 
           Note that the entries in the Delays list are integers representing sample delay.
    param: InputSequence  I  A numpy array with complex floating point entries
    param: SampleRate     I  An integer or floating point scalar
    param: Delays         I  An np.ndarray (np.floating) of delays in seconds
    param: Constants      I  An np.ndarray (np.complexfloating) 
    param: Dopplers       I  An np.ndarray (np.floating) of Doppler frequencies in Hz
    notes: This function returns an output that starts at time = 0. This function extrapolates
    """
    # -----------------------------------------
    # Error checking
    assert isinstance(InputSequence, np.ndarray),                             'The InputSequence must be a numpy array.'
    assert np.issubdtype(InputSequence.dtype, np.complexfloating),            'The InputSequence entries must be complex floating point numbers.'
    assert len(InputSequence.shape) == 1,                                     'The InputSequence must be a simple one dimensional array.'
    assert isinstance(SampleRate, float) or isinstance(SampleRate, int),      'The SampleRate input argument must be a numeric type.'
    assert isinstance(Delays, np.ndarray) and \
           np.issubdtype(Delays.dtype, np.floating),                          'The Delays argument must be an np.ndarray of type np.floating.'
    assert isinstance(Constants, np.ndarray) and \
           np.issubdtype(Constants.dtype, np.complexfloating),                'The Constants argument must be an nd.array of type np.complexfloating.'
    assert isinstance(Dopplers, np.ndarray) and \
           np.issubdtype(Dopplers.dtype, np.floating),                        'The Doppler argument must be an np.ndarray of type np.floating.' 
    assert len(Delays) == len(Constants) and len(Delays) == len(Dopplers),    'The length of Delays, Constants, and Doppler lists must be the same.'


    OutputSequence = np.zeros(len(InputSequence), InputSequence.dtype)
    TS             = 1/SampleRate
    x_Ind          = np.arange(0, len(InputSequence), 1, np.int32)
    x              = x_Ind * TS

    funcR          = interpolate.interp1d(x, InputSequence.real, 'cubic', bounds_error = False, fill_value = 'extrapolate')
    funcI          = interpolate.interp1d(x, InputSequence.imag, 'cubic', bounds_error = False, fill_value = 'extrapolate')

    for PathIndex in range(0, len(Delays)):
        # Delay the input sequence
        x_new                = x - Delays[PathIndex]
        InputSequenceDelayed = funcR(x_new) + 1j*funcI(x_new)
        # Add the Doppler to it
        Sinusoid             = Constants[PathIndex] * np.exp(1j*2*np.pi*x_new*Dopplers[PathIndex])
        OutputSequence      += InputSequenceDelayed * Sinusoid
   
    return OutputSequence
    








# ------------------------------------------------------------------------------------------------------------ #
#                                                                                                              #
# > 5. AddMultipath3()                                                                                          #
#                                                                                                              #
# ------------------------------------------------------------------------------------------------------------ #
def AddMultipath3(InputSequence:  np.ndarray
                , SampleBackOff:  int
                , SampleRate:     float
                , Delays:         np.ndarray
                , Constants:      np.ndarray
                , Dopplers:       np.ndarray):
    """
    brief: This function will add multipath distortion to the input signal. 
           Note that the entries in the Delays list are integers representing sample delay.
    param: InputSequence  I  A numpy array with complex floating point entries
    param: SampleBackOff  I  The sample backoff is a positive integer
    param: SampleRate     I  An integer or floating point scalar
    param: Delays         I  An np.ndarray (np.floating) of delays in seconds
    param: Constants      I  An np.ndarray (np.complexfloating) 
    param: Dopplers       I  An np.ndarray (np.floating) of Doppler frequencies in Hz
    notes: This function returns an output that starts at time = 0. This function extrapolates
    """
    # -----------------------------------------
    # Error checking
    assert isinstance(InputSequence, np.ndarray),                             'The InputSequence must be a numpy array.'
    assert np.issubdtype(InputSequence.dtype, np.complexfloating),            'The InputSequence entries must be complex floating point numbers.'
    assert len(InputSequence.shape) == 1,                                     'The InputSequence must be a simple one dimensional array.'
    assert isinstance(SampleBackOff, int) and SampleBackOff >= 0
    assert isinstance(SampleRate, float) or isinstance(SampleRate, int),      'The SampleRate input argument must be a numeric type.'
    assert isinstance(Delays, np.ndarray) and \
           np.issubdtype(Delays.dtype, np.floating),                          'The Delays argument must be an np.ndarray of type np.floating.'
    assert isinstance(Constants, np.ndarray) and \
           np.issubdtype(Constants.dtype, np.complexfloating),                'The Constants argument must be an nd.array of type np.complexfloating.'
    assert isinstance(Dopplers, np.ndarray) and \
           np.issubdtype(Dopplers.dtype, np.floating),                        'The Doppler argument must be an np.ndarray of type np.floating.' 
    assert len(Delays) == len(Constants) and len(Delays) == len(Dopplers),    'The length of Delays, Constants, and Doppler lists must be the same.'


    OutputSequence = np.zeros(len(InputSequence) - SampleBackOff * 2, InputSequence.dtype)
    TS             = 1/SampleRate
    x_Ind          = np.arange(0, len(InputSequence), 1, np.int32)
    x              = x_Ind * TS

    x_Ind1         = np.arange(SampleBackOff, len(InputSequence)-SampleBackOff, 1, np.int32)
    x1             = x_Ind1 * TS

    funcR          = interpolate.interp1d(x, InputSequence.real, 'cubic', bounds_error = False, fill_value = 'extrapolate')
    funcI          = interpolate.interp1d(x, InputSequence.imag, 'cubic', bounds_error = False, fill_value = 'extrapolate')

    for PathIndex in range(0, len(Delays)):
        # Delay the input sequence
        x_new                = x1 - Delays[PathIndex]
        InputSequenceDelayed = funcR(x_new) + 1j*funcI(x_new)
        # Add the Doppler to it
        Sinusoid             = Constants[PathIndex] * np.exp(1j*2*np.pi*x_new*Dopplers[PathIndex])
        OutputSequence      += InputSequenceDelayed * Sinusoid
   
    return OutputSequence











# ------------------------------------------------------------------------------------------------------------ #
#                                                                                                              #
# > 6. SpectrumAnalyzer()                                                                                      #
#                                                                                                              #
# ------------------------------------------------------------------------------------------------------------ #
def SpectrumAnalyzer( InputSequence: np.ndarray
                    , SampleRate     
                    , FFT_Size:      int
                    , bPlot:         bool):
    '''
    brief: This function computes the power spectrum of a real or complex sequence
    param: InputSequence - A 1D numpy array representing the sequence to be analyzed
    param: FFT_Size      - The FFT size (Resolution bandwidth = SampleRate/FFT_Size)
    param: bPlot         - A boolean indicating whether to plot the power spectrum
    '''

    # ----------------------------------------------
    # Type checking
    assert isinstance(InputSequence, np.ndarray)
    assert isinstance(SampleRate, float) or isinstance(SampleRate, int)
    assert isinstance(FFT_Size, int)
    assert isinstance(bPlot, bool)

    # Recast the input sequence to a complex type
    InputSequence = InputSequence.astype(np.complex128)

    # ----------------------------------------------
    # Determine the quantities N = FFT_Size, M, and P
    # M is the number of samples in one IQ sections 
    # N is the number of samples in one IQ subsections
    # R is the number of subsections in one full IQ section
    IqLength = len(InputSequence) 
    N        = FFT_Size
    MinR     = 4
    assert IqLength >= N * MinR, 'The number of IQ samples must be >= FFT_Size * 4.'

    R        = MinR
    if IqLength >= N*8:   R = 8
    if IqLength >= N*16:  R = 16
    if IqLength >= N*32:  R = 32
 
    M  = N * R     # M is the number of IQ sample in one section of the InputSequence 
    nn = np.arange(0, M, 1, np.int32)
    k  = np.arange(-np.floor(M/2), np.floor(M/2), 1, np.int32) 

    # ----------------------------------------------
    # Build the desired window
    NumSinusoids   = R + 1
    Ak             = np.ones(NumSinusoids, np.float32)
    Ak[0] = Ak[-1] = np.float32(0.91) # 416)

    Sinusoids      = np.zeros(M, np.complex128)
    for Index in range(0, NumSinusoids):
        f          = ((-R/2) + Index)/M
        Sinusoids += Ak[Index] * np.exp(1j*2*np.pi*k*f)
        Stop = 1

    Hanning       = 0.5 - 0.5 * np.cos(2*np.pi*(nn + 1) / (M + 1))
    DesiredWindow = Sinusoids * Hanning

    # ----------------------------------------------
    # Run through each iteration (section of M samples)
    # We want to use all samples in the IqSequence to compute the power spectrum.
    # Determine how many sections of the M samples are available for spectrum analysis
    NumSections   = int(np.ceil(IqLength/M))
    PowerSpectrum = np.zeros(N, np.complex128)
    for Section in range(0, NumSections): 
        if   Section == 0:
            IqMSequence = InputSequence[0:M]
        elif Section == NumSections-1:
            IqMSequence = InputSequence[(-M-1):-1]    
        else:
            StartPosition = Section * math.floor(IqLength / NumSections)
            StopPosition  = StartPosition + M
            IqMSequence   = InputSequence[StartPosition:StopPosition]
    
        IqMWindowed   = IqMSequence * DesiredWindow

        # -----------------------------------------------
        # Break up each section into R N-sized subsections and add them up
        IqNSequence = np.zeros(N, np.complex128)
        for Subsection in range(0, R):
            StartPosition = Subsection * N
            StopPosition  = StartPosition + N
            IqNSequence  += (1/R) * IqMWindowed[StartPosition:StopPosition]

        # ----------------------------------------------
        # Take the FFT
        Fft            = (1/N) * np.fft.fft(IqNSequence)
        PowerSpectrum += (1/NumSections) * (Fft * np.conj(Fft))

    # Rearrange the power spectrum such that the negative frequencies appear first
    PowerSpectrum = np.abs(np.roll(PowerSpectrum, math.floor(N/2)))
    Frequencies   = np.arange(-0.5*SampleRate, 0.5*SampleRate - 1e-6, SampleRate/N, \
                                                                           np.float32)
    ResolutionBW  = float(SampleRate / N)

    if bPlot == True:
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(Frequencies, np.abs(PowerSpectrum), 'b')
        plt.grid(True)
        plt.title('Linear Power Spectrum')
        plt.xlabel('Hz')
        plt.tight_layout()
        plt.subplot(2,1,2)
        plt.plot(Frequencies, 10*np.log10(np.abs(PowerSpectrum)), 'b')
        plt.grid(True)
        plt.title('Power Spectrum in dB')
        plt.xlabel('Hz')
        plt.ylabel('dB')
        plt.tight_layout()
        plt.show()

    return (PowerSpectrum, Frequencies, ResolutionBW)
















# ------------------------------------------------------------------------------------------------------------ #
#                                                                                                              #
# > 7. ComputeCorrelation()                                                                                    #
#                                                                                                              #
# ------------------------------------------------------------------------------------------------------------ #
def ComputeCorrelation(   domain:        str 
                        , InputWaveform: np.ndarray     
                        , h:             np.ndarray
                        , bPlot:         bool = False) -> np.ndarray:
    '''
    brief:         The following function will compute the sliding correlation of the
                   InputWaveform and the sequence h using either the original time 
                   domain algorithm or the frequency domain algorithm using FFTs.
                   If the Input Wavefrom and the sequence h have the same 
                   length, it will compute the normal correlation.
    domain:        'time'      - The time domain algorithm..  
                   'frequency' - The frequency domain algorithm is much faster than
                   the time domain algorithm in software. In some circumstances we 
                   may revert back to the time domain algorithm.
    InputWaveform: The input waveform within which we search for h.
    h:             This is the sequence for which we will search the InputWaveform
    bPlot:         Plot the result
    Reference:     Digital Signal Processing in ModernCommunication Systems sect. 2.5.2
    '''
    # Type and error checking
    assert isinstance(domain, str)
    assert domain.lower() == 'time' or domain.lower() == 'frequency'
    assert isinstance(InputWaveform, np.ndarray)
    assert isinstance(h, np.ndarray)
    assert isinstance(bPlot, bool)
    assert len(h) <= len(InputWaveform)   
    assert InputWaveform.dtype == h.dtype 
     
    # ------------------------------------------------------------------
    # Handle the case, when the two sequences have the same length
    # ------------------------------------------------------------------
    if len(h) == len(InputWaveform):
        Correlation = np.sum(InputWaveform * np.conj(h))
        return Correlation

    # ------------------------------------------------------------------
    # A lambda that handles the time domain algorithm for the sliding correlation
    # ------------------------------------------------------------------
    def UseTimeDomainAlgorithm(InputWaveform, h):
        L               = len(h)
        conj_h          = np.conj(h)
        NumberOfResults = len(InputWaveform) - len(h) + 1
        Correlation = np.zeros(NumberOfResults, dtype = h.dtype)
        for Index in range(0, NumberOfResults):
           Correlation[Index] = (1/L)*np.sum(InputWaveform[Index : Index + L] * conj_h)
        return Correlation

    # ------------------------------------------------------------------
    # A lambda that handles the frequency domain algorithm for the sliding correlation
    # ------------------------------------------------------------------
    def UseFrequencyDomainAlgorithm(InputWaveform, h, FFT_Size):
        N          = FFT_Size
        L          = len(h)
        M          = N - L  # Number of correlation results produced by each iteration
        assert M > 0        # This would not be good
        
        # zero pad h so it reaches the a length = N
        h_extended = np.hstack([h, np.zeros(M, h.dtype)]) 
        assert len(h_extended) == N
        H          = np.fft.fft(h_extended)
        H_conj     = np.conj(H)
        
        NumberOfResults = len(InputWaveform) - len(h) + 1
        Correlation     = np.zeros(NumberOfResults, dtype = h.dtype)
        NumIterations   = math.ceil((len(InputWaveform) - L) / M)

        # Iterate through all section. First the whole sections then the last section
        for Iteration in range(0, NumIterations):
            if Iteration != NumIterations - 1:   # Do all but the last iteration
                StartIndex  = Iteration * M
                StopIndex   = StartIndex + N
                FT_Waveform = np.fft.fft(InputWaveform[StartIndex:StopIndex], N)
                R           = np.fft.ifft(FT_Waveform * H_conj) 
                Correlation[StartIndex:StartIndex + M] = (1/L)*R[0:M]
            else:                                # Do the last iteration
                StartIndex  = len(InputWaveform) - N
                StopIndex   = StartIndex + N
                FT_Waveform = np.fft.fft(InputWaveform[StartIndex:StopIndex], N)
                R           = np.fft.ifft(FT_Waveform * H_conj) 
                Correlation[Iteration*M:] = (1/L)*R[Iteration*M - StartIndex:-L+1]

        return Correlation
    
    # ------------------------------------------------------------------
    # Handle time domain algorithm to compute sliding correlation
    # ------------------------------------------------------------------
    if domain.lower() == 'time':
        Correlation = UseTimeDomainAlgorithm(InputWaveform, h)
    
    # Handle the frequency domain algorithm to compute sliding correlation. 
    # Especially when h gets long this algorithm is much faster in software
    else:
        # The FFT size can't be bigger than the InputWavform
        # The FFT size should be larger 8 times the sequence h
        FFT_Size = 0
        for i in range(0, 32):
            if 2**i > len(InputWaveform):
                FFT_Size = 2**(i-1)
                break
            
            if 2**i > len(h) * 8:
                FFT_Size = 2**i
                break

        if FFT_Size < len(h):  # Here we can't use the frequency domain technique
            Correlation = UseTimeDomainAlgorithm(InputWaveform, h)
        else:
            Correlation = UseFrequencyDomainAlgorithm(InputWaveform, h, FFT_Size)

    # ------------------------------------------------------------------
    # Plot the sliding correlation result if desired
    # ------------------------------------------------------------------
    if bPlot == True:
        plt.figure()
        plt.plot(np.arange(0, len(Correlation), 1), Correlation.real,    'r')
        plt.plot(np.arange(0, len(Correlation), 1), Correlation.imag,    'b')
        plt.plot(np.arange(0, len(Correlation), 1), np.abs(Correlation), 'k')
        plt.title('Sliding Correlation Result')
        plt.grid(color='#999999') 
        plt.show()

    return Correlation
















# ------------------------------------------------------------------------------------------------------------ #
#                                                                                                              #
# > 8. FilterFrequency()                                                                                       #
#                                                                                                              #
# ------------------------------------------------------------------------------------------------------------ #
def FilterFrequency(InputSequence: np.ndarray     
                  , FFT_Size:      int
                  , SkirtLength:   int
                  , FreqResponse:  np.ndarray  
                  , bReverse:      bool = True
                  , SampleRate:    float = 1.0
                  , bPlot:         bool = False) -> np.ndarray:
    '''
    brief:  The following function filters a complex signal in the frequency domain
    param:  InputSequence - A 1D np.ndarray with real or complex entries
    param:  FFT_Size      - An integer of size 2**n and FFT_Size >= 128
    param:  SkirtLength   - Length of Hanning skirt at the start/end of a section
    param:  FreqResponse  - A 1D np.ndarray with real entries of lenght FFT_Size
    param:  SampleRate    - An integer or floating point number used for plotting 
    param:  bReverse      - If True, the frequency response should list negative
                            frequencies first, positive frequencies last.
                            If False, it is the reverse 
    param:  bPlot         - If True, plot the requested frequency response
    '''
    # --------------------------
    # Error and type checking
    # --------------------------
    assert isinstance(InputSequence, np.ndarray)
    assert np.issubdtype(InputSequence.dtype, np.complexfloating) or \
           np.issubdtype(InputSequence.dtype, np.floating)
    assert len(InputSequence.shape) == 1  # Must be a 1D array
    assert isinstance(FFT_Size, int)
    assert math.fmod(math.log2(16), 1) == 0, 'Ensure that FFT_Size = 2**n'
    assert FFT_Size >= 128
    assert isinstance(SkirtLength, int) and SkirtLength <= int(FFT_Size/4)
    assert isinstance(FreqResponse, np.ndarray) 
    assert np.issubdtype(FreqResponse.dtype, np.floating)
    assert len(FreqResponse) == FFT_Size
    assert isinstance(SampleRate, int) or isinstance(SampleRate, float)
    assert isinstance(bReverse, bool)
    assert isinstance(bPlot, bool)
    assert len(InputSequence) >= FFT_Size

    # ----------------------------
    # Basic setup
    # ----------------------------
    # Skirt is at the start and end of the section
    SLen       = SkirtLength
    RangeSkirt = np.arange(0, SLen, 1, np.int32)
    SkirtUp    = 0.5 - 0.5*np.cos(2*np.pi*(RangeSkirt        + 1) / (2*SLen))
    SkirtDown  = 0.5 - 0.5*np.cos(2*np.pi*(RangeSkirt + SLen + 1) / (2*SLen))

    # Compute the number of sections we need
    NumSectionFract  = (len(InputSequence) - SLen)/(FFT_Size - SLen)
    NumSections      = math.ceil(NumSectionFract)

    # -----------------------------
    # Copy input sequence and apply skirt 
    # -----------------------------
    LenInput = len(InputSequence)
    Input = np.zeros(NumSections * (FFT_Size - SLen) + SLen, dtype = np.complex128)
    Input[0:len(InputSequence)] = InputSequence.astype(np.complex128)
    Input[0:SLen]                   *= SkirtUp.astype(np.complex128)
    Input[LenInput - SLen:LenInput] *= SkirtDown.astype(np.complex128)

    # -----------------------------
    # Change and plot the Frequency Response if needed
    # -----------------------------
    # if bReverse == True, then we need to assume that the frequency response was
    # supplied with negative frequencies first.
    FResponse = FreqResponse.copy()
    if bReverse == True:
        FResponse = np.roll(FResponse, math.floor(FFT_Size/2))

    if bPlot == True:
        FResponsePlotting = np.roll(FResponse, math.floor(FFT_Size/2))
        FStep       = SampleRate / FFT_Size
        Frequencies = np.arange(-SampleRate/2, SampleRate/2 - 0.001, FStep, np.float32)
        assert len(Frequencies) == FFT_Size

        plt.figure()
        plt.plot( Frequencies, FResponsePlotting, 'b')
        plt.grid('cccccc')
        plt.title('The Requested Frequency Response')
        plt.xlabel('Hz')
        plt.show()

    # ----------------------------  
    # Run the filtering operation   
    # ----------------------------  
    Mask = np.hstack([SkirtUp, np.ones(FFT_Size - 2*SkirtLength, SkirtUp.dtype), SkirtDown])
    Mask = Mask.astype(np.complex128)

    Output = np.zeros(len(Input), Input.dtype) 
    FResponse = FResponse.astype(np.complex128)
    for SectionIndex in range(0, NumSections):
        StartIndex            = SectionIndex * (FFT_Size - SkirtLength)
        CurrentRange          = range(StartIndex, StartIndex + FFT_Size)
        CurrentSection        = Input[CurrentRange] * Mask 
        DFT_Section           = np.fft.fft(CurrentSection)
        Y                     = DFT_Section * FResponse
        Output_Section        = np.fft.ifft(Y)
        Output[CurrentRange] += Output_Section

    return Output[0:len(InputSequence)].astype(InputSequence.dtype)















# ------------------------------------------------------------------------------------------------------------ #
#                                                                                                              #
# > 9. ComputeFirFrequencyResponse()                                                                           #
#                                                                                                              #
# ------------------------------------------------------------------------------------------------------------ #
def ComputeFirFrequencyResponse(FirTaps:    np.ndarray
                              , SampleRate: float
                              , Plot:       bool = True) -> tuple:
    '''
    This function will compute the frequency response of a set of FIR coefficients
    '''
    # ----------------------- #
    # Error checking          #
    # ----------------------- #
    assert isinstance(FirTaps, np.ndarray), "The FirTaps argument must be an numpy array"
    assert isinstance(SampleRate, float) or isinstance(SampleRate, int), "The SampleRate type is invalid"
    assert isinstance(Plot, bool)
    assert np.issubdtype(FirTaps.dtype, np.complexfloating) or np.issubdtype(FirTaps.dtype, np.floating)

    Coefficients = FirTaps.astype(np.complex128).copy()
    NumTaps      = len(Coefficients)

    FStep        = 1/(4*NumTaps)
    n            = np.arange(0, NumTaps, 1, np.float64)
    FTest        = np.arange(-0.5, +0.5, FStep, np.float64)
    Output       = np.zeros(len(FTest), np.complex64)

    Index = 0
    for TestFrequency in FTest:
        Sinusoid      = np.exp(-1j*2*np.pi*n*TestFrequency)
        Result        = np.sum(Coefficients * Sinusoid)
        Output[Index] = Result 
        Index += 1

    if Plot == True:
        plt.figure(1)
        plt.subplot(2,1,1)
        plt.plot(n, Coefficients.real, 'r', n, Coefficients.imag, 'b')
        plt.grid(True)
        plt.title('Filter Impulse Response')
        plt.legend(['Real', 'Imag'])
        plt.tight_layout()
        plt.subplot(2,1,2)
        plt.plot(SampleRate * FTest, 20*np.log10(np.abs(Output)), 'k')
        plt.grid(True)
        plt.title('Magnitude Response of Filter')
        plt.xlabel('Hz')
        plt.tight_layout()
        plt.show()

    return (Output, FTest)












# ------------------------------------------------------------------------------------------------------------ #
#                                                                                                              #
# > 10. Resample()                                                                                             #
#                                                                                                              #
# ------------------------------------------------------------------------------------------------------------ #
def Resample(InputSequence:        np.ndarray
           , SourceSampleRate:     float
           , TargetSampleRate:     float
           , StartTimeFirstSample: float
           , bPlot:                bool) -> np.ndarray:
    '''
    brief:  The following function resamples a numpy array of real or complex values from one sample rate to another.
    :    :  This function uses cubic spline interpolation via the scipy import module
    param:  InputSequence        - A 1D np.ndarray with real or complex entries
    param:  SourceSampleRate     - The sample rate associated with the InputSequence  
    param:  TargetSampleRate     - The sample rate associated with the OutputSequence
    param:  StartTimeFirstSample - Select a start time where the first interpolated sample will be rendered  
    param:  bPlot                - If true, we plot the source and target waveform for comparison (debug feature) 
    '''
    # --------------------------
    # Error and type checking
    # --------------------------
    assert isinstance(InputSequence, np.ndarray)
    assert np.issubdtype(InputSequence.dtype, np.complexfloating) or np.issubdtype(InputSequence.dtype, np.floating) or \
           np.issubdtype(InputSequence.dtype, np.integer)
    assert isinstance(SourceSampleRate, float) or isinstance(SourceSampleRate, int)
    assert isinstance(TargetSampleRate, float) or isinstance(TargetSampleRate, int)
    assert isinstance(StartTimeFirstSample, float) or isinstance(StartTimeFirstSample, int)
    assert isinstance(bPlot, bool)
    assert len(InputSequence.shape) == 1, 'The input sequence needs to be a 1D numpy array, not a multidimensional matrix.'
    
    NumberSourceSamples = len(InputSequence)
    if(SourceSampleRate <= TargetSampleRate):
        NumberTargetSamples = int((NumberSourceSamples - 1) * TargetSampleRate/SourceSampleRate)
    else:
        NumberTargetSamples = int((NumberSourceSamples)     * TargetSampleRate/SourceSampleRate)

    SourceIndices    = np.arange(0, NumberSourceSamples, 1, np.int32)
    TargetIndices    = np.arange(0, NumberTargetSamples - int(StartTimeFirstSample * TargetSampleRate), 1, np.int32)
    SourceTime       = SourceIndices / SourceSampleRate
    TargetTime       = StartTimeFirstSample + TargetIndices / TargetSampleRate


    if np.issubdtype(InputSequence.dtype, np.floating):
        funcR          = interpolate.interp1d(SourceTime, InputSequence, 'cubic', bounds_error = False, fill_value = 'extrapolate')
        OutputSequence = funcR(TargetTime)

        if bPlot == True:
            plt.figure()
            plt.plot(SourceTime, InputSequence, 'r-o')
            plt.plot(TargetTime, OutputSequence,'b-x')
            plt.grid(True)
            plt.title('Source vs Resampled Target Waveform')
            plt.legend(['Source','Target'])
            plt.show()
    
    elif np.issubdtype(InputSequence.dtype, np.complexfloating):
        funcR          = interpolate.interp1d(SourceTime, InputSequence.real, 'cubic', bounds_error = False, fill_value = 'extrapolate')   
        funcI          = interpolate.interp1d(SourceTime, InputSequence.imag, 'cubic', bounds_error = False, fill_value = 'extrapolate')
        OutputSequence = funcR(TargetTime) + 1j*funcI(TargetTime)
        
        if bPlot == True:
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(SourceTime, InputSequence.real, 'r-o')
            plt.plot(TargetTime, OutputSequence.real,'b-x')
            plt.grid(True)
            plt.title('Real Portion of Source vs Resampled Target Waveform')
            plt.legend(['Source','Target'])
            plt.tight_layout()
            plt.subplot(2, 1, 2)
            plt.plot(SourceTime, InputSequence.imag, 'r-o')
            plt.plot(TargetTime, OutputSequence.imag,'b-x')
            plt.grid(True)
            plt.title('Imag Portion of Source vs Resampled Target Waveform')
            plt.legend(['Source','Target'])
            plt.tight_layout()
            plt.show()

    else:
        assert False, "Invalid InputSequence type."

    return OutputSequence















# ------------------------------------------------------------
# > Test bench
# ------------------------------------------------------------
if __name__ == '__main__':

    Test = 7



    # -------------------------------------------------------- #
    # Excercise the AddMultipath() function                    #
    # -------------------------------------------------------- #
    if Test == 1:
        InputSequence = np.ones(20, dtype = np.complex64)
        Delays        = [-1, 1, 2]
        Constants     = [1 + 0j, 0, -0.2j]
        Dopplers      = [1000, 0, 5000]
        SampleRate    = 100000
        Output, MinDelay = AddMultipath(InputSequence, SampleRate, Delays, Constants, Dopplers)

        print('MinSampleDelay: ' + str(MinDelay))
        print('MinTimeDelay:   ' + str(MinDelay / SampleRate))

        plt.figure(1)
        plt.plot( Output.real, c = 'red')
        plt.plot( Output.imag, c = 'blue')
        plt.grid(True)
        plt.show()
        stop = 1  # Example call



    # -------------------------------------------------------- #
    # Excercise the AddMultipath2() function                   #
    # -------------------------------------------------------- #
    if Test == 2:
        #InputSequence = np.array([1 -1j, 2 - 2j, 4 - 4j, 8 - 8j], np.complex128)
        InputSequence = np.ones(100, np.complex128)
        SampleRate    = 1
        TS            = 1/SampleRate
        x = np.arange(0, len(InputSequence), 1, np.int32) * TS

        Delays    = np.array([-0.5, 0, 0.5], np.float64)
        Constants = np.array([0.0,  0, -0.2 + 0.4j], np.complex128)
        Doppler   = np.array([20,   .1, -.01], np.float64) 

        Output = AddMultipath2(InputSequence 
                             , SampleRate
                             , Delays 
                             , Constants
                             , Doppler)

        plt.figure()
        plt.plot(x, InputSequence.real, 'ro', x, InputSequence.imag, 'bo')
        plt.plot(x, Output.real, 'r',         x, Output.imag, 'b')
        plt.grid(True)
        plt.show()

    # -------------------------------------------------------- #
    # Test the power spectrum analyzer                         #
    # -------------------------------------------------------- #
    if Test == 3:
        SampleRate      = 1
        TotalNumSamples = 800
        N               = 64
        n               = np.arange(0, TotalNumSamples, 1)
        IqSequence0     = np.exp(1j*2*np.pi*n*0.0625/SampleRate, dtype = np.complex64)
        IqSequence1     = np.exp(1j*2*np.pi*n*0.07/SampleRate, dtype = np.complex64)
        IqSequence2     = np.exp(1j*2*np.pi*n*0.1/SampleRate, dtype = np.complex64)
        MeanSquare      = np.mean(IqSequence0 * np.conj(IqSequence0)).real

        print('Total Power:          ' + str(MeanSquare))

        PowerSpectrum0, Frequencies, ResolutionBW = SpectrumAnalyzer( InputSequence = IqSequence0
                                                                    , SampleRate    = SampleRate
                                                                    , FFT_Size      = N
                                                                    , bPlot         = False)
        print('Total Power Detected: ' + str(np.sum(PowerSpectrum0).real))


        PowerSpectrum1, Frequencies, ResolutionBW = SpectrumAnalyzer( InputSequence = IqSequence1
                                                                    , SampleRate    = SampleRate
                                                                    , FFT_Size      = N
                                                                    , bPlot         = False)
        print('Total Power Detected: ' + str(np.sum(PowerSpectrum1).real))


        PowerSpectrum2, Frequencies, ResolutionBW = SpectrumAnalyzer( InputSequence = IqSequence2
                                                                    , SampleRate    = SampleRate
                                                                    , FFT_Size      = N
                                                                    , bPlot         = False)
        print('Total Power Detected: ' + str(np.sum(PowerSpectrum2).real))

        print('Resolution Bandwidth: ' + str(ResolutionBW) + ' Hz')



    # -------------------------------------------------------- #
    # Test the frequency / time domain correlation algorithm   #
    # -------------------------------------------------------- #
    if Test == 4:
        P      = complex(+0.7071  + 0.7071j)
        N      = complex(-0.7071  - 0.7071j)
        Barker = np.array([P, P, P,  N, N, N,  P,  N, N,  P,  N], np.complex64)

        SnrDb      = 50
        SnrLinear  = 10**(SnrDb/10)
        NoisePower = 1 / SnrLinear
        Scalar     = np.sqrt(NoisePower) * np.sqrt(1/2)
        Noise      = Scalar * np.array(np.random.randn(15) + 1j*np.random.randn(15), dtype = Barker.dtype)  

        Waveform          = Noise 
        Waveform[ 0: 11] += Barker
        #Waveform[130:141] += Barker

        Correlation1 = ComputeCorrelation(   domain    = 'time' 
                                           , InputWaveform = Waveform     
                                           , h             = Barker 
                                           , bPlot         = True)

        Correlation2 = ComputeCorrelation(   domain    = 'frequency' 
                                           , InputWaveform = Waveform     
                                           , h             = Barker 
                                           , bPlot         = False)

        plt.figure(2)
        plt.plot(np.arange(0, len(Correlation2), 1), Correlation2.real,    'r')
        plt.plot(np.arange(0, len(Correlation2), 1), Correlation2.imag,    'b')
        plt.plot(np.arange(0, len(Correlation2), 1), np.abs(Correlation2), 'k')
        plt.title('Sliding Correlation Result')
        plt.grid(color='#999999') 
        plt.show()

        Error = (Correlation2 - Correlation1)

        plt.figure(3)
        plt.plot(np.arange(0, len(Correlation2), 1), np.abs(Error),    'r')
        plt.title('Error')
        plt.grid(color='#999999') 
        plt.show()


    # ---------------------------------------------------- #
    # Test the frequency domain filtering function         #
    # ---------------------------------------------------- #
    if Test == 5:
        NumSamples = 8000
        SampleRate = 1000
        if False:
            InputSequence        = GenerateAwgn(NumSamples, 'complex64', 1)
        else:
            n                    = np.arange(0, NumSamples, 1, np.int32)
            InputSequence        = np.cos(-2*np.pi*n*480/SampleRate) + np.cos(2*np.pi*n*22/SampleRate)

        FFT_Size             = 2048
        SkirtLength          = 128
        FreqResponse         = np.zeros(FFT_Size, np.float32)
        #FreqResponse[0:41]   = np.ones(41, np.float32)  # Set up for +/- 20Hz filtering
        #FreqResponse[-40:]   = np.ones(40, np.float32)
        FreqResponse[0:160]    = np.ones(160, np.float32)
        
        OutputSequence = FilterFrequency(InputSequence 
                                       , FFT_Size
                                       , SkirtLength
                                       , FreqResponse 
                                       , bReverse   = False
                                       , SampleRate = SampleRate
                                       , bPlot      = True) 

        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(range(0, len(InputSequence)), InputSequence.real, 'r')
        plt.plot(range(0, len(InputSequence)), InputSequence.imag, 'b')
        plt.grid(color='#999999')
        plt.subplot(2,1,2)
        plt.plot(range(0, len(OutputSequence)), OutputSequence.real, 'r')
        plt.plot(range(0, len(OutputSequence)), OutputSequence.imag, 'b')
        plt.grid(color='#999999')
        plt.show()


        
    # ---------------------------------------------------- #
    # Test the ComputeFirFrequencyResponse()               #
    # ---------------------------------------------------- #
    if Test == 6:
         # We will load a FIR filter coefficient file produced by the P25Scanner
        # See ScannerWorker.cpp (ProcessMeasToolFrame())
        with open(file = "C:\\temp\\FirFilterTaps.txt", mode = 'r', encoding = 'utf-8') as f: 

            StringAll   = f.read()
        
        StringList  = StringAll.replace('\n', ' ')
        StringList  = StringList .split(' ') # ["Let's", "add", "a", "line", "here\n"] 

        # Replace all strings that are not numbers by ''
        for Index, String in enumerate(StringList):
            try:
                A = float(String) # if this fails, then it was not a number
            except:
                StringList[Index] = ''
        
        # Remove all entries in the list that are equal to ''
        while(True):
            try:
                StringList.remove('')
            except:
                break

        SampleRate           = float(StringList[0])
        DownconversionFactor = int(StringList[1]) 
        NumTaps              = int(StringList[2])     

        FilterTaps = np.zeros(NumTaps, np.complex128)
        TextCount  = 3
        for Index in range(0, NumTaps):
             TextCount += 1
             Real       = float(StringList[TextCount])
             TextCount += 1
             Imag       = float(StringList[TextCount])
             TextCount += 1
             FilterTaps[Index] = Real + 1j * Imag


        ComputeFirFrequencyResponse(FilterTaps
                                  , SampleRate * DownconversionFactor
                                  , True)
            

    # -------------------------------------------------------
    # Test the Resample() function
    # -------------------------------------------------------
    if Test == 7:
        # -----------------------------------------
        # Resampling sinusoids
        # -----------------------------------------
        SourceSampleRate = 1
        TargetSampleRate = 5

        NumSamples = 100
        n          = np.arange(0, NumSamples, 1, np.int32)
        Frequency  = 0.15

        InputSequence = np.cos(2*np.pi*n*Frequency/SourceSampleRate) + 1j*np.cos(2*np.pi*n*Frequency/SourceSampleRate)
        
        Resample(InputSequence 
               , SourceSampleRate 
               , TargetSampleRate 
               , 2.5
               , False)
        
        # -----------------------------------------
        # Resampling noise
        # -----------------------------------------
        # Assume we have a white noise sequence at a rate of 12.5KHz. I want to resample the noise and look at the spectrum
        SourceSampleRate = 12.5e3
        TargetSampleRate = 60.0e3

        AwgnNoise = np.random.randn(2000)
        ResampledNoise = Resample(AwgnNoise 
                                , SourceSampleRate 
                                , TargetSampleRate 
                                , 0
                                , False)
        
        PowerSpectrum0, Frequencies, ResolutionBW = SpectrumAnalyzer( InputSequence = ResampledNoise
                                                                    , SampleRate    = TargetSampleRate
                                                                    , FFT_Size      = 1024
                                                                    , bPlot         = True)