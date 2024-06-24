# File:       SpectrumAnalyzer.py
# Notes:      This script supports spectrum analysis

__title__     = "SpectrumAnalyzer"
__author__    = "Andreas Schwarzinger"
__status__    = "preliminary"
__date__      = "Oct, 9th, 2022"
__copyright__ = 'Andreas Schwarzinger'


# Import Statements
import numpy             as np
import math
import matplotlib.pyplot as plt


# ---------------------------------------------------------------
# > Spectrum Analyzer function
# ---------------------------------------------------------------
def PowerSpectrumAnalyzer (IqSequence:   np.ndarray
                         , SampleRate    # can be float or int 
                         , FFT_Size:     int
                         , bPlot:        bool = False):
    '''
    brief: This function uses the window presum FFT (Window Overlap Add) mechanism
    param: IqSequence   - This can be either a list or np.ndarray of numbers
    param: SampleRate   - The rate at which the IQ waveform was sampled
    param: FFT_Size     - Self Explanatory
    param: bPlot        - Plot the spectrum or not  
    '''
    # ---------------------------
    # Type checking 
    assert isinstance(IqSequence, list) or isinstance(IqSequence, np.ndarray), 'IqSequence is of invalid type'
    assert isinstance(SampleRate, int) or isinstance(SampleRate, float),       'SamlpeRate features invalid type'
    assert isinstance(FFT_Size, int)                                         , 'The FFT_Size must be an integer'
    assert isinstance(bPlot, bool)                                           , 'bPlot must be of type bool'

    # Convert whatever the IqSequence is into an np.ndarray of type np.complex64
    if isinstance(IqSequence, list):
        IqSequenceNew = np.array(IqSequence, dtype = np.complex64)
    else:
        IqSequenceNew = IqSequence.astype(np.complex64)

    # --------------------------
    # Error checking
    IqLength = len(IqSequenceNew)
    MinR = 4
    assert IqLength >= N * MinR, 'The Iq Signal has fewer than N * ' + str(MinR) + ' samples. Chose a smaller N'

    R = MinR                         # R is the ratio of M/N
    if IqLength >= N * 8:  R = 8
    if IqLength >= N * 16: R = 16    # We will not consider larger R than 16

    # M is the number of samples used in the presum FFT
    M  = R * N
    nn = np.arange(0, M, 1, np.int32) 
    k  = np.arange(-int(M/2), int(M/2), 1, np.int32)

    # ------------------------
    # Develop the desired window function.
    NumSinusoids = R + 1
    Ak = np.ones(NumSinusoids, np.float32)
    Ak[0]  = 0.91416
    Ak[-1] = 0.91416

    ImpulseResponse = np.zeros(M, dtype = np.complex64)
    for Index in range(0, NumSinusoids):
        f                = (-int(R/2) + Index)/M
        ImpulseResponse += Ak[Index] * np.exp(1j*2*np.pi*k*f) 

    # Compute the Hanning window and overlay it on top of the impulse response
    Hanning        = 0.5 - 0.5 * np.cos(2*np.pi * (nn + 1) / (M + 1)) 
    DesiredWindow  = Hanning * ImpulseResponse

    # ----------------------------------------
    # We want to use all samples in the IqWaveform to compute the spectrum.
    # Determine how many sections of M samples are available for spectrum analysis
    NumSections = int(math.ceil(IqLength/M))
    
    # --------------------------
    # Run the spectrum analysis for each section
    PowerSpectrum  = np.zeros(N, np.float32)
    for Iteration in range(0, NumSections):
        IqMSequence = np.zeros(M, IqSequenceNew.dtype)
        if Iteration == 0:
            IqMSequence      = IqSequenceNew[:M]    # The last M samples
        elif Iteration == NumSections - 1:
            IqMSequence      = IqSequenceNew[-M:]   # The last M samples
        else:
            StartPosition    = Iteration * int(math.floor(IqLength/NumSections))
            StopPosition     = StartPosition + M
            IqMSequence      = IqSequenceNew[StartPosition:StopPosition]
        
        # Apply the Desired window to the current length M section.
        IqMWindowed          = IqMSequence * DesiredWindow

        # ------------------------------------------------------    
        # We now need to break up this length M sequence into sub-sequences 
        # of length N and add them all up.
        IqNSequence = np.zeros(N, dtype = np.complex64)
        for IndexN in range(0, R):
            StartPosition    = IndexN * N
            StopPosition     = StartPosition + N
            IqNSequence     += (1/R)*IqMWindowed[StartPosition:StopPosition]

        # Take the FFT and average the magnitude spectra.
        Fft                = (1/N)*np.fft.fft(IqNSequence)
        PowerSpectrum += (1/NumSections) * (Fft * np.conj(Fft)).real  
 
    # We need to flip the positive and negative sides of the FFT Output and generate the accompanying frequencies
    PowerSpectrumRearranged = np.hstack([PowerSpectrum[int(N/2):], PowerSpectrum[:int(N/2)]])
    Frequencies                 = np.arange(-0.5 * SampleRate, 0.5 * SampleRate, SampleRate/N, np.float32)
    ResolutionBw                = SampleRate / N

    if bPlot == True:
        plt.figure()    
        plt.plot(Frequencies, PowerSpectrumRearranged, 'k-o')
        plt.title('Power Spectrum (Resolution BW = ' + str(ResolutionBw) + 'Hz)')
        plt.xlabel('Hz')
        plt.grid(True)
        plt.show()


    return PowerSpectrumRearranged, Frequencies, ResolutionBw





# ------------------------------------------------------
# Test Bench
# ------------------------------------------------------
if __name__ == '__main__':
    # Test the SpectrumAnalyzer() function
    SampleRate         = 20.48e6    # SubcarrierSpacing = 160000 for N = 128
                                    # 80KHz / 40KHz/ 20KHz for N = 256/512/1024
     
    NumSamples         = 20000
    N                  = 512
    n                  = np.arange(0, NumSamples, 1, np.uint32)
    IqSequence0        = np.exp(1j*2*np.pi*n*40e3/SampleRate) + np.exp(1j*2*np.pi*n*35e3/SampleRate)
    IqSequence1        = np.exp(1j*2*np.pi*n*80e3/SampleRate)
    IqSequence2        = np.exp(1j*2*np.pi*n*200e3/SampleRate)

    MagSpectrum0, Frequencies, BW0 = PowerSpectrumAnalyzer(IqSequence0, SampleRate, N)
    MagSpectrum1, Frequencies, BW1 = PowerSpectrumAnalyzer(IqSequence1, SampleRate, N)
    MagSpectrum2, Frequencies, BW2 = PowerSpectrumAnalyzer(IqSequence2, SampleRate, N)

     

    plt.figure(1)    
    plt.stem(Frequencies, MagSpectrum0, linefmt='k')
    plt.stem(Frequencies, MagSpectrum1, linefmt='r')
    plt.stem(Frequencies, MagSpectrum2, linefmt='b')
    plt.title('Power Spectrum with RB = ' + str(BW0) + 'Hz')
    plt.xlabel('Hz')
    #plt.grid(True)
    plt.show()