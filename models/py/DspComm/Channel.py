# File:       Channel.py
# Author:     Andreas Schwarzinger  
# Notes:      This module creates a model of the wireless channel       

__title__     = "Channel"
__author__    = "Andreas Schwarzinger"
__status__    = "preliminary"
__version__   = "0.1.0.1"
__date__      = "Feb, 23, 2022"
__license__   = 'MIT Licence'

import numpy             as np
import matplotlib.pyplot as plt



# ------------------------------------------------------------------------
# > Function: Create White Gaussian Noise
# ------------------------------------------------------------------------
def CreateGaussianNoise(fLinearNoisePower: float
                      , iNumberOfSamples:  int
                      , iSeed:             int  = -1
                      , bComplexNoise:     bool = False) -> np.ndarray:

    '''
    brief: This function produces a real or complex gaussian noise sequence
    param: fLinearNoisePower - The dimensionless linear noise power
    param: iNumberOfSamples  - The number of samples in our noise sequence
    param: iSeed             - The random seed. If == -1 or None (Different every time we call)
    param: bComplexNoise     - True/False -> Complex/Real Noise
    '''

    # ---------------------------------------
    # Error checking
    assert isinstance(fLinearNoisePower, float),    'The fLinearNoisePower argument must be of type float'
    assert isinstance(iNumberOfSamples,    int),    'The iNumberOfSamples argument must be of type int'
    assert isinstance(iSeed, int) or iSeed == None, 'The iSeed argument must be of type int or None' 
    assert isinstance(bComplexNoise, bool)     ,    'The bComplexNoise argument must be of type'
    assert iNumberOfSamples > 0                ,    'The number of noise samples to generate must be > 0.'

    # ----------------------------------------
    # Select a seed for the random number generator
    if iSeed == None or iSeed < 0:
        r = np.random.RandomState(None)    # Seed should vary based on time
    else:
        r = np.random.RandomState(iSeed)   # Seed is fixed

    # ----------------------------------------
    if bComplexNoise == True:
        Scalar       = np.sqrt(fLinearNoisePower) * 0.7071
        ComplexNoise = Scalar * np.complex64(r.randn(iNumberOfSamples) +  \
                                        1j * r.randn(iNumberOfSamples)) 
        return ComplexNoise
    else:
        Scalar       = np.sqrt(fLinearNoisePower) 
        RealNoise    = Scalar * np.float32(r.randn(iNumberOfSamples))
        return RealNoise





# -------------------------------------------------------------------------
# > Function: Add White Gaussian Noise
# -------------------------------------------------------------------------
def AddGaussianNoise(fSnrDb:            float
                   , InputSequence:     np.ndarray
                   , iSeed:             int  = -1
                   , bCopy:             bool = True
                   ) -> np.ndarray:

    '''
    brief: This function adds either complex or real noise to the input sequence
    param: fSnrDb        - The desired signal to noise ratio in dB
    param: InputSequence - Either a real or complex input sequence of type np.ndarray
    param: iSeed         - The random seed. If == -1 or None (Different every time we call)
    param: bCopy         - True/False -> Generates seperate output sequence / Overlays noise
    '''

    # -----------------------------------------
    # Error checking
    assert isinstance(fSnrDb,             float), 'The fLinearNoisePower argument must be of type float'
    assert isinstance(InputSequence, np.ndarray), 'The InputSequence argument must be of type np.array'
    assert isinstance(iSeed, int) or iSeed == None, 'The iSeed argument must be of type int or None' 
    assert isinstance(bCopy, bool)              , 'The bCopy argument must be of type bool'
    
    bInputIsComplex = isinstance(InputSequence[0], np.complex64) or isinstance(InputSequence[0], np.complex128)
    bInputIsReal    = isinstance(InputSequence[0], np.float32)   or isinstance(InputSequence[0], np.float64)
    assert bInputIsComplex or bInputIsReal, 'The input sequence must be complex64/128 or real32/64'
    
    # -----------------------------------------
    # Start processing
    fLinearSnr = 10 ** (fSnrDb/10)
    N          = len(InputSequence)

    # Find the mean square of the sequence
    if bInputIsComplex:
        fMeanSquare = (1/N) * np.sum(InputSequence * np.conj(InputSequence)).real
    else:
        fMeanSquare = (1/N) * np.sum(InputSequence * InputSequence)

    # Determine the needed noise power and generate the Gaussian noise signal
    fNoisePower = float(fMeanSquare/fLinearSnr)
    NoiseSequence = CreateGaussianNoise(fNoisePower, N, iSeed, bInputIsComplex)

    # Do we want to overwrite the InputSequence object or make a copy of it before adding noise.
    if bCopy:  # Do not change the InputSequence vector
        Output = InputSequence.copy() + NoiseSequence 
        return Output
    else:      # Overwrite the InputSequence with its noisy counterpart
        InputSequence += NoiseSequence
        return None








# ------------------------------------------------------------------------
# Class Definition: CMultipathChannel
# ------------------------------------------------------------------------
class CMultipathChannel():
    """
    Notes: This channel model represents a set of multipaths defined by a complex amplityde
    a delay in seconds, and a doppler frequency in Hz.
    """
    def __init__(self):
        self.CMultipathChannel   = {}
        self.PathIndex           = 0


    # -----------------------------------------------------------------
    # Function: AddPath()
    # -----------------------------------------------------------------
    def AddPath(self
              , Amplitude:                 np.complex64
              , DelayInSec:                np.float32
              , DopplerHz:                 np.float32   
              , RoundToNearestSampleTime:  bool       = False
              , SampleRate:                np.float32 = None):
        """
        brief: Allows us to add paths to the multipath channel
        param: Amplitude  -> The complex valued amplitude of the path
        param: DelayInSec -> The delay in seconds of the path compared to the FFT time in the receiver
        param: DopplerHz  -> The Doppler frequency in Hz due to the motion of the transceivers.
        param: RoundToNearestSampleTime -> Round the delay in seconds to the nearest sample time 
        param: SampleRate -> Compute the nearest sample time via the sample period 1/SampleRate
        """

        assert DelayInSec > -50e-6 and DelayInSec < 50e-6, 'The path delay of ' + str(DelayInSec) + \
                                                           ' seconds is outside of the valid range of +/- 50usec.'
        if (RoundToNearestSampleTime == True and SampleRate!= None):
            SampleStep = 1/SampleRate
            DelayInSec = SampleStep * round(DelayInSec / SampleStep)

        self.CMultipathChannel[self.PathIndex] = [Amplitude, DelayInSec, DopplerHz]
        self.PathIndex += 1


    # -----------------------------------------------------------------
    # Function: ResetCMultipathChannel()
    # -----------------------------------------------------------------
    def ResetMultipathChannel(self):
        """
        brief: This function resets the internal Multipath channel dictionary for new use
        """
        self.CMultipathChannel = {}
        self.PathIndex          = 0


    # -----------------------------------------------------------------
    # Function: ComputeFreqResponse()
    # -----------------------------------------------------------------
    def ComputeFreqResponse(self
                          , Coordinates: np.ndarray) -> np.array:
        """
        brief: ComputeFreqResponse computes the frequency response of the channel using the following parameters
        param: Coordinates -> A 2xN array holding frequency (Hz) in row 1 and time (seconds) in row 2
        """
        (NumRows, NumElements) = Coordinates.shape
        assert NumRows == 2, 'The number of rows of the Coordinate matrix must be 2'
        
        FreqResponse = np.zeros(NumElements, dtype = np.complex64)

        for ElementIndex in range(0, NumElements):
            FrequencyHz = Coordinates[0][ElementIndex]
            TimeSec     = Coordinates[1][ElementIndex]
            for PathIndex in range(0, self.PathIndex):
                [Amplitude, DelayInSec, DopplerHz] = self.CMultipathChannel[PathIndex]  
                FreqResponse[ElementIndex] += Amplitude * np.exp(-1j*2*np.pi*DelayInSec*FrequencyHz) * \
                                                          np.exp(1j*2*np.pi*TimeSec*DopplerHz)

        return FreqResponse


    # -----------------------------------------------------------------
    # Function: PlotFreqResponse()
    # -----------------------------------------------------------------
    def PlotFrequencyResponse(self
                            , time:        float
                            , frequencies: np.ndarray):
        """
        brief: This function plots the frequency response of the channel at a particular time and frequency sequence
        param: time        - A floating point scalar representing an instance in time.
        param: frequencies - A sequence of frequencies were to plot the frequency response 
        """
        # Build the 2D coordinate array. Remember that the frequency is in row 1, and time in row 2
        NumberOfElements  = len(frequencies)
        Coordinates       = np.zeros((2, NumberOfElements), dtype = np.float32)
        Coordinates[0, :] = frequencies
        Coordinates[1, :] = time * np.ones((1, NumberOfElements), dtype = np.float32) 

        # Compute and plot the frequency response
        FrequencyResponse = self.ComputeFreqResponse(Coordinates)

        # fig = plt.figure(1, figsize=(6,7))
        ax1 = plt.subplot(2, 1, 1)
        plt.plot(frequencies, FrequencyResponse.real, 'r', frequencies, FrequencyResponse.imag, 'b')
        plt.plot(frequencies, np.abs(FrequencyResponse), 'k')
        plt.title('Magnitude and IQ Response')
        plt.xlabel('Frequency in Hz')
        plt.tight_layout()
        plt.legend(['Real', 'Imag', 'Mag'], loc='lower right')
        ax1.grid()
        ax2 = plt.subplot(2, 1, 2)
        plt.plot(frequencies, np.angle(FrequencyResponse), 'k')
        plt.title('Phase Response')
        plt.xlabel('Frequency in Hz')
        plt.ylabel('radians')
        plt.tight_layout()
        ax2.grid()
        plt.show()


    # -----------------------------------------------------------------
    # Function: ConvolveTimeDomainSignal()
    # -----------------------------------------------------------------
    def ConvolveTimeDomainSignal(self
                               , InputSignal: np.ndarray
                               , StartTime:   np.float64
                               , SampleRate:  np.float64) -> np.ndarray:
        """
        brief: This function computes the time domain output signal of the channel.
        param: InputSignal  ->  A numpy array representing the complex time domain input signal.
        param: Time -> A numpy array representing the equally spaced time value of each sample
        note1: The sample rate of the sequency can be computed via the time step in Time.
        note2: We will use exterpolation in time by using the closes sample value
        """
        assert type(InputSignal[0]) == np.complex64, 'The Input signal must be of type numpy.complex64'

        # Same constants
        NumElements  = len(InputSignal)
        SampleStep   = 1/SampleRate
        StopTime     = StartTime + NumElements * SampleStep

        # Create memory for the output signal
        OutputSignal = np.zeros(NumElements, dtype = np.complex64)
    
        # Compute the OutputSignal
        for PathIndex in range(0, self.PathIndex):
            [Amplitude, DelayInSec, DopplerHz] = self.CMultipathChannel[PathIndex]
            
            # The time assigned to the doppler wave
            DopplerTime     = np.arange(StartTime, StopTime, SampleStep)
            DopplerSignal   = Amplitude * np.exp(1j*2*np.pi*DopplerTime*DopplerHz)  
            SignalNoDelay   = InputSignal * DopplerSignal
            SignalWithDelay = np.interp(DopplerTime - DelayInSec,  DopplerTime, SignalNoDelay)

            OutputSignal   += SignalWithDelay

        return OutputSignal











# ---------------------------------------------------------
# Function: The test bench
# ---------------------------------------------------------

if __name__ == "__main__":
    MyChannel = CMultipathChannel()

    Test = 5

    # -------------------------------------------------------------------------------
    # Testing the PlotFrequencyResponse() and ComputeFrequencyResponse() functions
    # -------------------------------------------------------------------------------
    if(Test == 1):
        MyChannel.AddPath(1+0.j, 1e-6, 100)
        MyChannel.AddPath(0+0.0j, -2e-6, 100)
        Frequencies = np.arange(-15e5, 15e5+15e3, 15e3, dtype = np.float32)
        MyChannel.PlotFrequencyResponse(0, Frequencies)


    # -------------------------------------------------------------
    # Testing the ConvolveTimeDomainSignal() function
    # -------------------------------------------------------------
    if(Test == 2):
        SampleRate   = 1e6
        SamplePeriod = 1/SampleRate
        
        MyChannel.AddPath(1+0.j, 3*SamplePeriod, -10000)
        MyChannel.AddPath(1+0.j, 1*SamplePeriod,  1000)

        NumElements  = 1000
        StartTime    = 0
        TimeSpan     = NumElements * SamplePeriod
        Time         = np.arange(StartTime, TimeSpan, SamplePeriod)
        InputSignal  = np.exp(1j*2*np.pi*Time*0, dtype = np.complex64)

        OutputSignal = MyChannel.ConvolveTimeDomainSignal(InputSignal, StartTime, SampleRate)

        ax1 = plt.subplot(1, 1, 1)
        plt.plot(Time, OutputSignal.real, 'r', Time, OutputSignal.imag, 'b')
        plt.xlabel('Time in seconds')
        ax1.grid()
        plt.show()


    # -------------------------------------------------------------
    # Testing the CreateGaussianNoise() function
    # -------------------------------------------------------------
    if(Test == 3):
        fLinearNoisePower = float(2)
        iNumberOfSamples  = int(1000)
        iSeed             = -1
        bComplexNoise     = False
        Noise = CreateGaussianNoise(fLinearNoisePower 
                                , iNumberOfSamples 
                                , iSeed 
                                , bComplexNoise)

        if bComplexNoise:
            fMeanSquare = (1/iNumberOfSamples) * np.sum(Noise * np.conj(Noise)).real
        else:
            fMeanSquare = (1/iNumberOfSamples) * np.sum(Noise * Noise)
                    
        print('Desired fLinearNoisePower = '     + str(fLinearNoisePower) + \
              '   Measured fLinearNoisePower = ' + str(fMeanSquare) )


    # -------------------------------------------------------------
    # Testing the AddGaussianNoise() function
    # -------------------------------------------------------------
    if(Test == 4):
        fSnrLinear        = float(2)
        InputSequence     = np.ones(100, dtype = np.complex128)
        iSeed             = 0
        bCopy             = True

        Output = AddGaussianNoise(fSnrLinear 
                                , InputSequence 
                                , iSeed 
                                , bCopy)

        plt.figure(1, figsize=(6,7))
        if(bCopy):
            plt.plot(np.arange(0, len(Output)), Output.real, 'r', np.arange(0, len(Output)), Output.imag, 'b')
        else:
            plt.plot(np.arange(0, len(InputSequence)), InputSequence.real, 'r', \
                     np.arange(0, len(InputSequence)), InputSequence.imag, 'b')
        
        plt.grid(True)
        plt.show()

        

    # -------------------------------------------------------------------------------
    # Testing Delay Detector
    # -------------------------------------------------------------------------------
    if(Test == 5):
        # The following code means to simulate two methods to determine the time delay
        # of a multipath channel given that we have the frequency response.
        # --------------------
        # Method 1 - Compute the average of the time delays of all paths
        # Method 2 - Compute the time delay of the largest path
        #
        # 1.  Defining the multipath channel and signal to noise ratio of the measured frequency response
        # 1a. Define Signal to Noise ratio in dB
        fSnrDb       = float(20)

        # 1b. Define the multipath channel
        MyChannel.AddPath(0.2 + 0.0j, 0.1e-6,    100)
        MyChannel.AddPath(0.0 + 0.0j, 0.6e-6,   -300)
        MyChannel.AddPath(0.0 + 0.9j, -2.3e-6,   -100)
        MyChannel.AddPath(0.0 - 0.2j, 3.0e-6,    200)

        # 2.  Define the size of the PSSCH and compute the the frequency response of the multipath channel
        # 2a. Define the PSSCH size and associated subcarrier frequencies
        NumPRB            = 20             # The number of physical resource blocks in the PSSCH
        N                 = NumPRB * 12   # The number of subcarriers in the PSSCH
        FreqStart         = 0 * 15e3      # The frequency of the first subcarrier. Arbitrarily = 0
        FreqStop          = N * 15e3      # The frequency of the last + 1 subcarrier.
        frequencies       = np.arange(FreqStart, FreqStop, 15e3, dtype = np.float32)
        
        print('N:          ' + str(N))
        print('Resolution: ' + str(1 / (15000 * N)) + ' seconds')
        
        # 2b. The Compute frequency response requires a Matrix of coordinates. The matrix is of size 2xN
        #     The first row holds the  frequency portion of the coordinate
        #     The second row holds the time portion of the coordinate
        Coordinates       = np.zeros((2, N), dtype = np.float32)
        Coordinates[0, :] = frequencies
        Coordinates[1, :] = 0 * np.ones((1, N), dtype = np.float32) # Arbitrarily se to 0
        FreqResponse      = MyChannel.ComputeFreqResponse(Coordinates)

        # 3.   Corrupt the frequency response with noise
        iSeed = int(-1)    # Ensure that the noise has a random seed everytime we run this command
                           # Ensure that we don't overwrite the FreqResponse vector (bCopy = True)
        FreqResponseNoisy  = AddGaussianNoise(fSnrDb, FreqResponse, iSeed, bCopy = True)
        
        # 4.    Plot the frequency response
        ax1 = plt.subplot(2, 1, 1)
        plt.plot(frequencies, FreqResponseNoisy.real, 'r', frequencies, FreqResponseNoisy.imag, 'b')
        plt.plot(frequencies, np.abs(FreqResponseNoisy), 'k')
        plt.xlabel('Frequency in Hz')
        plt.tight_layout()
        plt.legend(['Real', 'Imag', 'Mag'], loc='lower right')
        ax1.grid()
        ax2 = plt.subplot(2, 1, 2)
        plt.plot(frequencies, np.angle(FreqResponseNoisy), 'k')
        plt.xlabel('Frequency in Hz')
        plt.ylabel('radians')
        plt.tight_layout()
        ax2.grid()
        #plt.show()


        # -------------------------------------------------------------------------------------
        # 5. Find the average delay (Method 1)
        # -------------------------------------------------------------------------------------
        #    Compute the rotation of the frequency response as we move from negative to positive frequencies
        AverageRotation = np.mean(FreqResponseNoisy[1:] * np.conj(FreqResponseNoisy[0:-1]))
        #    What is the average rotation and associated delay
        AngleRadians    = np.angle(AverageRotation)
        AverageDelay    = -AngleRadians/(2*3.1415*15000)
        print("Average Delay Method = " + str(AverageDelay))


        # -------------------------------------------------------------------------------------
        # 6. Find the delay of the strongest path (Method 2)
        # -------------------------------------------------------------------------------------
        # 6a. The main approach to computing the delay of the strongest path is to take the ifft 
        #     of the frequency response. As the peak may be in between bins, we zero stuff the
        #     noisy frequency response to increase the resolution of our plot. The subcarrier spacing
        #     doesn't change, we just see values in between the original bins that are spaced at 15000.
        if   N > 256:
            NewFreqResponseNoisy = FreqResponseNoisy[0:256]
        else:
            NewFreqResponseNoisy = np.hstack([FreqResponseNoisy, np.zeros((256 - N), dtype = np.float32)])

        FFT_Output = np.fft.ifft(NewFreqResponseNoisy)

        # 6b. Find the index of the maximum peak. Notice, as an index > N_OVR/2 indicates a negative time,
        #     undo the aliasing effect by subtracting N_OVR to map indices to their proper negative domain.
        IndexMax   = np.argmax(np.abs(FFT_Output))
        if IndexMax > int(256 / 2):
            IndexMax -= 256

        # 6c. Determine the time delay associated with the maximum index
        PeakDelay     = IndexMax / (256 * 15e3)
        print("Peak Delay Method    = " + str(PeakDelay))

        plt.figure(2)
        plt.plot(np.abs(FFT_Output))
        plt.title('The IFFT of the Zero Stuffed Frequency Response')
        plt.xlabel('Zero Stuffed Bins')
        plt.ylabel('Magnitude')
        plt.grid(True)       
        plt.show()


        

        #if       NumPRB < 6:
        #    OVR   = 8
        #elif  NumPRB < 10:
        #    OVR   = 4
        #elif  NumPRB < 20:
        #    OVR   = 2
        #else:
        #    OVR   = 1
        
        #ZeroStuffedFreqResponseNoisy = np.hstack([FreqResponseNoisy, np.zeros((OVR - 1)*N, dtype = np.float32)])
        #N_OVR      = N * OVR
        #FFT_Output = np.fft.ifft(ZeroStuffedFreqResponseNoisy)

        # 6b. Find the index of the maximum peak. Notice, as an index > N_OVR/2 indicates a negative time,
        #     undo the aliasing effect by subtracting N_OVR to map indices to their proper negative domain.
        #IndexMax   = np.argmax(np.abs(FFT_Output))
        #if IndexMax > int(N_OVR / 2):
        #    IndexMax -= N_OVR

        # 6c. Determine the time delay associated with the maximum index
        #PeakDelay     = IndexMax / (N_OVR * 15e3)
        #print("Peak Delay Method    = " + str(PeakDelay))

        #plt.figure(2)
        #plt.plot(np.abs(FFT_Output))
        #plt.title('The IFFT of the Zero Stuffed Frequency Response')
        #plt.xlabel('Zero Stuffed Bins')
        #plt.ylabel('Magnitude')
        #plt.grid(True)       
        #plt.show()

    