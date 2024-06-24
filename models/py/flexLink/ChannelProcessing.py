# File:       ChannelProcessing.py
# Notes:      This script supports a number of tasks including the following:
#             1. ComputeDopplerFrequency(VelocityKmh, CenterFrequencyHz)
#             2. 
#

__title__     = "ChannelProcessing"
__author__    = "Andreas Schwarzinger"
__status__    = "preliminary"
__date__      = "Sept, 30rd, 2022"
__copyright__ = 'Andreas Schwarzinger'


# ---------------------------------------------------------------#
# Import Modules                                                 #
# ---------------------------------------------------------------#
import numpy              as np
from   FlexLinkParameters import *
import random
import matplotlib.pyplot as plt
import math



# ---------------------------------------------------------------#
# Stand Alone Functions                                          #
# ---------------------------------------------------------------#



# ---------------------------------------------------------------
# > Function: Compute Doppler Frequency for terminals in motion
# ---------------------------------------------------------------
def ComputeDopplerFrequency(VelocityKmh:       float   # Velocity of mobile terminal is in km/h. The eNodeB is not moving.
                          , CenterFrequencyHz: float): # The center frequency of the transmission in Hz
    '''
    brief: This function computes the maximum doppler frequency for a mobile terminal with the following inputs.
    param: VelocityKmh  I    -> The velocity at which the transmitter and receiver are approaching one another.
    param: CenterFrequencyHz -> The center frequency of the communication link
    notes: The velocity may be positive and negative
    '''
    assert VelocityKmh       >= -500 and VelocityKmh       <= 500,  "The velocity in km/h is unreasonable."
    assert CenterFrequencyHz > 0     and CenterFrequencyHz <  60e9, "The Center frequency is unreasonable."

    LightSpeed   = 300e6                       # Meters per Second
    VelocityMps  = VelocityKmh * 1000 / 3600   # 1000 m/Km  -> 3600 sec/h
    DopplerHz    = CenterFrequencyHz*(VelocityMps/LightSpeed)
    return DopplerHz





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
    assert isinstance(fLinearNoisePower, float), 'The fLinearNoisePower argument must be of type float'
    assert isinstance(iNumberOfSamples,    int), 'The iNumberOfSamples argument must be of type int'
    assert isinstance(iSeed, int) or iSeed == None, 'The iSeed argument must be of type int or None' 
    assert isinstance(bComplexNoise, bool)     , 'The bComplexNoise argument must be of type'
    assert iNumberOfSamples > 0                , 'The number of noise samples to generate must be > 0.'

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
def AddGaussianNoise(SnrDb              # float or int 
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
    assert isinstance(SnrDb, float) or isinstance(SnrDb, int), 'The fLinearNoisePower argument must be of type float'
    assert isinstance(InputSequence, np.ndarray)         , 'The InputSequence argument must be of type np.array'
    assert isinstance(iSeed, int) or iSeed == None       , 'The iSeed argument must be of type int or None' 
    assert isinstance(bCopy, bool)                       , 'The bCopy argument must be of type bool'
    assert np.issubdtype(InputSequence.dtype, np.inexact), 'The input sequence must be complex64/128 or real32/64'
    
    # -----------------------------------------
    # Start processing
    fLinearSnr = 10 ** (SnrDb/10)
    N          = len(InputSequence)

    bInputIsComplex = np.issubdtype(InputSequence.dtype, np.complexfloating)

    # Find the mean square of the sequence
    if bInputIsComplex:
        fMeanSquare = (1/N) * np.sum(InputSequence * np.conj(InputSequence)).real
    else:
        fMeanSquare = (1/N) * np.sum(InputSequence * InputSequence)

    # Determine the needed noise power and generate the Gaussian noise signal
    fNoisePower   = float(fMeanSquare/fLinearSnr)
    NoiseSequence = CreateGaussianNoise(fNoisePower, N, iSeed, bInputIsComplex)

    # Do we want to overwrite the InputSequence object or make a copy of it before adding noise.
    if bCopy:  # Do not change the InputSequence vector
        Output = InputSequence + NoiseSequence 
        return Output
    else:      # Overwrite the InputSequence with its noisy counterpart
        InputSequence += NoiseSequence
        return None


 










# ---------------------------------------------------------------#
# Class Definitions                                              #
# ---------------------------------------------------------------#

# ------------------------------------------------------------------------
# > CMultipathChannel
# ------------------------------------------------------------------------
class CMultipathChannel():
    """
    Notes: This channel model represents a set of multipaths defined by a complex amplitude,
    a delay in seconds, and a doppler frequency in Hz.
    """
    def __init__(self):
        self.CMultipathChannel   = {}
        self.PathIndex           = 0


    # -----------------------------------------------------------------
    # Function: AddPath()
    # -----------------------------------------------------------------
    def AddPath(self
              , Amplitude:                 complex
              , DelayInSec:                float
              , DopplerHz:                 float   
              , RoundToNearestSampleTime:  bool   = False
              , SampleRate:                float  = None):
        """
        brief: Allows us to add paths to the multipath channel
        param: Amplitude  -> The complex valued amplitude of the path
        param: DelayInSec -> The delay in seconds of the path compared to the FFT time in the receiver
        param: DopplerHz  -> The Doppler frequency in Hz due to the motion of the transceivers.
        param: RoundToNearestSampleTime -> Round the delay in seconds to the nearest sample time 
        param: SampleRate -> Compute the nearest sample time via the sample period 1/SampleRate
        """
        assert isinstance(Amplitude, complex), 'Amplitude must be of type complex'
        assert isinstance(DelayInSec, float),  'DelayInSec must be of type float'
        assert isinstance(RoundToNearestSampleTime, bool), 'Error. This should be a bool'
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
                          , ListResourceElements: np.ndarray) -> np.array:
        """
        brief: ComputeFreqResponse computes the frequency response of the channel using the following parameters
        param: ListResourceElements -> This is a list of C Resource Elements
        """
                
        NumEntries   = len(ListResourceElements)
        FreqResponse = np.zeros(NumEntries, dtype = np.complex64)

        for EntryIndex, ResourceElement in enumerate(ListResourceElements):
            FrequencyHz = ResourceElement.FrequencyHz
            TimeSec     = ResourceElement.TimeSec
            for PathIndex in range(0, self.PathIndex):
                [Amplitude, DelayInSec, DopplerHz] = self.CMultipathChannel[PathIndex]  
                FreqResponse[EntryIndex]   += Amplitude * np.exp(-1j*2*np.pi*DelayInSec*FrequencyHz) * \
                                                          np.exp(1j*2*np.pi*TimeSec*DopplerHz)
            
            ResourceElement.IdealFreqResponse = FreqResponse[EntryIndex]

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














# ---------------------------------------------------------------------------------------------------------------------#
# > The CChannelModel class encapsulates the bounds of the channels multipath configurations                             #
# ---------------------------------------------------------------------------------------------------------------------#
class CChannelModel():
    # -------------------------------------------------------------------------------------------------
    # brief   -> The channel model describes the bounds on the channels multipath configuration:
    # Example -> Channel_Model = CChannel_Model( , , , )
    # Inputs  -> The Inputs are explained below and are valid for the __init__, __call__, Initialize functions
    def __init__(self
               , MinDelaySec   : float     # The minimum delay for an LTE path is around -1e6 seconds (earlist arrival time)
               , MaxDelaySec   : float     # The maximum delay for an LTE path is around  4e6 seconds (latest  arrival time)
               , MaxDopplerHz  : float):   # The maximum doppler frequency for an LTE path.
                                           # The Doppler range will be [-MaxDopplerHz, +MaxDopplerHz]
        self.Initialize(MinDelaySec, MaxDelaySec, MaxDopplerHz)   
        self.EigenVectors = np.array([0,0], dtype=np.complex64)   # The eigen vectors -> The principle components
        self.EigenValues  = np.array([0,0], dtype=np.complex64)   # The eigen values  -> The degree to which the data extends
                                                                  #                      along the principle components
        self.NumberOfUsefulPrincipleComponents = 0
                                              # The number of principle components we will use for the projection process
        self.ListResourceElements = []


    # -------------------------------------------------------------------------------------------------
    # brief   -> The class is callable and reinitializes the object with new parameters:
    # Example -> Channel_Model( , , , )
    # Inputs  -> See inputs of constructor __init__
    def __call__(self
               , MinDelaySec   : float
               , MaxDelaySec   : float
               , MaxDopplerHz  : float):
        self.Initialize(MinDelaySec, MaxDelaySec, MaxDopplerHz) 



    # -------------------------------------------------------------------------------------------------
    # brief   -> Overload the str function
    def __str__(self):
        ReturnString = "CChannelModel >>MinPathDelaySec: "   + str(self.MinDelaySec) + \
                       "     >>MaxPathDelaySec: "            + str(self.MaxDelaySec) + \
                       "     >>MaxDopplerHz: "               + str(self.MaxDopplerHz)
        return ReturnString



    # -------------------------------------------------------------------------------------------------
    # brief   -> Initializing the class.
    # Example -> Initialize( , , , )
    # Inputs  -> See inputs of constructor __init__
    def Initialize(self
                 , MinDelaySec   : float
                 , MaxDelaySec   : float
                 , MaxDopplerHz  : float):

        # Error checking
        assert MinDelaySec   > -10e6                                , "The minimum delay is unreasonable"
        assert MaxDelaySec   > MinDelaySec and MaxDelaySec  <= 20e-6, "The maximum delay is unreasonable."
        assert MaxDopplerHz  > 0           and MaxDopplerHz <  5000,  "The maximum doppler is unreasonable."

        # Initializing
        self.MinDelaySec   = MinDelaySec
        self.MaxDelaySec   = MaxDelaySec
        self.MaxDopplerHz  = MaxDopplerHz



    # -------------------------------------------------------------------------------------------------
    # brief   -> The following function computes the principle components of the space occupied by the
    #            channel model. This technique will produce a set of eigen vectors and eigen values that
    #            characterize this space. 
    # Input   -> A list of CResourceElements indicating the position at which we have available CRS
    # Output  -> An array of eigenvectors, which represent the orthogonal unit vectors describing our space.
    #            An array of eigenvalues that indicate to which extent the data extends along each eigen vector.
    def PCA_Computation(self
                      , ListResourceElements: list):
        '''
        This function computes the principle component analysis for the N dimensional vector space.
        '''
        assert isinstance(ListResourceElements, list), 'ListResourceElements must be of type list.'
        assert isinstance(ListResourceElements[0], CResourceElement), 'Type error.'

        # Save this list, so the next list can be compared to this one and so we can know whether we need to
        # recompute the PCA for the new list or we may simply use the known EigenValues and EigenVectors
        self.ListResourceElements = ListResourceElements

        # Determine the dimensionality of the space
        NumDimensions = len(ListResourceElements)

        # Determine how many observation in the space we will generate
        DelayStepInSec  = 50e-9 
        DopplerStepInHz = 20 

        DelayArray       = np.arange(self.MinDelaySec, self.MaxDelaySec+DelayStepInSec, DelayStepInSec)
        DopplerArray     = np.arange(-self.MaxDopplerHz, self.MaxDopplerHz, DopplerStepInHz)

        # Determine the number of channel configurations
        NumChannelConfigurations = len(DelayArray)*len(DopplerArray)

        # Make space for the Frequency Response matrix
        FreqResponseMatrix = np.zeros([NumDimensions, NumChannelConfigurations], dtype=np.complex64)

        ChannelConditionsCount = 0
        for Delay in DelayArray:
            for Doppler in DopplerArray:
                RandomPhase1 = random.uniform(-np.pi, np.pi)   # uniformly distributed phase from -pi to pi
                RandomPhase2 = random.uniform(-np.pi, np.pi)   # uniformly distributed phase from -pi to pi
                for DimensionCount, Re in enumerate(ListResourceElements):
                    Frequency = Re.FrequencyHz
                    Time      = Re.TimeSec
                    FreqResponse = np.exp(-1j*(2*np.pi*Delay*Frequency + RandomPhase1)) * \
                                   np.exp( 1j*(2*np.pi*Time*Doppler    + RandomPhase2))
                    FreqResponseMatrix[DimensionCount, ChannelConditionsCount] = FreqResponse
                    DimensionCount += 1
                ChannelConditionsCount += 1

        # Compute the covariance matrix
        CovarianceMatrix = np.cov(FreqResponseMatrix, None, True)

        # Compute the Eigenvalue decomposition
        self.EigenValues, self.EigenVectors = np.linalg.eigh(CovarianceMatrix.transpose())

        # Determine the number of principle components which we will use for the projection. We will use those
        # Principle components whose eigenvalues are larger than PowerSeperator tims the largest eigenvalue.
        PowerSeperator = 0.02   
        LargestEigenvalue = max(self.EigenValues)
        self.NumberOfUsefulPrincipleComponents = 0
        for EigenValue in self.EigenValues:
            CurrentEigenValueRatio = EigenValue/LargestEigenvalue
            if CurrentEigenValueRatio > PowerSeperator:
                self.NumberOfUsefulPrincipleComponents += 1

        # Return the Eigenvalues and Eigenvectors
        return self.EigenValues, self.EigenVectors



    # -------------------------------------------------------------------------------------------------
    # brief   -> The following function verifies that the new list of RE may be used with the currently
    #            computed eigenvalues and eigenvectors
    # Input   -> A list of resource elements for which we want to compute the frequency response and CINR
    def CheckDmrsReConfiguration(self
                               , ListNewResourceElements) -> bool:
        """
        brief: This function determines whether the previously computed Eigenvalues and Eigenvectors
        are still valid for this new list of DMRS resource elements.
        param: A list of resource elements for which we want to use the current Eigenvalues and Eigenvectors.
        """
        assert isinstance(ListNewResourceElements, list), 'The ListNewResourceElements agrument must be a list'
        assert isinstance(ListNewResourceElements[0], CResourceElement), 'The ListNewResourceElements entris must be CResourceLements.'
        
        ReturnValue = True

        # Initial Error checking
        if len(ListNewResourceElements) != len(self.ListResourceElements): return False
        
        # Check compatibility
        # -> Ensure that the change in FreqUnits and Time units is identical as we move through each list
        # -> Ensure that the subcarrier spacings and cyclic prefixes are the same
        for Index in range(1, len(ListNewResourceElements)):
            OldRe0   = self.ListResourceElements[Index]
            NewRe0   = ListNewResourceElements[Index]
            OldRe1M  = self.ListResourceElements[Index-1]
            NewRe1M  = ListNewResourceElements[Index-1]

            # Compute frequency and time delta betwee successive CResoureElement entries
            OldSubcarrierDelta     = OldRe0.FreqUnit - OldRe1M.FreqUnit
            NewSubcarrierDelta     = NewRe0.FreqUnit - NewRe1M.FreqUnit 
            OldTimeDelta           = OldRe0.TimeUnit - OldRe1M.TimeUnit
            NewTimeDelta           = NewRe0.TimeUnit - NewRe1M.TimeUnit 

            # Make sure these deltas are the same as well as the subcarrier spacing and cyclic prefix
            bProperSubcarrierDelta = OldSubcarrierDelta == NewSubcarrierDelta
            bProperTimeDelta       = OldTimeDelta       == NewTimeDelta
            bProperConfiguration   = OldRe0.Sc == NewRe0.Sc and OldRe0.Cp == NewRe0.Cp

            if bProperConfiguration == False or bProperSubcarrierDelta == False or bProperTimeDelta == False:
                ReturnValue = False
                break

        return ReturnValue





    # -------------------------------------------------------------------------------------------------
    # brief   -> The following function computes the channel estimation 
    # Input   -> A list of resource elements
    def FrequencyResponseEstimator(self, ListResourceElements):
        '''
        This function will compute the frequency response estimate by removing noise from the raw frequency response
        '''
        # Error checking
        assert len(ListResourceElements) > 0,              "Ensure the list of Received DMRS values is not empty."
        assert len(self.EigenValues) > 0,                  "The principle components have not yet been computed. Run PCA_Computation()"
        assert self.NumberOfUsefulPrincipleComponents > 0, "NumberOfUsefulPrincipleComponents = 0."

        NumDmrs = len(ListResourceElements)
        
        # Check if the input DMRS list is compatible with the list used to compute the PCA earlier
        bValid = self.CheckDmrsReConfiguration(ListResourceElements)
        assert bValid, "The input Crs list is not compatible with the principle components computed earlier."

        # Compute the observed frequency response and deposit it in an array
        RawFrequencyResponse = np.zeros(NumDmrs, dtype=np.complex64)
        for Index  in range(0, NumDmrs):
            RawFrequencyResponse[Index] = ListResourceElements[Index].RxValue / ListResourceElements[Index].TxValue 

        # Begin the Projection Process for the signal 
        ProjectedSignal = np.zeros(NumDmrs, dtype=np.complex64)
        for Index in range(NumDmrs - self.NumberOfUsefulPrincipleComponents, NumDmrs):
            CurrentEigenVector = np.flip(self.EigenVectors[:, Index], 0)
            Dot1               = np.matmul(RawFrequencyResponse , np.conj(CurrentEigenVector).transpose()) 
            Dot2               = np.matmul(CurrentEigenVector,    np.conj(CurrentEigenVector).transpose())
            Projection         = (Dot1/Dot2) * CurrentEigenVector
            ProjectedSignal    += Projection

        # Begin the Projection Process for the Noise 
        ProjectedNoise  = np.zeros(NumDmrs, dtype=np.complex64)
        for Index in range(0, NumDmrs - self.NumberOfUsefulPrincipleComponents): 
            CurrentEigenVector = np.flip(self.EigenVectors[:, Index], 0)
            Dot1               = np.matmul(RawFrequencyResponse , np.conj(CurrentEigenVector).transpose()) 
            Dot2               = np.matmul(CurrentEigenVector,    np.conj(CurrentEigenVector).transpose())
            Projection         = (Dot1/Dot2) * CurrentEigenVector
            ProjectedNoise     += Projection

        # Compute the total noise power that must have extended along all principle components
        MeanSquareNoise_InNoiseArea  = np.mean(ProjectedNoise  * np.conj(ProjectedNoise)).real
        Ratio      = self.NumberOfUsefulPrincipleComponents / (NumDmrs - self.NumberOfUsefulPrincipleComponents)
        MeanSquareNoise_InSignalArea = MeanSquareNoise_InNoiseArea * Ratio
        MeanSquareNoise_Total        = MeanSquareNoise_InNoiseArea + MeanSquareNoise_InSignalArea

        # Compute the total signal power that is confined along only the useful principle components
        MeanSquareSignal_InSignalArea = np.mean(ProjectedSignal * np.conj(ProjectedSignal)).real
        MeanSquareSignal_Total        = MeanSquareSignal_InSignalArea - MeanSquareNoise_InSignalArea

        # Compute the RS-Cinr value in dB
        SNR_Linear                    = MeanSquareSignal_Total.real/MeanSquareNoise_Total.real
        if SNR_Linear < 0: SNR_Linear = 0.00001
        DMRSCinrdB                    = 10*np.log10(SNR_Linear)
        
        # Build the new Frequency response estimate back into the resource elements 
        for Index in range(0, NumDmrs):
            ListResourceElements[Index].EstFreqResponse = ProjectedSignal[Index].copy()
             
        return DMRSCinrdB    

 


# -------------------------------------------------------------------------------
# Test Bench
# -------------------------------------------------------------------------------
if __name__ == "__main__":

    Test = 1

    # ------------------------------------
    # Testing the ComputeDopplerFrequency() function
    # ------------------------------------
    if Test == 0:
        VelocityKmh       = 100
        CenterFrequencyHz = 5e9
        print('The Doppler in Hz = ' + str(ComputeDopplerFrequency(VelocityKmh, CenterFrequencyHz)))

    # ------------------------------------
    # Testing the CChannelModel() class
    # ------------------------------------
    if Test == 1:
        # --------------------------------------------------
        # 1. Create and initialize a CMultipath Model
        # --------------------------------------------------
        SampleRate       = 20.48e6
        MultipathChannel = CMultipathChannel()
        Amplitudes       = [      -.3j,     1+0j,  0.2 - .2j,    -0.2 + .1j]
        DelaysInSec      = [   -50e-9,   000e-9,    600e-9,       1000e-9]
        DopplersInHz     = [       200,      100,        400,          -500]

        assert len(Amplitudes) == len(DelaysInSec) and len(Amplitudes) == len(DopplersInHz)
        for Index in range(0, len(Amplitudes)):
            MultipathChannel.AddPath(Amplitudes[Index]
                                   , DelaysInSec[Index]
                                   , DopplersInHz[Index]   
                                   , False   # Round to nearest sample
                                   , SampleRate) 
        
        
        
        # -----------------------------------------------------
        # 2. Create and initialized the channel model
        # -----------------------------------------------------
        MinDelaySec  = -0.2e-6
        MaxDelaySec  =  1.0e-6
        MaxDopplerHz =  700

        ChannelModel = CChannelModel(MinDelaySec, MaxDelaySec, MaxDopplerHz)

        # -------------------------------------
        # Create the list of resource elements at which DMRS are located
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
        # ---------------------------
        # Compute the PCA Analysis
        [EigenValues, EigenVectors] = ChannelModel.PCA_Computation(DmrsList)

        plt.figure(1)
        plt.stem(np.arange(0, len(EigenValues)), np.abs(EigenValues))
        plt.title('Eigenvalues')
        plt.grid(True)
        plt.show()
        Stop = 1




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
        CinrdB           = 10
        CinrLinear       = 10**(CinrdB/10)
        LinearNoisePower = MeanSquare / CinrLinear
        Noise = CreateGaussianNoise(LinearNoisePower 
                                  , len(DmrsList) 
                                  , -1
                                  , True)    

        for Index, RE in enumerate(DmrsList):
            RE.RxValue += Noise[Index]
            RE.RawFreqResponse = RE.RxValue / RE.TxValue


        # -------------------------------------------------------
        # 5. Estimate the freuquency response by reducing noise from the raw frequency response
        # -------------------------------------------------------
        CinrEstimatedB = ChannelModel.FrequencyResponseEstimator(DmrsList)
        print('The Estimated Cinr (dB) = ' + str(CinrEstimatedB))

        # -------------------------
        # Compute the CINR after noise reduction
        MeanSquareIdealFResponse = 0
        MeanSquareError          = 0
        for Index, RE in enumerate(DmrsList):
            MeanSquareIdealFResponse += (1/len(DmrsList)) * (RE.IdealFreqResponse * RE.IdealFreqResponse.conj()).real
            Error = RE.IdealFreqResponse - RE.EstFreqResponse
            MeanSquareError += (1/len(DmrsList)) * (Error * Error.conj()).real

        print('The CINR of the frequency estimate = ' + str(10*math.log10(MeanSquareIdealFResponse/MeanSquareError)))            

        IdealFreqResponse = [Dmrs.IdealFreqResponse for Dmrs in DmrsList]
        RawFreqResponse   = [Dmrs.RawFreqResponse   for Dmrs in DmrsList]
        EstFreqResponse   = [Dmrs.EstFreqResponse   for Dmrs in DmrsList]
        FrequencyList     = [Dmrs.FrequencyHz       for Dmrs in DmrsList]

        plt.figure(1)
        plt.subplot(2,1,1)
        plt.plot(FrequencyList, np.array(IdealFreqResponse).real, 'r')
        plt.plot(FrequencyList, np.array(RawFreqResponse).real, 'r.')
        plt.plot(FrequencyList, np.array(EstFreqResponse).real, 'r:')
        plt.title('Frequency Response')
        plt.legend(['Ideal', 'Raw', 'Estimated'])
        plt.xlabel('Hz')
        plt.grid(True)
        plt.tight_layout()
        plt.subplot(2,1,2)
        plt.plot(FrequencyList, np.array(IdealFreqResponse).imag, 'b')
        plt.plot(FrequencyList, np.array(RawFreqResponse).imag, 'b.')
        plt.plot(FrequencyList, np.array(EstFreqResponse).imag, 'b:')
        plt.title('Frequency Response')
        plt.legend(['Ideal', 'Raw', 'Estimated'])
        plt.xlabel('Hz')       
        plt.grid(True)
        plt.tight_layout()
        plt.show()



    # ------------------------------------------------------------------
    # The following test is a new type of channel estimator
    # ------------------------------------------------------------------
    if Test == 2:
        # --------------------------------------------------
        # 1. Create and initialize a CMultipath Model
        # --------------------------------------------------
        SampleRate       = 20.48e6
        MultipathChannel = CMultipathChannel()
        Amplitudes       = [      -.3j,     1+0j,  0.2 - .2j,    -0.2 + .1j]
        DelaysInSec      = [   -250e-9,   000e-9,      600e-9,       1600e-9]
        DopplersInHz     = [       200,      100,        400,          -500]

        assert len(Amplitudes) == len(DelaysInSec) and len(Amplitudes) == len(DopplersInHz)
        for Index in range(0, len(Amplitudes)):
            MultipathChannel.AddPath(Amplitudes[Index]
                                   , DelaysInSec[Index]
                                   , DopplersInHz[Index]   
                                   , False   # Round to nearest sample
                                   , SampleRate) 

        # -------------------------------------
        # Create the list of resource elements at which DMRS are located
        ReType        = EReType.DmrsPort0
        Sc            = ESubcarrierSpacing.Sc20KHz
        Cp            = ECyclicPrefix.Ec4MicroSec
        FreqUnitArray = list(range(-450, 450, 3))
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

        IdealFreqResponse = [Dmrs.IdealFreqResponse for Dmrs in DmrsList]
        FrequencyList     = [Dmrs.FrequencyHz       for Dmrs in DmrsList]

        NumDmrs            = 512
        
        # ----------------------------------------------
        # Add noise to the frequency response
        CinrdB = 0
        NoisyFreqResponse  = AddGaussianNoise(CinrdB, np.array(IdealFreqResponse), -1, True)
        IdealFreqResponse1 = np.hstack([np.array(IdealFreqResponse, np.complex64), np.zeros(212, np.complex64)])
        NoisyFreqResponse1 = np.hstack([np.array(NoisyFreqResponse, np.complex64), np.zeros(212, np.complex64)])
        FrequencyList1     = np.arange(FrequencyList[0], FrequencyList[len(FrequencyList)-1] + 30e3, 60e3 * 300/512)

        # ----------------------------------------------
        # Apply the Hamming window to the NoisyFreqResponse
        SkirtLength = 64
        N           = 2*SkirtLength
        n             = np.arange(0, N)
        HanningWindow = 0.5 - 0.5 * np.cos(2*np.pi*(n+1)/(N+1))
        SkirtUP       = HanningWindow[:int(N/2)]
        SkirtDown     = HanningWindow[int(N/2):]
        OverlaidFreqResponse = NoisyFreqResponse1[0:512].copy()
        OverlaidFreqResponse[:SkirtLength] *= SkirtUP
        OverlaidFreqResponse[300-SkirtLength:300]  *= SkirtDown

        OverlaidCleanResponse = IdealFreqResponse1[0:512].copy()
        OverlaidCleanResponse[:SkirtLength] *= SkirtUP
        OverlaidCleanResponse[300-SkirtLength:300]  *= SkirtDown

        plt.figure(1)
        plt.subplot(3,1,1)
        plt.plot(FrequencyList1, np.array(IdealFreqResponse1).real, 'r')
        plt.plot(FrequencyList1, np.array(NoisyFreqResponse1).real, 'r:')
        plt.title('Ideal and Noisy Real Frequency Response')
        plt.xlabel('Hz')
        plt.grid(True)
        plt.subplot(3,1,2)
        plt.plot(FrequencyList1, np.array(IdealFreqResponse1).imag, 'b')
        plt.plot(FrequencyList1, np.array(NoisyFreqResponse1).imag, 'b:')
        plt.title('Ideal and Noisy Imag Frequency Response')
        plt.xlabel('Hz')
        plt.grid(True)
        plt.subplot(3,1,3)
        plt.plot(FrequencyList1, np.array(OverlaidFreqResponse).real, 'r')
        plt.plot(FrequencyList1, np.array(OverlaidFreqResponse).imag, 'b')
        plt.title('Overlaid realImag Frequency Response')
        plt.xlabel('Hz')
        plt.grid(True)
  

        iFftOut        = np.fft.ifft(OverlaidFreqResponse)
        iFftOutTemp    = np.zeros(512, dtype = iFftOut.dtype)
        iFftRearranged = np.hstack([iFftOut[256:], iFftOut[:256]]) 

        plt.figure(2)
        plt.stem(np.arange(0, 512), np.abs(iFftRearranged))
        plt.grid(True)
    

        MaxAbs = np.max(np.abs(iFftRearranged))
        for Index in range(2, len(iFftRearranged) -2):
            if np.abs(iFftRearranged[Index]) > MaxAbs * 0.2:
                #iFftOutTemp[Index-1] = iFftRearranged[Index - 2]
                iFftOutTemp[Index-1] = iFftRearranged[Index - 1]
                iFftOutTemp[Index]   = iFftRearranged[Index]
                iFftOutTemp[Index+1] = iFftRearranged[Index + 1]
                #iFftOutTemp[Index+1] = iFftRearranged[Index + 2]

        iFftOrg  = np.hstack([iFftOutTemp[256:], iFftOutTemp[:256]]) 

        FftOut   = np.fft.fft(iFftOrg)

        plt.figure(3)
        plt.subplot(2,1,1)
        plt.plot(FrequencyList1, np.array(FftOut).real, 'r')
        plt.plot(FrequencyList1 , np.array(OverlaidCleanResponse).real, 'r:') 
        plt.title('Recreated Frequency Response')
        plt.xlabel('Hz')
        plt.grid(True)
        plt.subplot(2,1,2)
        plt.plot(FrequencyList1, np.array(FftOut).imag, 'b')
        plt.plot(FrequencyList1, np.array(OverlaidCleanResponse).imag, 'b:')   
        plt.title('Recreated Frequency Response')
        plt.xlabel('Hz')
        plt.grid(True)
        plt.show()