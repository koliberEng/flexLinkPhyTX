# File:   Channel_Manager.py
# Author: Andreas Schwarzinger                                                         Date: July, 29, 2020
# Notes:  The following class provides channel simulation services. Specifically, it provides the following:
# -> It takes a dictionary detailing the channel model (Max/Min 
# -> It takes a dictionary detailing the channel conditions

import random                               # Used during AWGN creation
import copy                                 # used for deepcopying a list of CResourceElements
import numpy as np
from   LteDefinitions import*
import matplotlib.pyplot as plt
from   mpl_toolkits.mplot3d import axes3d 


# --------------------------------------------------------------------------------------------------
# Declaration of the eFrPlotConfig enum, which determines the manner in which the frequency response is plotted
class eFrPlotConfig(enum.Enum):
    Polar_KL  =  0     # Magnitude and phase response using subcarrier, k, and Ofdm symbol index, l
    Polar_FT  =  1     # Mangitude and phase respones using frequency and time
    Rect_KL   =  2     # I and Q respones using subcarrier, k, and Ofdm symbol index, l
    Rect_FT   =  3     # I and Q response using frequency and time

    # This function checks to see that provided value is valid
    def CheckValidEnumeration(EnumInput):
        IsValid = 0
        for Member in eFRPlotConfig:
            if EnumInput == Member:
                IsValid = 1
        assert IsValid == 1, "The input argument is not a valid member of eFRPlotConfig."



# -------------------------------------------------------------------------------------------------
# brief   -> This function computes the maximum doppler frequency for a mobile terminal with the following inputs.
# Output  -> The Doppler frequency in Hz
# Inputs  -> The Inputs are explained below.
def ComputeDopplerFrequency(VelocityKmh:       float   # Velocity of mobile terminal is in km/h. The eNodeB is not moving.
                          , CenterFrequencyHz: float): # The center frequency of the transmission in Hz

    assert VelocityKmh       > 0 and VelocityKmh       < 500,  "The velocity in km/h is unreasonable."
    assert CenterFrequencyHz > 0 and CenterFrequencyHz < 40e9, "The Center frequency is unreasonable."

    LightSpeed   = 300e6                       # Meters per Second
    VelocityMps  = VelocityKmh * 1000 / 3600   # 1000 m/Km  -> 3600 sec/h
    DopplerHz    = CenterFrequencyHz*(VelocityMps/LightSpeed)
    return DopplerHz


# -------------------------------------------------------------------------------------------------
# brief   -> The following function generates and returns a resource grid or ReList of complex noise values.
#            Specifically, it does the following:
#            1. It computes the mean square of the input resource grid or ReList (based on bAllREs)
#            2. It computes the noise power given the supplied SNRdB
#            3. It fills an empty resource grid or ReList with complex noise based on the computed noise power.
# Inputs  -> The Inputs are explained below.
# Output  -> A noise resource grid or a noise list of CResourceElements.
def GenAWGN(SNRdB:          float               # Represents the SNR in dB
          , bAllREs:        bool                # If true, the function calculates the signal power for all REs
                                                # If false, the function will calculate the signal power for all non-zero REs
          , bAddOrGen:      bool                # If  true, the function returns a ResourceGrid or ReList of noise + original RE values
                                                # If false, the function returns a ResourceGrid or ReList only
          , ResourceGrid:   np.ndarray = None   # Either a resource grid is present
          , ReList:         list = None         # Or a list of CResourceElements
          , Seed:           int  = None):

    # Set the seed of the random generator
    random.seed(Seed)   # Will use system time if the seed was not supplied

    # Compute linear signal power
    SNR_Linear = 10**(SNRdB/10)

    # Processing the ResourceGrid
    if(ReList == None):
        NumSubcarriers, NumSymbols = ResourceGrid.shape
        
        # Define the Output argument
        OutputGrid = np.zeros([NumSubcarriers, NumSymbols], dtype=np.complex64)

        # Compute the mean square of the ResourceGrid
        MeanSquare = float(0)
        NumREs     = float(0)
        for k in range(0, NumSubcarriers):
            for l in range(0, NumSymbols):
                Re     = ResourceGrid[k,l];
                Square = (Re*np.conj(Re)).real
                if (bAllREs == False and np.abs(Re) == 0): 
                    continue
                MeanSquare += Square
                NumREs     += 1
        MeanSquare = MeanSquare / NumREs

        # Compute desired noise power. 
        # Generate noise for each RE and sum the RE value and the noise in the OutputGrid
        NoisePower = MeanSquare/SNR_Linear
        NoiseStd   = np.sqrt(NoisePower)
        for k in range(0, NumSubcarriers):
            for l in range(0, NumSymbols):
                # The output RE is the noise
                OutputGrid[k,l]    = random.gauss(0, NoiseStd * 0.7071) + 1j*random.gauss(0, NoiseStd * 0.7071)
                if bAddOrGen:
                    # The output is the noise added to the complex value of the resource element
                    OutputGrid[k,l] += ResourceGrid[k,l]
                 
        return OutputGrid

    else:
        ListLength = len(ReList)
        assert ListLength > 0, "The list of CResourceElements is empty."

        # Define the output argument - Copy by value (we don't want to change in the input arguments)
        OutputList = []
        for Re in ReList:
            OutputList.append(Re.copy())    # Causes the object in the list to be physically copied rather        
                                            # than simply copying the pointers (I had to create a copy constructor)
        # Compute the mean square of the RE values in the ReList
        MeanSquare = float(0)
        NumREs     = float(0)
        for Re in OutputList:
            Square = Re.Value * np.conj(Re.Value)
            if (bAllREs == False and np.abs(Re) == 0): 
                    continue
            MeanSquare += Square
            NumREs     += 1
        MeanSquare = MeanSquare / NumREs

        # Compute desired noise power. 
        # Generate noise for each RE. 
        NoisePower = (MeanSquare.real)/SNR_Linear
        NoiseStd   = np.sqrt(NoisePower)
        for Re in OutputList:
            Noise    = random.gauss(0, NoiseStd * 0.7071) + 1j*random.gauss(0, NoiseStd * 0.7071)
            if bAddOrGen:
                # The noise is added to the output CResourceElement.Value
                Re.Value += Noise
            else:
                # The noise is the output CResourceElement.Value
                Re.Value = Noise
        
        return OutputList
    





















# ---------------------------------------------------------------------------------------------------------------------#
# The CChannelModel class encapsulates the bounds of the channels multipath configurations                             #
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
        self.RefCrsList   = []                # This is the CrsList with which we computed the PCA
                                              # When we call FrequencyResponseEstimator(), the input CrsList will be 
                                              # compared to the list we used for the PCA. The new list doesn't need to
                                              # be identical to the RefCrsList, but its Subcarrier and OfdmSymbol 
                                              # indices must be the same otherwise the principle components are not valid. 
        self.NumberOfUsefulPrincipleComponents = 0
                                              # The number of principle components we will use for the projection process

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
        assert MinDelaySec   > -10e6       and MinDelaySec  <   2e-6, "The minimum delay is inappropriate for LTE"
        assert MaxDelaySec   > MinDelaySec and MaxDelaySec  <= 10e-6, "The maximum delay is inappropriate for LTE or <= the mimum delay."
        assert MaxDopplerHz  > 0           and MaxDopplerHz <  1000,  "The maximum doppler is out of range for LTE."

        # Initializing
        self.MinDelaySec   = MinDelaySec
        self.MaxDelaySec   = MaxDelaySec
        self.MaxDopplerHz  = MaxDopplerHz



    # -------------------------------------------------------------------------------------------------
    # brief   -> The following function checks to see whether a potential input CrsList may be used
    #            with the principle components that have been previously computed in this object.
    #            If the principle components have not yet been computed, or the input CrsList is too
    #            different from the one that was used to compute the principle components, we return False
    def CheckCrsList(self
                   , CrsList: list
                    ) -> bool:

        ReturnValue = True

        # Initial Error checking
        if len(CrsList) != len(self.RefCrsList): return False

        # Check compatibility
        # -> Ensure that the subcarrier indices are the same
        # -> Compute the difference between the OfdmSymbol index of the current RE and the first RE in the list. This 
        #    difference must match as we work our way through both lists. 
        #    Only if these two checks are valid can we reuse the previously computed principle components
        for Index in range(0, len(CrsList)):
            bSubcarrierIndexCheck = CrsList[Index].SubcarrierIndex == self.RefCrsList[Index].SubcarrierIndex
            OfdmDistanceRef       = self.RefCrsList[Index].OfdmSymbolIndex - self.RefCrsList[0].OfdmSymbolIndex
            OfdmDistanceInput     = CrsList[Index].OfdmSymbolIndex - CrsList[0].OfdmSymbolIndex
            bDistancesCheck       = OfdmDistanceRef == OfdmDistanceInput
            if bSubcarrierIndexCheck == False or bDistancesCheck == False:
                ReturnValue = False
                break

        return ReturnValue




    # -------------------------------------------------------------------------------------------------
    # brief   -> The following function computes the principle components of the space occupied by the
    #            channel model. This technique will produce a set of eigen vectors and eigen values that
    #            characterize this space. 
    # Input   -> A list of CResourceElements indicating the position at which we have available CRS
    # Output  -> An array of eigenvectors, which represent the orthogonal unit vectors describing our space.
    #            An array of eigenvalues that indicate to which extent the data extends along each eigen vector.
    def PCA_Computation(self
                      , LteConst: CLteConst
                      , ReList:   list
                      ) -> [np.ndarray, np.ndarray]:

        # Determine the dimensionality of the space
        NumDimensions = len(ReList)

        # Determine how many observation in the space we will generate
        DelayStepInSec  = 100e-9 
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
                DimensionCount = 0
                for Re in ReList:
                    Frequency = Re.GetFrequency(LteConst)
                    SampleIndex, Time = Re.GetStartTime(LteConst)
                    FreqResponse = np.exp(-1j*(2*np.pi*Delay*Frequency + RandomPhase1)) * \
                                   np.exp( 1j*(2*np.pi*Time*Doppler    + RandomPhase2))
                    FreqResponseMatrix[DimensionCount, ChannelConditionsCount] = FreqResponse
                    DimensionCount += 1
                ChannelConditionsCount += 1

        # Compute the covariance matrix
        CovarianceMatrix = np.cov(FreqResponseMatrix, None, True)

        # Compute the Eigenvalue decomposition
        self.EigenValues, self.EigenVectors = np.linalg.eigh(CovarianceMatrix.transpose())

        # Make a copy of the ReList. When we call FrequencyResponseEstimator later, we will compare
        # its input ReList the ReList that we used for this PCA computation
        self.RefCrsList = copy.deepcopy(ReList)

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
    # brief   -> The following function computes the channel estimation 
    # Input   -> A list of distorted and noisy CRS values that have been demodulated. 
    #            A list of perfect             CRS values that have been computed by the CrsModule.
    #            These two list allow us to compute the noisy frequency response
    # Output  -> A list of CResourceElements featuring the same CRS with less noise.
    #            The RS-CINR estimate
    def FrequencyResponseEstimator(self
                                 , LteConst:       CLteConst
                                 , InputCrsList:   list
                                 , CorrectCrsList: list
                                 ) -> list:
        # Error checking
        assert len(InputCrsList) > 0,                     "The list of noisy CRS values is empty."
        assert len(InputCrsList) == len(CorrectCrsList),  "The two Crs lists have different length."
        assert len(self.EigenValues) > 0, "The principle components have not yet been computed. Run PCA_Computation()"
        assert self.NumberOfUsefulPrincipleComponents > 0, "NumberOfUsefulPrincipleComponents = 0."
        
        # Check if the input Crs list is compatible with the list used to compute the PCA earlier
        bValid = self.CheckCrsList(InputCrsList)
        assert bValid, "The input Crs list is not compatible with the principle components computed earlier."

        # Check if the correct Crs list is compatible with the list used for compute the PCA earlier.
        bValid = self.CheckCrsList(CorrectCrsList)
        assert bValid, "The correct Crs list is not compatible with the principle component computed earlier."

        
        # Compute the observed frequency response and deposit it in an array
        RawFrequencyResponse = np.zeros([1, len(InputCrsList)], dtype=np.complex64)
        for Index  in range(0, len(InputCrsList)):
            RawFrequencyResponse[0, Index] = InputCrsList[Index].Value * np.conj(CorrectCrsList[Index].Value)

        # Begin the Projection Process for the signal 
        ProjectedSignal = np.zeros([1, len(InputCrsList)], dtype=np.complex64)
        for Index in range(len(InputCrsList) - self.NumberOfUsefulPrincipleComponents, len(InputCrsList)):
            CurrentEigenVector = np.flip(self.EigenVectors[:, Index], 0)
            Dot1               = np.matmul(RawFrequencyResponse , np.conj(CurrentEigenVector).transpose()) 
            Dot2               = np.matmul(CurrentEigenVector, np.conj(CurrentEigenVector).transpose())
            Projection         = (Dot1/Dot2) * CurrentEigenVector
            ProjectedSignal    += Projection

        # Begin the Projection Process for the Noise 
        ProjectedNoise  = np.zeros([1, len(InputCrsList)], dtype=np.complex64)
        for Index in range(0, len(InputCrsList) - self.NumberOfUsefulPrincipleComponents): 
            CurrentEigenVector = np.flip(self.EigenVectors[:, Index], 0)
            Dot1               = np.matmul(RawFrequencyResponse , np.conj(CurrentEigenVector).transpose()) 
            Dot2               = np.matmul(CurrentEigenVector, np.conj(CurrentEigenVector).transpose())
            Projection         = (Dot1/Dot2) * CurrentEigenVector
            ProjectedNoise     += Projection

        # Compute the total noise power that must have extended along all principle components
        MeanSquareNoise_InNoiseArea  = np.mean(ProjectedNoise  * np.conj(ProjectedNoise))
        Ratio      = self.NumberOfUsefulPrincipleComponents / (len(InputCrsList) - self.NumberOfUsefulPrincipleComponents)
        MeanSquareNoise_InSignalArea = MeanSquareNoise_InNoiseArea * Ratio
        MeanSquareNoise_Total        = MeanSquareNoise_InNoiseArea + MeanSquareNoise_InSignalArea

        # Compute the total signal power that is confined along only the useful principle components
        MeanSquareSignal_InSignalArea = np.mean(ProjectedSignal * np.conj(ProjectedSignal))
        MeanSquareSignal_Total        = MeanSquareSignal_InSignalArea - MeanSquareNoise_InSignalArea

        # Compute the RS-Cinr value in dB
        SNR_Linear                    = MeanSquareSignal_Total.real/MeanSquareNoise_Total.real
        if SNR_Linear < 0: SNR_Linear = 0.00001
        CrsCinrdB                     = 10*np.log10(SNR_Linear)
        
        # Build the new CrsList with the estimated frequency response 
        FreqResponseList = []
        for Index in range(0, len(InputCrsList)):
            RE_FreqResponse = InputCrsList[Index].copy()
            RE_FreqResponse.Value = ProjectedSignal[0, Index]
            FreqResponseList.append(RE_FreqResponse)

        return FreqResponseList, CrsCinrdB






                                    

# --------------------------------------------------------------------------------------------------------------------#
# The CMultipathConfiguration class details the multipath configurtion for a link between 2 antennas                  #
# --------------------------------------------------------------------------------------------------------------------#
class CMultipathConfiguration():
    """description of class"""
    # brief  -> The constructor for a CMultipathConfiguration object.
    # Inputs -> This class takes a dictionary with the following key:
    #           'Scalars' -> np.array([ ... ], dtype=np.complex64)  - holds complex scalar for each path
    #           'Delays'  -> np.array([ ... ], dtype=np.float32)    - holds the path delays in seconds
    #           'Doppler' -> np.array([ ... ], dtype-np.float32)    - holds the doppler for each path 
    def __init__(self
               , MultipathConfiguration: dict):
        self.Initialize(MultipathConfiguration)


    # ---------------------------------------------------
    # Functor definition
    def __call__(self
               , MultipathConfiguration: dict):
        self.Initialize(MultipathConfiguration)


    # ---------------------------------------------------
    # Overloading the str() function
    def __str__(self):
        ReturnString = "CMultipathConfiguration  >>Scalars  " + str(self.MultipathConfiguration['Scalars']) + "\n" \
                       "                         >>Delays:  " + str(self.MultipathConfiguration['Delays'])  + "\n" \
                       "                         >>Doppler: " + str(self.MultipathConfiguration['Doppler'])
        return ReturnString


    # --------------------------------------------------
    # Overload the copy function
    def __copy__(self):
        NewMultipathConfiguration = type(self)(self.MultipathConfiguration.copy())
        return NewMultipathConfiguration


    
    # ---------------------------------------------------
    # brief -> Initializes the class instance 
    # Input -> A Dictionary with the Multipath configuration
    def Initialize(self
                 , MultipathConfiguration: dict):
        # Error checking - Are all keys present?
        Valid = True
        if not 'Scalars' in MultipathConfiguration: Valid = False
        if not 'Delays'  in MultipathConfiguration: Valid = False
        if not 'Doppler' in MultipathConfiguration: Valid = False
        assert Valid, "The MultipathConfiguration dictoriary does not feature all required keys."

        # Do all values associated with the keys have the same size?
        Length = MultipathConfiguration['Scalars'].size
        if MultipathConfiguration['Delays'].size   != Length:  Valid = False
        if MultipathConfiguration['Doppler'].size  != Length:  Valid = False
        assert Valid, "The sizes of the MultipathConfiguration member entries are not equal."

        # Do the dimensions on the delays and doppler values make sense?
        if (abs(MultipathConfiguration['Delays'])  > 10e-6).sum() > 0: Valid = False
        if (abs(MultipathConfiguration['Doppler']) > 900).sum() > 0:   Valid = False
        assert Valid, "Either the 'Delays' or the 'Doppler' entries are out of bounds."

        self.MultipathConfiguration = MultipathConfiguration


    #---------------------------------------------------
    # The ComputeFreqResponse() function (Prototype 1) - Returns a resource element grid
    # --------------------------------------------------
    # brief   -> This function computes the frequency response of a ResourceGrid
    # Inputs  -> A CLteConst object and the number of slots in the desired resource grid
    # Outputs -> A ResourceGrid of type np.ndarray
    def ComputeFreqResponseReG(self
                             , LteConst:   CLteConst
                             , NumOfSlot:  int
                              ) -> np.ndarray:
        # Error Checking
        assert NumOfSlot <= 2*10240, "The number of slots in the resource grid is invalid."

        # Compute the frequency response over the resource grid
        NumOfdmSymbols = NumOfSlot * LteConst.NUM_SYMB_PER_SLOT

        # Get Number of paths
        NumPaths = len(self.MultipathConfiguration['Scalars'])

        # Define the output
        FrequencyResponseGrid = np.zeros([LteConst.NUM_SUBCARRIERS, NumOfdmSymbols], dtype=np.complex64)      

        for SymbolIndex in range(0, NumOfdmSymbols):
            for SubcarrierIndex in range(0, LteConst.NUM_SUBCARRIERS):
                Frequency, Time = LteConst.ComputeReCoordinates(SubcarrierIndex, SymbolIndex)
                FreqResponseForRe = float(0)
                for PathIndex in range(0, NumPaths):
                    Scalar  = self.MultipathConfiguration['Scalars'][PathIndex]
                    Delay   = self.MultipathConfiguration['Delays'][PathIndex]
                    Doppler = self.MultipathConfiguration['Doppler'][PathIndex]
                    FreqResponseTemp  = Scalar * np.exp(-1j*2*np.pi*Delay*Frequency)*np.exp(1j*2*np.pi*Doppler*Time)
                    FreqResponseForRe += FreqResponseTemp
                
                FrequencyResponseGrid[SubcarrierIndex, SymbolIndex] = FreqResponseForRe
        
        # We are all done. Return the frequency response
        return FrequencyResponseGrid
        


    # --------------------------------------------------
    # The ComputeFreqResponse() function (Prototype 2)
    # --------------------------------------------------
    ## brief   -> This function computes the frequency response for a list of CResourceElements
    #  Inputs  -> A CLteConst object and the number and the list of resource elements
    #  Outputs -> We return a second list of CResourceElements. The Freuency response is saved in the Value parameter
    def ComputeFreqResponseReL(self
                             , LteConst:        CLteConst    # Our CLteConst object
                             , ReInputList:     list         # The list of CResourceLements
                             ) ->list:

        # Get Number of paths
        NumPaths = len(self.MultipathConfiguration['Scalars'])

        ReOutputList = []
        # Loop through the list of CResourceElements
        for ReIndex in range(0, len(ReInputList)):
            ReOutputList.append(ReInputList[ReIndex].copy())   # Must copy by value. Need to create new object
            SubcarrierIndex, SymbolIndex, ReType = ReOutputList[ReIndex].ConvertToResourceGridCoordinate(LteConst)
            Frequency, Time                      = LteConst.ComputeReCoordinates(SubcarrierIndex, SymbolIndex)
            FreqResponseForRe = float(0)
            for PathIndex in range(0, NumPaths):
                Scalar  = self.MultipathConfiguration['Scalars'][PathIndex]
                Delay   = self.MultipathConfiguration['Delays'][PathIndex]
                Doppler = self.MultipathConfiguration['Doppler'][PathIndex]
                FreqResponseTemp  = Scalar * np.exp(-1j*2*np.pi*Delay*Frequency)*np.exp(1j*2*np.pi*Doppler*Time)
                FreqResponseForRe += FreqResponseTemp
                
            ReOutputList[ReIndex].Value = FreqResponseForRe

        # Return the ReOutputList
        return ReOutputList



    # --------------------------------------------------
    #  Declaration of PlotFrequencyResponse() function
    # --------------------------------------------------
    ## brief   -> This function plots the frequency response across the provided resource grid
    #  Inputs  -> The resource grid holding the complex frequency response values.
    #          -> The plotting configuration (eFRPlotConfig
    #  Outputs -> Pretty 3D plots of the Frequency response     
    def PlotFrequencyResponse(self 
                            , LteConst:                CLteConst       # The allmighty CLteConst object
                            , FrequencyResponseGrid:   np.ndarray      # The numpy array holding the Frequency Response
                            , FrPlotConfig:            eFrPlotConfig=eFrPlotConfig.Polar_KL # The plotting configuration
                             ):
        
        # Error Checking
        NumSubcarriers, NumOfdmSymbols = FrequencyResponseGrid.shape
        assert NumSubcarriers == LteConst.NUM_SUBCARRIERS, "The number of subcarriers in incompatible with signal bandwidth."
        
        # Building the X and Y grids needed for the 3D plot
        SubcarrierGrid    = np.zeros([NumSubcarriers, NumOfdmSymbols], dtype=np.float32)
        OfdmSymbolGrid    = np.zeros([NumSubcarriers, NumOfdmSymbols], dtype=np.float32)

        Xlabel_str = []
        Ylabel_str = []

        for k in range(0, NumSubcarriers):
            for l in range(0, NumOfdmSymbols):
                if(FrPlotConfig == eFrPlotConfig.Polar_KL or
                   FrPlotConfig == eFrPlotConfig.Rect_KL):
                    SubcarrierGrid[k, l] = k
                    OfdmSymbolGrid[k, l] = l
                else:
                    Frequency, Time = LteConst.ComputeReCoordinates(k, l)
                    SubcarrierGrid[k, l] = Frequency *  1/1000   # H/KHz
                    OfdmSymbolGrid[k, l] = Time * 1000           # msec/sec


        # Building the Z grid as Polar or Rectangular
        if(FrPlotConfig == eFrPlotConfig.Polar_KL or
           FrPlotConfig == eFrPlotConfig.Polar_FT):
            XGrid1 = abs(FrequencyResponseGrid)
            XGrid2 = np.angle(FrequencyResponseGrid)
            Titel1_Str = 'Magnitude Response of Channel'
            Titel2_str = 'Phase Response of Channel'
        else:
            XGrid1 = np.real(FrequencyResponseGrid)
            XGrid2 = np.imag(FrequencyResponseGrid)
            Titel1_Str = 'Real Portion of Channel Response'
            Titel2_str = 'Imaginary Portion of Channel Response'


        # Determining the x and y labels - Do you want k,l or frequency, time
        if(FrPlotConfig == eFrPlotConfig.Polar_KL or
           FrPlotConfig == eFrPlotConfig.Rect_KL):
            Xlabel_str = "Subcarriers"
            ylabel_str = "Ofdm Symbol Index"
        else:
            Xlabel_str = "Frequency (KHz)"
            Ylabel_str = "Time (msec)"

        # Build 3D plots of the Frequency Response
        fig = plt.figure(1)
        axes = fig.add_subplot(111, projection='3d')
        axes.plot_wireframe(SubcarrierGrid, OfdmSymbolGrid, XGrid1)
        if FrPlotConfig == eFrPlotConfig.Polar_KL:
           axes.set(xlim=(0, NumSubcarriers), ylim=(0, NumOfdmSymbols), zlim=(0, 1.1*XGrid1.max())) 
        elif FrPlotConfig == eFrPlotConfig.Polar_FT:
           axes.set(xlim=(SubcarrierGrid.min(), SubcarrierGrid.max()),
                    ylim=(OfdmSymbolGrid.min(), OfdmSymbolGrid.max()), zlim=(0, 1.1*XGrid1.max())) 
          
 
        plt.title(Titel1_Str)
        plt.xlabel(Xlabel_str)
        plt.ylabel(Ylabel_str)
         
        fig = plt.figure(2)
        axes = fig.add_subplot(111, projection='3d')
        axes.plot_wireframe(SubcarrierGrid, OfdmSymbolGrid, XGrid2)
        plt.title(Titel2_str)
        plt.xlabel(Xlabel_str)
        plt.ylabel(Ylabel_str)

        # Show the plots
        plt.ion()   # Interactive mode will make figures non-blocking
        plt.show()

