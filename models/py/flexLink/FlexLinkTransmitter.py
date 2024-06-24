# File:       FlexLinkTransmitter.py
# Notes:      This script unifies the transmit chain of the FlexLink modem.
#             Currently, it allow building all components of the packet except for PayloadB
#             as the construction of portion requires the complete definition of all MAC
#             configuration elements
#
# User Instructions to be done by the caller (see example in the test bench at the end of this file):
#             1. Construct the FlexLinkTransmitter
#                a. Decide what BW (LTE / WLAN) you wish to use.
#                b. Decide whether to create the packet at 20MHz (debugging) or 40MHz (for IQ DAC inputs)
#                   Producing the info at 40MHz makes it easy to low pass filter aliasing images after the DACs
#             2. Construct the CControlInformation object that defines all control information
#                You can print() the CControlInformation object to get an explanation of what the impact of
#                parameters is that where used to construct the CControlInformation
#             3. Initialize the resource grid using InitializeResourceGrid(). It requires the CControlInformation object
#                and the number of OFDM symbols that will be in the packet. This function will create the resource grid
#                of the right size and place control and reference signal QAM values into the first reference symbol
#             4. Construct the CSignalField object with the applicable parameters. Again, you may print the CSignalField
#                object to get an explanation of what the CSignalField object means for the packet.
#             5. Add the signal field using the AddSignalField() function. This will encode the signal field and map
#                its QAM values into the resource grid.
#             6. Create a list of the DataBlocks that will be CRCed, Encoded, Interleaved, RateMatched, QAM mapped
#                and mapped into the resource grid.
#                -> Initially the user may fill these blocks with random data. 
#                -> Later, the blocks will contain MAC information for a variaty of different purposes.
#                   i.e.: ARQ, ConfigurationInfo for PayloadA, ConfigurationInfo for PayloadB, ClientGrants, 
#                         UserData, Received SNR estimates, ... etc
#                Use the AddPayloadA() function to process the DataBlockList and map it into the resource grid.
#             7. Create a list of Transport blocks that will be placed in PayloadB. From this list of Transport blocks
#                create a number of DataBlockLists that hold the information for each Transport block.
#                -> Initially the user may fill the data blocks with random data. However, some of the MAC information
#                   in PayloadA must be valid so that the receiver can know where information is located in the 
#                   resource grid belonging to PayloadB.
#                Use the AddPayloadB() function to process the DataBlockLists and map them into the resource grid.
#             8. Use the BuildTxPacket() function to OFDM modulate the resource grid and prepend the Preamble.
#
# ---------------------------------------
# --------- Helper Functions ------------
#           1.  Use the PlotResourceGrid() function to visually inspect the resource grid.
#                        


__title__     = "FlexLinkTransmitter"
__author__    = "Andreas Schwarzinger"
__status__    = "preliminary"
__date__      = "April, 20th, 2024"
__copyright__ = 'Andreas Schwarzinger'

# --------------------------------------------------------
# Import Statements
# --------------------------------------------------------
from   FlexLinkParameters import *
from   FlexLinkCoder      import CCrcProcessor, CLdpcProcessor, CPolarProcessor, InterleaverFlexLink, CLinearFeedbackShiftRegister
from   QamMapping         import CQamMappingIEEE
import Preamble
import numpy              as np
import matplotlib.pyplot as plt


# --------------------------
# The CFlexLinkTransmitter contains the following functions
# 1. Constructor()             -> Requires the a bandwidth and oversampling argument
# 2. InitializeResourceGrid()  -> Requires a CControlInfo object and the number of OFDM symbols in the resource grid
#                                 This function initializes the resource grid and add control and reference signal information
# 3. AddSignalField()          -> Requires a CSignalField object. This function encodes the signal field information and
#                                 maps all related QAM values into the resource grid.
# 4. AddPayloadA()             -> Maps PayloadA into the resource grid
# 5. AddPayloadB()             -> Maps PayloadB into the resource grid
# 6. BuilTxdPacket()           -> This function builds two waveforms, one for TX0 and one for TX1, which is all 0+0j in case only TX0 is used. 
# 7. OfdmModulate()            -> This function does not require input arguments. It Ofdm modulates the information in the resource grid
# 8. PlotResourceGrid()        -> This function creates a PColor plot that shows to what purpose each resource element is assigned.
#                                 ControlInfo, ReferenceSignals, Data, DC, ...



# --------------------------------------------------------
# > Class: CFlexLinkTransmitter
# --------------------------------------------------------
class CFlexLinkTransmitter():
    '''
    brief: A class that unifies the FlexLink transmitter portion of the modem
    '''
    # -----------------------------------
    # General class attributes
    # -----------------------------------
    AntPort0                 = 0
    AntPort1                 = 1

    # ----------------------------------
    # > Function: Constructor
    # ----------------------------------
    def __init__(self
               , bLteBw:         bool = True
               , bOverSampling:  bool = True):
        
        # Error checking
        assert isinstance(bLteBw, bool),         'The input argument bLteBw must be a boolean value'
        assert isinstance(bOverSampling, bool),  'The input argument bOverSampling must be a boolean value'  
           
        # --------------------------------------------------
        # Set parameters based on the oversampling flag
        if bOverSampling == True:               # The IFFT will produce samples at 40MHz sample rate
            self.IfftSize        = 2048         # This is the suggested mode as the DAC should not be running at 20MHz but 40MHz --> 2048/40MHz = 51.2 microseconds
            self.CP_Length       = 232          # 232 / 40MHz = 5.8 microseconds
            self.SampleRate      = 40e6         # This reduces sample and hold distortion produced by the DAC and makes analog filtering easier.
        else:                                   # The IFFT will produce samples at 20MHz sample rate
            self.IfftSize        = 1024         # This mode is mostly meant for debugging and illustration                       --> 1024/20MHz = 51.2 microseconds
            self.CP_Length       = 116          # 116 / 20MHz = 5.8 microseconds
            self.SampleRate      = 20e6   
               
        self.ScSpacing           = self.SampleRate/self.IfftSize                                                          # = 19.53125KHz
        
        # ---------------------------------------------------
        # Set parameters based on the bandwidth flag (See the resource grid layout figure in the specification)
        if bLteBw == True:
            self.NumSubcarriers       = 913
            self.MaxSubcarrier        = 912
            self.CenterSubcarrier     = 456
            self.NumResourceBlocks    = 76
        else:
            self.NumSubcarriers       = 841
            self.MaxSubcarrier        = 840
            self.CenterSubcarrier     = 420
            self.NumResourceBlocks    = 70

        # The following parameters are required when mapping between the resource grid and the Fourier Transform Modules
        self.PosIfftIndices      = np.arange(0,                                     self.CenterSubcarrier + 1, 1, np.int16)   # These are the IFFT input indices, to which we 
        self.NegIfftIndices      = np.arange(self.IfftSize - self.CenterSubcarrier, self.IfftSize,          1, np.int16)      # connect the negative and positive subcarriers

        self.OccupiedBw           = self.NumSubcarriers * self.ScSpacing
        self.PosSubcarrierIndices = np.arange(self.CenterSubcarrier, self.NumSubcarriers,   1, np.int16)
        self.NegSubcarrierIndices = np.arange(0,                     self.CenterSubcarrier, 1, np.int16)


        # Uninitialized parameters 
        # --> self.ResourceGrid is self explanatory
        # --> self.ResourceGridEnum holds an integer representing the type of information present in the resource element 
        self.ResourceGrid       = None         # This will be initialized to a complex valued numpy matrix in the InitializeResourceGrid()
        self.ResourceGridEnum   = None         # This will be initialized to an enum   valued numpy matrix in the InitializeResourceGrid()
        self.ControlInformation = None         # An object that references the current control information
        self.SignalField        = None         # An object that references the current signal field information
        self.PayloadA           = None         # An object that references the current PayloadA information
        self.PayloadB           = None         # An object that references the current PayloadB information 
        
        # Construct some of the required objects for the transmitter
        self.PolarProcessor     = CPolarProcessor()
        self.LdpcProcessor      = None
        self.LFSR               = CLinearFeedbackShiftRegister(bUseMaxLengthTable = True, IndexOrTapConfig = 8)
        self.LFSR.InitializeShiftRegister([0, 0, 0, 0, 0, 0, 0, 1])
        self.ScrambingSequence  = self.LFSR.RunShiftRegister(NumberOutputBits= 255) 


        # Boolean Flags
        self.ResourceGridInitialized = False
        self.SignalFieldWasAdded     = False
        self.PayloadAWasAdded        = False
        self.PayloadBWasAdded        = False
        self.PacketWasBuild          = False



    # --------------------------------------------------------------------------------------------------------------------------------------- #
    # > Function: SFBC_Encode()                                                                                                               #
    # --------------------------------------------------------------------------------------------------------------------------------------- #
    @staticmethod
    def SFBC_Encode(X0: np.complex_
                  , X1: np.complex_) -> np.array:
        '''
        This function return the SFBC encoded information for a pair of input QAM values.
        '''
        Port0First  = X0
        Port0Second = X1
        Port1First  = -1*np.conj(X1)
        Port1Second = -1*np.conj(X0)
        Output      = np.array([Port0First, Port0Second, Port1First, Port1Second])
        
        return Output




    # --------------------------------------------------------------------------------------------------------------------------------------- #
    #                                                                                                                                         #
    #                                                                                                                                         #
    # > Function: InitializeResourceGrid()                                                                                                    #
    #                                                                                                                                         #
    #                                                                                                                                         #
    # --------------------------------------------------------------------------------------------------------------------------------------- #
    def InitializeResourceGrid(self
                             , ControlInfo: CControlInformation
                             , NumberOfdmSymbols: int):
        '''
        This function will populate the first symbol the resource grid based with control information and reference signals
        '''
        # Error checking
        assert isinstance(ControlInfo, CControlInformation), 'The ControlInfo input argument is of invalid type'
        assert isinstance(NumberOfdmSymbols, int),           'Only use integers for the NumberOfdmSymbols argument'
        assert NumberOfdmSymbols > 3,                        'The resource grid must have at least 3 OFDM symbols - 1 Ref Symbol, 1 Signal Field Symbol, 1 PayloadA symbol'

        # ----------------------------------
        # The resource grid is a matrix of complex values that will be read out to the IFFT
        # ----------------------------------
        MaxNumberTxAntennas     = 2
        self.ResourceGrid       = np.zeros([self.NumSubcarriers, NumberOfdmSymbols, MaxNumberTxAntennas], np.complex64)
        self.ResourceGridEnum   = EReType.Unknown.value * np.ones([self.NumSubcarriers, NumberOfdmSymbols, MaxNumberTxAntennas], np.uint16)
        self.ControlInformation = ControlInfo 

        # ----------------------------------
        # > Place control information into the first reference symbol
        # ----------------------------------
        NumberSubcarriers = self.ResourceGrid.shape[0]
        Ob                = -1                       # ControlInfoOpportunityIndex
        for k in range(0, NumberSubcarriers):        # k is the subcarrier index

            # Can we place a control element at this subcarrier???
            bControlElementOpportunity = (k % 3 == 0)
            if bControlElementOpportunity == False or k == self.CenterSubcarrier:
                continue

            # Increment the control opportunity index
            Ob                 += 1
            ControlBitIndex     = Ob % 12
            CurrentControlBit   = ControlInfo.ControlInformationArray[ControlBitIndex]
            ScramblingBit       = self.ScrambingSequence[Ob % len(self.ScrambingSequence)]
            ScrambledControlBit = (1 + CurrentControlBit + ScramblingBit) % 2 

            # Convert the scrambled control bit into a BPSK symbol
            QamSymbol = np.complex64(2*ScrambledControlBit - 1)
            
            # Place the QamSymbol - Only at Antenna port 0 as control information is only places there.
            self.ResourceGrid[k, 0, CFlexLinkTransmitter.AntPort0]    = QamSymbol
            self.ResourceGridEnum[k,0, CFlexLinkTransmitter.AntPort0] = EReType.Control.value

        # ----------------------------------
        # > Place the reference signals of antanna port 0 into the first reference symbol
        # ----------------------------------
        if ControlInfo.NumberTxAntennas == 1:
            Scaling = np.complex64(1)                   # BPSK modulation 0/1 -> -1 + j0 / 1 + j0 (No other scaling)
        else:  
            Scaling = np.complex64(1.414)               # Reference signals are BPSK modulated with 3dB boost 
                                                        # Thus bits 0/1 map to -1.414 + j0 / 1.414 + j0

        Rb = -1                                         # Reference signal opportunity Index
        for k in range(0, NumberSubcarriers):           # k is the subcarrier index
            bRefSignalP0Opportunity = ((k-2) % 3 == 0)  # Can we place a port 0 reference signal here???
            if bRefSignalP0Opportunity == True:
                Rb += 1
                ScrambledRefSignalBit = self.ScrambingSequence[Rb % len(self.ScrambingSequence)] % 2

                # Convert the scrambled reference bit into a BPSK symbol
                RefSignalP0 = np.complex64(2*ScrambledRefSignalBit - 1) * Scaling

                # Place the reference signal into the resource grid
                self.ResourceGrid[k, 0, CFlexLinkTransmitter.AntPort0]     = RefSignalP0
                self.ResourceGridEnum[k, 0, CFlexLinkTransmitter.AntPort0] = EReType.RefSignalPort0.value
                
                if ControlInfo.NumberTxAntennas == 2:
                    # The reference signal position in the other resource grid of the other antenna port must be empty
                    self.ResourceGrid[k, 0, CFlexLinkTransmitter.AntPort1]     = 0  
                    self.ResourceGridEnum[k, 0, CFlexLinkTransmitter.AntPort1] = EReType.Emtpy.value


        # ----------------------------------
        # > Place the reference signals of antanna port 1 into the first reference symbol
        # ----------------------------------
        Scaling = np.complex64(-1.414)               # Reference signals are BPSK modulated with 3dB boost 
                                                     # Thus bits 0/1 map to -1.414 + j0 / 1.414 + j0

        Rb = -1                                      # Reference signal opportunity Index
        for k in range(0, NumberSubcarriers):        # k is the subcarrier index
            bRefSignalP1Opportunity = ((k-1) % 3 == 0) and ControlInfo.NumberTxAntennas == 2
            if bRefSignalP1Opportunity == True:
                Rb += 1
                ScrambledRefSignalBit = self.ScrambingSequence[Rb % len(self.ScrambingSequence)] % 2

                # Convert the scrambled reference bit into a BPSK symbol
                RefSignalP1 = np.complex64(2*ScrambledRefSignalBit - 1) * Scaling

                # Place the reference signal into the resource grid
                self.ResourceGrid[k, 0, CFlexLinkTransmitter.AntPort1]     = RefSignalP1
                self.ResourceGridEnum[k, 0, CFlexLinkTransmitter.AntPort1] = EReType.RefSignalPort1.value

                # The reference signal position in the other resource grid of the other antenna port must be empty
                self.ResourceGrid[k, 0, CFlexLinkTransmitter.AntPort0]     = 0  
                self.ResourceGridEnum[k, 0, CFlexLinkTransmitter.AntPort0] = EReType.Emtpy.value


        # ----------------------------------
        # > Place the reference signals of antanna port 0 into the remaining reference symbols
        # ----------------------------------
        if ControlInfo.NumberTxAntennas == 1:
            Scaling = np.complex64(1)                   # BPSK modulation 0/1 -> -1 + j0 / 1 + j0 (No other scaling)
        else:  
            Scaling = np.complex64(-1.414)              # Reference signals are BPSK modulated with 3dB boost 
                                                        # Thus bits 0/1 map to -1.414 + j0 / 1.414 + j0

        for l in range(1, NumberOfdmSymbols):               # l is the OFDM symbol index
            if l % ControlInfo.ReferenceSymbolPeriodicity != 0:
                continue                                    # Only proceed if l is a reference symbol

            for n in range(0, 1000):      
                ScrambledRefSignalBit = self.ScrambingSequence[n % len(self.ScrambingSequence)] % 2  
                RefSignalP0           = np.complex64(2*ScrambledRefSignalBit - 1) * Scaling      
                match ControlInfo.ReferenceSignalSpacing:
                    case 3:
                        k = 2 + n * 3
                    case 6:
                        k = 2 + n * 6
                    case 12:
                        k = 5 + n * 12
                    case 24:
                        k = 11 + n * 24            
                    case _:
                        assert False
                if k >= self.NumSubcarriers:
                    break
                else:
                    # Place the reference signal into the resource grid
                    self.ResourceGrid[k, l, CFlexLinkTransmitter.AntPort0]     = RefSignalP0
                    self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort0] = EReType.RefSignalPort0.value
                    RefSignalP0                                               *= np.sign(np.random.randn(1))[0]
                    
                    if ControlInfo.NumberTxAntennas == 2:
                        # The reference signal position in the other resource grid of the other antenna port must be empty
                        self.ResourceGrid[k, 0, CFlexLinkTransmitter.AntPort1]     = 0  
                        self.ResourceGridEnum[k, 0, CFlexLinkTransmitter.AntPort1] = EReType.Emtpy.value


        # ----------------------------------
        # > Place the reference signals of antanna port 1 into the remaining reference symbols
        # ----------------------------------
        Scaling = np.complex64(-1.414)              # Reference signals are BPSK modulated with 3dB boost 
                                                        # Thus bits 0/1 map to -1.414 + j0 / 1.414 + j0

        for l in range(1, NumberOfdmSymbols):           # l is the OFDM symbol index
            if ControlInfo.NumberTxAntennas == 1:
                break                                   # Exit as there is one one antenna
            if l % ControlInfo.ReferenceSymbolPeriodicity != 0:
                continue                                # Only proceed if l is a reference symbol

            for n in range(0, 1000):  
                ScrambledRefSignalBit = self.ScrambingSequence[n % len(self.ScrambingSequence)] % 2  
                RefSignalP1           = np.complex64(2*ScrambledRefSignalBit - 1) * Scaling   
                match ControlInfo.ReferenceSignalSpacing:
                    case 3:
                        k = 1 + n * 3
                    case 6:
                        k = 1 + n * 6
                    case 12:
                        k = 4 + n * 12
                    case 24:
                        k = 10 + n * 24            
                    case _:
                        assert False

                if k >= self.NumSubcarriers:
                    break
                else:
                    # Place the reference signal into the resource grid
                    self.ResourceGrid[k, l, CFlexLinkTransmitter.AntPort1]     = RefSignalP1
                    self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort1] = EReType.RefSignalPort1.value
                     

                    # The reference signal position in the other resource grid of the other antenna port must be empty
                    self.ResourceGrid[k, 0, CFlexLinkTransmitter.AntPort0]     = 0  
                    self.ResourceGridEnum[k, 0, CFlexLinkTransmitter.AntPort0] = EReType.Emtpy.value


        # ----------------------------------
        # > Place the DC Signals in the correct area of the resource grid
        # ----------------------------------
        Half    = int(ControlInfo.NumberDcSubcarriers / 2)
        DcRange = range(self.CenterSubcarrier - Half, self.CenterSubcarrier + Half + 1)

        for l in range(0, NumberOfdmSymbols):               # l is the OFDM symbol index
            
            for k in DcRange:                               # k is the subcarrier index
                if self.ResourceGridEnum [k, l, CFlexLinkTransmitter.AntPort0] == -1:       # Nothing has yet been placed here
                    self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort0] = EReType.Emtpy.value
                    self.ResourceGrid    [k, l, CFlexLinkTransmitter.AntPort0] = np.complex64(0)

                if ControlInfo.NumberTxAntennas == 2:
                    if self.ResourceGridEnum [k, l, CFlexLinkTransmitter.AntPort1] == -1:       # Nothing has yet been placed here
                        self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort1] = EReType.Emtpy.value
                        self.ResourceGrid    [k, l, CFlexLinkTransmitter.AntPort1] = np.complex64(0)


        # Indicate that the resource grid was initialized
        self.ResourceGridInitialized = True
        self.SignalFieldWasAdded     = False
        self.PayloadAWasAdded        = False
        self.PayloadBWasAdded        = False
        self.PacketWasBuild          = False











    # --------------------------------------------------------------------------------------------------------------------------------------- #
    #                                                                                                                                         #
    # > AddSignalField()                                                                                                                      #
    #                                                                                                                                         #                                                                                                    
    # --------------------------------------------------------------------------------------------------------------------------------------- #
    def AddSignalField(self
                     , SignalField: CSignalField):
        '''
        This function adds the Signal field to the Resource Grid
        '''

        # Error checking
        assert isinstance(SignalField, CSignalField),                            'The SignalField input argument is of improper type'
        assert len(SignalField.SignalFieldArray) == 64 or len(SignalField.SignalFieldArray) == 290, 'The signal field array is invalid'
        assert isinstance(self.ControlInformation, CControlInformation),         'The control information is invalid'
        assert self.ResourceGridInitialized,                                     'The resource grid must be initialized before adding the signal field'


        SignalFieldFormat        = self.ControlInformation.SignalFieldFormat         
        assert SignalFieldFormat == 1 or SignalFieldFormat == 2,                 'The Signal field format must be either 1 or 2'
        
        if SignalFieldFormat == 1:
            assert isinstance(SignalField.BPS_A_Flag, int),                      'The signal field format 1 must have only one BPS value in BPS_A_Flag'
        else:
            assert len(SignalField.BPS_A_Flag) == 76,                            'The signal field format 2 must have       76 BPS values in BPS_A_Flag'

        NumberSignalFieldSymbols = self.ControlInformation.NumberSignalFieldSymbols 
        assert NumberSignalFieldSymbols == 1 or NumberSignalFieldSymbols == 2 or \
               NumberSignalFieldSymbols == 4 or NumberSignalFieldSymbols == 10,  'Unexpected number of signal field OFDM symbols'
        
        NumBitsPerQamValue       = self.ControlInformation.NumBitsPerQamSymbols
        assert NumBitsPerQamValue == 1 or NumBitsPerQamValue == 2,               'The number of bits per QAM symbol is invalid'
        
        NumberTxAntennas         = self.ControlInformation.NumberTxAntennas  
        assert NumberTxAntennas == 1 or NumberTxAntennas == 2,                   'The number of TX antennas is invalid'


        # Save off the signal field and extract relevant data from the self.ControlInformation object
        self.SignalField = SignalField 

        # --------------------------------------------------------------------------------
        # Step 1: Polar encode the signal field 
        # --------------------------------------------------------------------------------
        #         Note, the CRC has already been attached to the SignalField
        #         SignalFieldFormat1: Use  256 bit polar encoder with p = 0.55
        #         SignalFieldFormat2: Use 1024 bit polar encoder with p = 0.55
        #         p is the erasure probabiliy, which we keep at 0.55
        p   = 0.55

        if SignalFieldFormat == 1:
            N = 256     # Number of encoded bits
            K = 64      # Number of message bits
        else:
            N = 1024    # Number of encoded bits
            K = 290     # Number of message bits

        ErasureProbabilities          = self.PolarProcessor.FindChannelProbabilities(N, p)
        SortedProbabilities           = np.sort(ErasureProbabilities)  # Nice for debugging, but nothing else
        SortedIndices                 = np.argsort(ErasureProbabilities)
        MessageBitIndices             = SortedIndices[0: K]
        FrozenBitIndices              = SortedIndices[K:]              # Technically, this object is only needed in the decoder

        SourceBits                    = np.zeros(N, np.uint8)
        SourceBits[MessageBitIndices] = self.SignalField.SignalFieldArray.copy()
        EncodedBits                   = self.PolarProcessor.RunPolarEncoder(SourceBits)

        # ----------------------------------------------------------------------------
        # Step 2: Interleave the encoded information (simulation will have to show how useful this step actually is.)
        # ----------------------------------------------------------------------------
        FEC_Mode = 1   # Polar
        if SignalFieldFormat == 1:
            CBS_A_Flag = 0 # Indicates 256 encoded bits
        else:
            CBS_A_Flag = 2 # Indicates 1024 encoded bits

        InterleavingIndices = InterleaverFlexLink(FEC_Mode, CBS_A_Flag)

        InterleavedBits                      = np.zeros(N, np.uint8)
        InterleavedBits[InterleavingIndices] = EncodedBits 

        # ----------------------------------------------------------------------------
        # Step 3: Rate Matching / Repetition
        # ----------------------------------------------------------------------------
        # First we need to find out how many resource elements are available within the number of OFDM
        # symbols that we have available for the signal field
        ReCount           = 0
        NumberSubcarriers = self.ResourceGrid.shape[0]

        # For single antenna case
        if self.ControlInformation.NumberTxAntennas == 1:
            for l in range(1, NumberSignalFieldSymbols + 1):  # Remember, symbol l = 0 is reserved for the control information
                for k in range(0, NumberSubcarriers):
                    ResourceElementValue     = self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort0]
                    bAvailableForSignalField = ResourceElementValue != EReType.Emtpy.value          and \
                                               ResourceElementValue != EReType.RefSignalPort0.value  
                    
                    if bAvailableForSignalField == True:
                        # This statement is really meant for visualization when calling PlotResourceGrid()
                        self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort0]     = EReType.SignalField.value
                        if self.ControlInformation.NumberTxAntennas == 2:
                            self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort1] = EReType.SignalField.value
                        
                        ReCount += 1

        # Dual Antenna case
        else:   
            for l in range(1, NumberSignalFieldSymbols + 1):  # Remember, symbol l = 0 is reserved for the control information
                for k in range(0, NumberSubcarriers):
                    # Count the resource elements for self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort0]
                    ResourceElementValueP0    = self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort0]
                    ResourceElementValueP1    = self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort1]
                    bAvailableForSignalField  = ResourceElementValueP0 != EReType.Emtpy.value          and \
                                                ResourceElementValueP0 != EReType.RefSignalPort0.value and \
                                                ResourceElementValueP1 != EReType.Emtpy.value          and \
                                                ResourceElementValueP1 != EReType.RefSignalPort1.value
                    
                    if bAvailableForSignalField == True:
                        # This statement is really meant for visualization when calling PlotResourceGrid()
                        self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort0] = EReType.SignalField.value
                        if self.ControlInformation.NumberTxAntennas == 2:
                            self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort1] = EReType.SignalField.value
                        
                        ReCount += 1

        # Compute the number of available bits 
        NumAvailableSignalFieldBits = ReCount * NumBitsPerQamValue

        # Create the rate matched bit array
        RateMatchedBits = np.zeros(NumAvailableSignalFieldBits, np.uint8)                    
        for BitIndex in range(0, NumAvailableSignalFieldBits):
            RateMatchedBits[BitIndex] = InterleavedBits[BitIndex % N]


        # ------------------------------------------------------------------------
        # Step 4: Scramble the Rate Matched bits
        # ------------------------------------------------------------------------
        for BitIndex in range(0, NumAvailableSignalFieldBits):
            RateMatchedBits[BitIndex] = (RateMatchedBits[BitIndex]  + self.ScrambingSequence[BitIndex % len(self.ScrambingSequence)]) % 2



        # ------------------------------------------------------------------------
        # Step 5: Convert the Rate Matched Bit stream into BPSK or QPSK values and map them into the Resource Grid
        # ------------------------------------------------------------------------
        if self.ControlInformation.NumberTxAntennas == 1:
            StartBit = 0
            for l in range(1, NumberSignalFieldSymbols + 1):  # Remember, symbol l = 0 is reserved for the control information
                for k in range(0, NumberSubcarriers):

                    if self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort0] == EReType.SignalField.value:
                        assert StartBit <= len(RateMatchedBits) - NumBitsPerQamValue
                        self.ResourceGrid[k,l, CFlexLinkTransmitter.AntPort0] = CQamMappingIEEE.Mapping(NumBitsPerQamValue, RateMatchedBits[StartBit:StartBit+NumBitsPerQamValue])[0]
                        StartBit += NumBitsPerQamValue

            # Ensure that we have gone through the entire RateMatchedBits array. If not, there is a mapping error
            assert StartBit == len(RateMatchedBits), 'A mapping error has occured.'  

        else:   
            # We have two antenna ports and we must use SFBC during the mapping process
            StartBit                    = 0
            SignalFieldOpportunityCount = 0
            CurrentReLocation           = [0, 1]
            LastReLocation              = [0, 1]
            for l in range(1, NumberSignalFieldSymbols + 1):  # Remember, symbol l = 0 is reserved for the control information
                for k in range(0, NumberSubcarriers):
                    CurrentReLocation           = [k, l]

                    # Only take action if the current resource element is reserved for the signal field
                    if self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort0] == EReType.SignalField.value:
                        # Ensure that the resource element in the P1 resource grid is also reserved for the signal field                          
                        assert self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort1] == EReType.SignalField.value, \
                                                'For the two antenna case, the ReType in both resource grids must match'
                        
                        SignalFieldOpportunityCount       += 1

                        # Every two signal field opportunities we will want to map the SFCB encoded information
                        if SignalFieldOpportunityCount % 2 == 0:
                            assert StartBit <= len(RateMatchedBits) - 2 * NumBitsPerQamValue
                            X0  = CQamMappingIEEE.Mapping(NumBitsPerQamValue, RateMatchedBits[StartBit:StartBit+NumBitsPerQamValue])
                            StartBit += NumBitsPerQamValue
                            X1  = CQamMappingIEEE.Mapping(NumBitsPerQamValue, RateMatchedBits[StartBit:StartBit+NumBitsPerQamValue])
                            StartBit += NumBitsPerQamValue                        

                            SFBC_Symbols = CFlexLinkTransmitter.SFBC_Encode(X0, X1)
                            Port0First   = SFBC_Symbols[0] 
                            Port0Second  = SFBC_Symbols[1] 
                            Port1First   = SFBC_Symbols[2] 
                            Port1Second  = SFBC_Symbols[3] 

                            Current_k = CurrentReLocation[0]
                            Current_l = CurrentReLocation[1]
                            Last_k    = LastReLocation[0]
                            Last_l    = LastReLocation[1]

                            self.ResourceGrid[Last_k,    Last_l,    CFlexLinkTransmitter.AntPort0 ] = Port0First[0]
                            self.ResourceGrid[Current_k, Current_l, CFlexLinkTransmitter.AntPort0 ] = Port0Second[0]
                            self.ResourceGrid[Last_k,    Last_l,    CFlexLinkTransmitter.AntPort1 ] = Port1First[0]
                            self.ResourceGrid[Current_k, Current_l, CFlexLinkTransmitter.AntPort1 ] = Port1Second[0]  

                        LastReLocation = CurrentReLocation

            # Ensure that we have gone through the entire RateMatchedBits array. If not, there is a mapping error
            assert StartBit == len(RateMatchedBits), 'A mapping error has occured.'  

        # Indicate the the signal field was added
        self.SignalFieldWasAdded     = True
        self.PayloadAWasAdded        = False
        self.PayloadBWasAdded        = False
        self.PacketWasBuild          = False

















    # --------------------------------------------------------------------------------------------------------------------------------------- #
    #                                                                                                                                         #
    # > AddPayloadA() - Using Vertical Mapping                                                                                                #
    #                                                                                                                                         #                                                                                                    
    # --------------------------------------------------------------------------------------------------------------------------------------- #
    def AddPayloadA(self
                  , DataBlockList: list):
        '''
        This function adds the data blocks provided by the MAC
        '''
        # From the Signal field information, we can figure out what the data block size must be.
        CodeBlockSize = self.SignalField.CodeBlockSizeA
        DataWordSize  = int(CodeBlockSize * self.SignalField.CodingRate)
        DataBlockSize = DataWordSize - 16                              # 16 CRC bits


        # Error Checking
        assert self.SignalFieldWasAdded,                                    'You must add the signal field before adding payloadA'
        assert isinstance(DataBlockList, list)
        for ListIndex in range(0, len(DataBlockList)):
            assert isinstance(DataBlockList[ListIndex], np.ndarray),        'Each data block must be a numpy array'
            assert np.issubdtype(DataBlockList[ListIndex].dtype, np.uint8), 'Each data block must a numpy array of type np.uint8'
            
            assert len(DataBlockList[ListIndex]) == DataBlockSize,          'The number of bits in the Data Blocks is invalid'
            for BitIndex in range(0, DataBlockSize):
                assert DataBlockList[ListIndex][BitIndex] == 0 or DataBlockList[ListIndex][BitIndex], 'Only 0s and 1s are allowed in the data blocks'


        # If LDPC processing is desired, we need to construct the CLdpcProcessor object
        if self.SignalField.FEC_Mode == 0:
            match self.SignalField.FEC_A_Flag:
                case 1:
                    strEncodingRate = '2/3'
                case 2:
                    strEncodingRate = '3/4'
                case 3:
                    strEncodingRate = '5/6'
                case _:
                    strEncodingRate = '1/2'

            # Construct the LdpcProcessor
            self.LdpcProcessor('WLAN', None, CodeBlockSize, strEncodingRate)
        else:                                        # Configure the Polar encoder
            p                     = 0.55             # The base erasure probability
            N                     = CodeBlockSize    # The number of final encoded bits (code block size)
            K                     = DataWordSize     # The number of bits we want to encode
            ErasureProbabilities  = self.PolarProcessor.FindChannelProbabilities(CodeBlockSize, p)
            SortedIndices         = np.argsort(ErasureProbabilities)
            MessageBitIndices     = SortedIndices[0: K]


        # Determine where we will start to map PayloadA information into the ResourceGrid
        StartOfdmSymbolIndex    = self.ControlInformation.NumberSignalFieldSymbols + 1     
        StartSubcarrierIndex    = 0                               

        # To make our life easier let's create an array of 76 entries representing the number of bits
        # per QAM symbol for each resource block. This will facilitate the accounting of available resource
        # when we deteremine the number of bits available in each resource block
        if self.SignalField.SignalFieldFormat == 1:
            BpsEachResourceBlock = self.SignalField.BitsPerQamSymbol * np.ones(76, np.uint8)
        else:
            BpsEachResourceBlock = self.SignalField.BitsPerQamSymbol   # Which must be a vector this time


        # Compute the CRC, Encode, interleave, rate match and map each data block into the resource grid
        for ListIndex in range(0, len(DataBlockList)):
            DataBlock = DataBlockList[ListIndex]

            # -----------------------------------------------------------------------
            # > Compute the CRC and form the data word
            # -----------------------------------------------------------------------
            DataWord                  = np.zeros(DataBlockSize + 16, np.uint8)
            DataWord[0:DataBlockSize] = DataBlock

            CrcOutput = CCrcProcessor.ComputeCrc(CCrcProcessor.Generator16_LTE, DataBlock.tolist())

            # The CRC Bits are inserted after the data block bits
            for BitIndex in range(DataBlockSize, DataBlockSize + 16):
                DataWord[BitIndex] = CrcOutput[BitIndex - DataBlockSize]


            # -----------------------------------------------------------------------
            # > FEC encode the data word into an encoded data word
            # -----------------------------------------------------------------------
            if self.SignalField.FEC_Mode == 0:     # This is LDCP coding
                EncodedDataWord               = self.LdpcProcessor.EncodeBits(DataWord)
            else:
                SourceBits                    = np.zeros(N, np.uint8)
                SourceBits[MessageBitIndices] = DataWord 
                EncodedDataWord               = self.PolarProcessor.RunPolarEncoder(SourceBits)         

            assert len(EncodedDataWord) == CodeBlockSize, 'An error occured during the FEC encoding process'
                
                
            # -----------------------------------------------------------------------
            # > Interleave the encoded data word to form the code block
            # -----------------------------------------------------------------------    
            InterleavingIndices            = InterleaverFlexLink(self.SignalField.FEC_Mode, self.SignalField.CBS_A_Flag)
            CodeBlock                      = np.zeros(N, np.uint8)
            CodeBlock[InterleavingIndices] = EncodedDataWord 

            # -----------------------------------------------------------------------
            # > Determine the minimum size of the code word, which is the rate matched bit stream.
            # -----------------------------------------------------------------------    
            MinimumCodeWordSize            = int(CodeBlockSize * (1 + self.SignalField.RateMatchingFactor))

            # ----------------------------------------------------------------------
            # > Determine the actual number of rate matched bits, which will be >= MinimumCodeWordSize. Remember, the MinimumCodeWordSize
            #   will likely not completely fill up the last resource block. We must find the number of code word bits that will in 
            #   fact fill the last resource block completely. To find the actual number of code word bits, we need to iterate through
            #   self.ResourceGridEnum and see which resource elements are available for placement of PayloadA data. We count the number of
            #   resource elements available until we get to the subcarrier index that fills a complete resource block. Then we check to 
            #   see whether the code word bit count is >= MinimumCodeWordSize. This has to be done for both the single and dual TX antenna case
            # ----------------------------------------------------------------------
            # We start at coordinates [k, l] in the resource grid and start counting bits
            NumCodeWordBits       = 0                        # Keep track of the number of bits available for PayloadA in the resource grid
            NumResourceBlocksUsed = 0                        # Keep track of the number of resource blocks used
            k                     = StartSubcarrierIndex 
            l                     = StartOfdmSymbolIndex
            bFirstPass            = True

            while True:
                k = k % self.NumSubcarriers

                # Don't take any action if you hit the center subcarrier
                if k == self.CenterSubcarrier:
                    k += 1 
                    continue

                # Check to see whether the last resource blocks has been completely used. Be careful here.
                # The center carrier does not belong to any resource block. Therefore, to detect whether a resource block
                # has been filled for subcarrier indices, k, that are larger than self.CenterSubcarrier, we need to 
                # make an adjustment
                # If kVirtual = 0,  then this is the first subcarrier of the current resource block
                # if kVirtual = 11, then we are at the last subcarrier of the current resource block
                kVirtual = k
                if kVirtual > self.CenterSubcarrier: 
                    kVirtual = k -1 
                    
                CurrentResourceBlockIndex   = int(kVirtual / 12)            # As there are 12 subcarriers in a resource block    
                # Did we completely fill a resource block during the last iteration????    
                bLastResourceBlockFull = kVirtual % 12 == 0 and bFirstPass == False

                # If the a resource block was filled during the last iteration, then check to see whether we have used all bits 
                # in the code work. If so, let's break out of the loop as we have found the actual number of bits needed for the CodeWord.
                if bLastResourceBlockFull == True:
                    NumResourceBlocksUsed += 1
                    # Note, the actual number of bits that we use will likely be slightly larger than the
                    # MinimumCodeWordSize. Remember, that we want to fill up an integer number of resource blocks
                    # The MinimumCodeWordSize may not be enough bits to do that. 
                    #print(NumCodeWordBits/NumResourceBlocksUsed)
                    if NumCodeWordBits >= MinimumCodeWordSize:
                        break

                CurrentResourceElementTypeP0 = self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort0]
                CurrentResourceElementTypeP1 = self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort1]

                # This is an idiots check. Just make sure we are not doing something silly like attempting to write on top of the signal field
                assert CurrentResourceElementTypeP0 != EReType.SignalField.value, 'We should not be mapping PayloadA on top of the signal field'
                assert CurrentResourceElementTypeP1 != EReType.SignalField.value, 'We should not be mapping PayloadA on top of the signal field'

                bAvailableForPayloadA       = CurrentResourceElementTypeP0 != EReType.Emtpy.value          and \
                                              CurrentResourceElementTypeP0 != EReType.RefSignalPort0.value and \
                                              CurrentResourceElementTypeP0 != EReType.Control              and \
                                              CurrentResourceElementTypeP1 != EReType.Emtpy.value          and \
                                              CurrentResourceElementTypeP1 != EReType.RefSignalPort1.value 

                # The BPS in each resource block can vary depending on the channel conditions. The SignalFieldFormat2 is used to
                # To construct a BpsEachResourceBlock array that has different BPS in the entries.
                # For SignalFieldFormat1, BpsEachResourceBlock would have the same BPS value in all entries                                      
                BPS                         = BpsEachResourceBlock[CurrentResourceBlockIndex]

                # If BPS is 0, then it was determined to not place any data in this resource block
                if BPS == 0:
                    bAvailableForPayloadA = False

                # If we can in fact place payload A data here, then ... do it.
                if bAvailableForPayloadA == True:    
                    NumCodeWordBits           += BPS
                    if ListIndex % 2 == 0:   # Thus even
                        self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort0] = EReType.PayloadAEvenCodeWord.value
                        if self.ControlInformation.NumberTxAntennas == 2:
                            self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort1] = EReType.PayloadAEvenCodeWord.value
                    else:
                        self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort0] = EReType.PayloadAOddCodeWord.value
                        if self.ControlInformation.NumberTxAntennas == 2:
                            self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort1] = EReType.PayloadAOddCodeWord.value
                    
                #print('k = ' + str(k) + '    Available: ' + str(bAvailableForPayloadA) + '   NumberOfBits: ' + str(NumCodeWordBits))

                # Increment the subcarrier and OFDM symbol index
                k += 1                                                  # Proceed to next subcarrier
                if k % self.NumSubcarriers == 0:                        # Proceed to next OFDM symbol
                    l += 1      

                bFirstPass = False

            #print('CodeWord: ' + str(ListIndex) + '  Resource blocks used: ' + str(NumResourceBlocksUsed) + "    CodeWordSize: " + str(NumCodeWordBits))

            # -----------------------------------------------------------------------
            # > With the correct code word size, execute the rate matching operation
            # -----------------------------------------------------------------------
            CodeWordSize     = NumCodeWordBits
            CodeWord         = np.zeros(CodeWordSize, np.uint8)
            for BitIndex in range(0, CodeWordSize):
                CodeWord[BitIndex] = CodeBlock[BitIndex % CodeBlockSize]   # Rate matching / Repetition happens here


            # ------------------------------------------------------------------------
            # > Scramble the Rate Matched bits
            # ------------------------------------------------------------------------
            for BitIndex in range(0, len(CodeWord)):
                CodeWord[BitIndex] = (CodeWord[BitIndex]  + self.ScrambingSequence[BitIndex % len(self.ScrambingSequence)]) % 2


            # -----------------------------------------------------------------------
            # > Map the CodeWord 
            # -----------------------------------------------------------------------
            NumCodeWordBits          = 0
            NumResourceBlocksUsed    = 0
            k                        = StartSubcarrierIndex       # Remember k is the subcarrier index
            l                        = StartOfdmSymbolIndex       # Remember l is the OFDM symbol index
            bFirstPass               = True
            PayloadAOpportunityCount = 0
            CurrentReLocation        = [0,0]
            LastReLocation           = [0, StartOfdmSymbolIndex]
            while True:
                k = k % self.NumSubcarriers

                # Don't take any action if you hit the center subcarrier
                if k == self.CenterSubcarrier:
                    k += 1 
                    continue

                # Check to see whether the last resource blocks has been completely used. Be careful here.
                # The center carrier does not belong to any resource block. Therefore, to detect whether a resource block
                # has been filled for subcarrier indices, k, that are larger than self.CenterSubcarrier, we need to 
                # make an adjustment
                # If kVirtual = 0,  then this is the first subcarrier of the current resource block
                # if kVirtual = 11, then we are at the last subcarrier of the current resource block
                kVirtual = k
                if kVirtual > self.CenterSubcarrier: 
                    kVirtual = k - 1 
                    
                CurrentResourceBlockIndex   = int(kVirtual / 12)            # As there are 12 subcarriers in a resource block    
                # Did we completely fill a resource block during the last iteration????    
                bLastResourceBlockFull      = kVirtual % 12 == 0 and bFirstPass == False

                # If the resource block is full, then check to see whether we have used all bits in the code work.
                # If so, let's break out of the loop as we have found the actual number of bits needed for the CodeWord
                if bLastResourceBlockFull == True:
                    NumResourceBlocksUsed += 1
                    # Note, the actual number of bits that we use will likely be slightly larger than the 
                    # MinimumCodeWordSize. Remember, that we want to fill up an integer number of resource blocks
                    # The MinimumCodeWordSize may not be enough bits to do that. 
                    if NumCodeWordBits >= MinimumCodeWordSize:
                        break

                CurrentResourceElementTypeP0 = self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort0]
                CurrentResourceElementTypeP1 = self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort1] 

                # This is an idiots check. Just make sure we are not doing something silly like attempting to write on top of the signal field
                assert CurrentResourceElementTypeP0 != EReType.SignalField.value, 'We should not be mapping PayloadA on top of the signal field'
                assert CurrentResourceElementTypeP1 != EReType.SignalField.value, 'We should not be mapping PayloadA on top of the signal field'
                
                bAvailableForPayloadA       = CurrentResourceElementTypeP0 == EReType.PayloadAEvenCodeWord.value or \
                                              CurrentResourceElementTypeP0 == EReType.PayloadAOddCodeWord.value            
                
                # The BPS in each resource block can vary depending on the channel conditions. The SignalFieldFormat2 is used to
                # To construct a BpsEachResourceBlock array that has different BPS in the entries.
                # For SignalFieldFormat1, BpsEachResourceBlock would have the same BPS value in all entries  
                BPS                         = BpsEachResourceBlock[CurrentResourceBlockIndex]

                # If BPS is 0, then it was determined to not place any data in this resource block
                if BPS == 0:
                    self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort0]   = EReType.Emtpy.value
                    self.ResourceGrid    [k, l, CFlexLinkTransmitter.AntPort0]   = 0 + 0j
                    if self.ControlInformation.NumberTxAntennas == 2:
                        self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort1] = EReType.Emtpy.value
                        self.ResourceGrid    [k, l, CFlexLinkTransmitter.AntPort1] = 0 + 0j
                    bAvailableForPayloadA = False

                # If we can in fact place payload A data here, then ... do it.
                if bAvailableForPayloadA == True:
                    if self.ControlInformation.NumberTxAntennas == 1:
                        QamValue                                               = CQamMappingIEEE.Mapping(int(BPS), CodeWord[NumCodeWordBits:NumCodeWordBits + BPS])
                        self.ResourceGrid[k,l, CFlexLinkTransmitter.AntPort0]  = QamValue[0]
                        NumCodeWordBits                                       += BPS

                    else:
                        CurrentReLocation         = [k, l]
                        PayloadAOpportunityCount += 1

                        # We want to take action every second resource element opportunity for PayloadA
                        # Otherwise, we can't execute the SFBC algorithm
                        if PayloadAOpportunityCount % 2 == 0:    
                            assert NumCodeWordBits <= CodeWordSize - 2 * BPS
                            X0  = CQamMappingIEEE.Mapping(int(BPS), CodeWord[NumCodeWordBits:NumCodeWordBits+BPS])
                            NumCodeWordBits += BPS
                            X1  = CQamMappingIEEE.Mapping(int(BPS), CodeWord[NumCodeWordBits:NumCodeWordBits+BPS])
                            NumCodeWordBits += BPS                        

                            SFBC_Symbols = CFlexLinkTransmitter.SFBC_Encode(X0, X1)
                            Port0First   = SFBC_Symbols[0]
                            Port0Second  = SFBC_Symbols[1]
                            Port1First   = SFBC_Symbols[2]
                            Port1Second  = SFBC_Symbols[3]

                            Current_k = CurrentReLocation[0]
                            Current_l = CurrentReLocation[1]
                            Last_k    = LastReLocation[0]
                            Last_l    = LastReLocation[1]

                            self.ResourceGrid[Last_k,    Last_l,    CFlexLinkTransmitter.AntPort0 ] = Port0First[0]
                            self.ResourceGrid[Current_k, Current_l, CFlexLinkTransmitter.AntPort0 ] = Port0Second[0]
                            self.ResourceGrid[Last_k,    Last_l,    CFlexLinkTransmitter.AntPort1 ] = Port1First[0]
                            self.ResourceGrid[Current_k, Current_l, CFlexLinkTransmitter.AntPort1 ] = Port1Second [0]        

                        LastReLocation = CurrentReLocation                                   
      

                # print('k = ' + str(k) + '    Available: ' + str(bAvailableForPayloadA) + '   NumberOfBits: ' + str(NumCodeWordBits))

                # Increment the subcarrier and OFDM symbol index
                k += 1                                                  # Proceed to next subcarrier
                if k % self.NumSubcarriers == 0:                        # Proceed to next OFDM symbol
                    l += 1 

                bFirstPass = False

            assert NumCodeWordBits == CodeWordSize, 'A mistake occurred while mapping payload A data into the resource grid.'
            StartSubcarrierIndex = k
            StartOfdmSymbolIndex = l


        # Set the flags indicating what has been done
        self.PayloadAWasAdded        = True
        self.PayloadBWasAdded        = False
        self.PacketWasBuild          = False























    # ---------------------------------------------------------------------------------------------------------------- #
    #                                                                                                                  #
    #                                                                                                                  #
    # > Function: OfdmModulate() - This function transforms the resource grid into the time domain OFDM symbols        #
    #                                                                                                                  #
    #                                                                                                                  #
    # ---------------------------------------------------------------------------------------------------------------- #
    def OfdmModulate(self
                   , AntennaPort: int ) -> np.ndarray:
        '''
        This function OFDM modulates the QAM values in the resource grid
        '''
        assert isinstance(AntennaPort, int),         'The AntennaPort input argument is of invalid type' 
        assert AntennaPort == 0 or AntennaPort == 1, 'The AntennaPort input argument is invalid'
        if AntennaPort == 1 and self.ControlInformation.NumberTxAntennas == 1:
            assert False, 'Attemping to render non-existing antenna port 1 signal'

        ResourceGrid = self.ResourceGrid[:, :, AntennaPort]

        NumberSubcarriers = ResourceGrid.shape[0]
        NumberOfdmSymbols = ResourceGrid.shape[1]

        # Error Checking
        assert NumberSubcarriers == self.NumSubcarriers, 'The resource grid is of invalid size'

        # Determine the length of the OFDM modulated information and construct output array
        OfdmSymbolLength = self.IfftSize + self.CP_Length
        NumberOfSamples  = NumberOfdmSymbols * OfdmSymbolLength

        # Start the IFFT loop
        OfdmTxOutput        = np.zeros(NumberOfSamples, np.complex64)
        IfftInputBuffer     = np.zeros(self.IfftSize, np.complex64)
        OutputSequenceIndex = 0
        for l in range(0, NumberOfdmSymbols):                       # l is the OFDM symbol index
            OfdmSymbol = np.zeros(OfdmSymbolLength, np.complex64)

            # Map the information from the resource grid into the Ifft input buffer
            IfftInputBuffer[self.PosIfftIndices] = ResourceGrid[self.PosSubcarrierIndices, l]
            IfftInputBuffer[self.NegIfftIndices] = ResourceGrid[self.NegSubcarrierIndices, l]
            IfftOutputBuffer                     = np.sqrt(self.IfftSize) * np.fft.ifft(IfftInputBuffer)
            
            # plt.figure(1)
            # plt.subplot(2,1,1)
            # plt.plot(IfftInputBuffer.real, 'r', IfftInputBuffer.imag, 'b')
            # plt.title('Ifft Input Buffer')
            # plt.grid('#cccccc')
            # plt.tight_layout()


            # Fetch the Cyclic Prefix from the end of the IFFT output buffer
            CyclicPrefix                         = IfftOutputBuffer[-self.CP_Length:]
            
            # Place the Cyclic Prefix at the start of the OFDM symbol
            OfdmSymbol[0:self.CP_Length]         = CyclicPrefix
            
            # Place the Ifft Output at the end of the Ofdm Symbol
            OfdmSymbol[self.CP_Length:]          = IfftOutputBuffer

            # plt.subplot(2,1,2)
            # plt.plot(OfdmSymbol.real, 'r', OfdmSymbol.imag, 'b')
            # plt.title('OfdmSymbol')
            # plt.grid('#cccccc')
            # plt.tight_layout()
            # plt.show()
            
            # Copy the Ofdm Symbol into the OfdmTxOutput array
            Scaling                              = np.sqrt(self.IfftSize / self.NumSubcarriers)
            OfdmTxOutput[OutputSequenceIndex:OutputSequenceIndex+OfdmSymbolLength] = Scaling * OfdmSymbol

            print(np.var(Scaling * IfftOutputBuffer))

            # Increment the OutputSequenceIndex
            OutputSequenceIndex += OfdmSymbolLength

        return OfdmTxOutput
















    # -------------------------------------------------------------------------------------------------------- #
    #                                                                                                          #
    #                                                                                                          #
    # > Function: BuildTxPacket                                                                                #
    #                                                                                                          #
    #                                                                                                          #
    # -------------------------------------------------------------------------------------------------------- # 
    def BuildTxPacket(self
                    , strPreambleALength: str  ) -> np.ndarray:
        '''
        In this function assumes that the resource grid has already been completely populated and is ready to be transmitted.
        It will generate the preambles. (The strPreambleALength input argument will determine the length of PreambleA)
        It will OFDM modulate the resource grid.
        It will connect concatenate all sequence into a single TX signal
        '''
        # ----------------------------------------------
        # Type and Error Checking
        # ----------------------------------------------
        assert self.ResourceGridInitialized == True, 'At the very least, the resource grid must be initialized'
        assert isinstance(strPreambleALength, str),  'The strPreambleALength argument must be a string'
        assert strPreambleALength.lower() == 'short' or strPreambleALength.lower() == 'long' 

        # ----------------------------------------------
        # Build the preamble
        # ----------------------------------------------
        # Build the AGC Burst
        AgcBurst = Preamble.GenerateAgcBurst(self.SampleRate)
        SampleLengthAgcBurst = len(AgcBurst)

        # Build the PreambleA
        PreambleA = Preamble.GeneratePreambleA(self.SampleRate, strPreambleALength)
        SampleLengthPreambleA = len(PreambleA)

        # Build the PreambleB
        PreambleB = Preamble.GeneratePreambleB(self.SampleRate)
        SampleLengthPreambleB = len(PreambleB)
        
        PreambleLength        = SampleLengthAgcBurst + SampleLengthPreambleA + SampleLengthPreambleB

        # Ofdm Modulate the Resource grid from Port 0
        OfdmTxOutputP0 = self.OfdmModulate(0)

        # Concatenate all portions of the Tx Waveform for Port 0
        TxOutputP0     = np.hstack((AgcBurst.astype(np.complex64), PreambleA.astype(np.complex64), PreambleB.astype(np.complex64), OfdmTxOutputP0.astype(np.complex64)))
        TxOutputP1     = np.zeros(len(TxOutputP0), np.complex64)
        
        # Render the waveform for Port 1 if required
        if self.ControlInformation.NumberTxAntennas == 2:
            OfdmTxOutputP1 = self.OfdmModulate(1)
            assert len(OfdmTxOutputP0) == len(OfdmTxOutputP1), 'The OFDM modulated signals for P0 and P1 must be the same'
            TxOutputP1[PreambleLength:] = OfdmTxOutputP1 

        # Therefore the output is a matrix with two rows, where the first tows represents the Tx Waveform for P0, whereas the second row
        # represents the Tx Waveform for P1
        TxOutput = np.vstack([TxOutputP0, TxOutputP1])

        return TxOutput










    # --------------------------------------------------------------------------------------------------------------------------------------- #
    #                                                                                                                                         #
    # > Plot the resource grid                                                                                                                #
    #                                                                                                                                         #                                                                                                    
    # --------------------------------------------------------------------------------------------------------------------------------------- #
    def PlotResourceGrid(self
                       , AntennaPort: int ):
        '''
        This function will great a PCOLOR plot of the resource
        '''
        assert isinstance(AntennaPort, int),         "The input argument 'AntennaPort' is of invalid type"
        assert AntennaPort == 0 or AntennaPort == 1, "The input argument 'AntennaPort' is invalid" 

        assert isinstance(self.ResourceGridEnum[:,:, AntennaPort], np.ndarray)
        NumberSubcarriers, NumberOfdmSymbols = self.ResourceGridEnum[:,:,AntennaPort].shape
        
        # Limit the number of OFDM symbols to 25. Otherwise the plot gets to large
        if NumberOfdmSymbols > 25:
            NumberOfdmSymbols = 25

        # Please check out how cmap="tab20c" is defined at the following location:
        # https://matplotlib.org/stable/users/explain/colors/colormaps.html
        # Our goal here is to recreate a pcolor plot that matches the colors of the resource elements
        # used in the plots of the FlexLink specification.
        Z               = np.zeros([NumberSubcarriers, NumberOfdmSymbols], np.float32)
        for k in range(0, NumberSubcarriers):
            for l in range(0, NumberOfdmSymbols):
                Value = self.ResourceGridEnum[k, l, AntennaPort]

                if Value == -1:  # Unknown --> Yellow
                    Z[k,l] = 0.42

                if Value == 0:   # Empty --> White / Light Grey
                    Z[k,l] = 1 

                if Value == 1:   # RefSignalPort0 --> Grey
                    Z[k,l] = 0.90

                if Value == 2:   # RefSignalPort1 --> Red
                    Z[k,l] = 0.28

                if Value == 3:   # Control information --> Blue
                    Z[k,l] = 0.08 

                if Value == 4:   # Data --> Yellow
                    Z[k,l] = 0.42

                if Value == 5:   # Signal Field --> Green
                    Z[k,l] = 0.5

                if Value == 6:   # PayloadA Even Code Word
                    Z[k,l] = 0.65

                if Value == 7:   # PayloadA Odd  Code Word --> Blue
                    Z[k,l] = 0.69

                # Here I have to establish the min and max values of the CMAP. These two resource elements do not
                # represent any content.
                if k == NumberSubcarriers-1 and l == NumberOfdmSymbols - 1:
                    Z[k,l] = 1.0    # This is the maximum color value
                
                if k == NumberSubcarriers - 2 and l == NumberOfdmSymbols - 1:
                    Z[k,l] = 0.08   # This is the minimum color value

        X, Y            = np.mgrid[0:NumberSubcarriers+1:1, 0:NumberOfdmSymbols+1:1]

        fig = plt.figure(1) 
        plt.pcolor(Y, X, Z, cmap="tab20c", )
        plt.title('Resource Grid\n\n(Blue - Control / Dark Grey - Ref Signal P0 / Red - Ref Signal P1 / Dark Purple - PayloadA Even CodeWord)\n( Light Purple - PayloadA Odd CodeWord / Orange - Available for Data / Light Gray - Empty = 0+0j)')
        plt.ylabel('Subcarriers') 
        plt.xlabel('Ofdm Symbols') 
        plt.grid(color='#999999') 
        plt.show()















    


# ----------------------------------------------------------------------
# > Testbench
# ----------------------------------------------------------------------
if __name__ == '__main__':
    
    Test = 0

    if Test == 0:

        # ----------------------------------------------------------------------------------------------
        # > Construct the CFlexTransmitter object
        # ---------------------------------------------------------------------------------------------- 

        bLteBw        = True                           # Indicate the bandwidth of the FlexLink signal
                                                       # For LTE  -> 913 Subcarriers * SubcarrierSpacing = 17.832MHz
                                                       # For WLAN -> 841 Subcarriers * SubcarrierSpacing = 16.426MHz
        bOverSampling = False                          # Whereas the main processing in both transmitter and receiver happens
                                                       # at 20MHz, the DACs and ADC can not run at 20MHz.
                                                       # If bOverSampling == False, then the waveform is produced at 20MHz (1024 length IFFT)  
                                                       # and additional polyphase upsampling filter is required.
                                                       # If bOverSampling == True, then a 2048 IFFT is used to produce the OFDM
                                                       # signal and the output waveform is properly sampled to be read out at 40MHz 
        
                                                       # We use either a 1024 (bOverSampling == False) or 2048 (bOverSampling == True) size IFFT
                                           
        FlexLinkTransmitter = CFlexLinkTransmitter(bLteBw, bOverSampling)

        # -----------------------------------------------------------------------------------------
        # > Construct the control information object
        # -----------------------------------------------------------------------------------------
        NumOfdmSymbols = 20
        ControlInfo = CControlInformation(ReferenceSymbolPeriodicityIndex= 3      # 0/1/2/3 - [1, 3,  6, 12]
                                        , ReferenceSignalSpacingIndex    = 2      # 0/1/2/3 - [3, 6, 12, 24]
                                        , NumSignalFieldSymbolsIndex     = 1      # 0/1/2/3 - [1, 2,  4, 10]
                                        , SignalFieldModulationFlag      = 1      # 0/1     - BPSK / QPSK
                                        , NumberTxAntennaPortFlag        = 1      # 0/1     - 1TX  / 2TX
                                        , DcSubcarrierFlag               = 0)     # 0/1     - 1/13 Subcarriers
        
        print(ControlInfo)

        # -----------------------------------------------------------------------------------------
        # > Create and initialize the resource grid with control information and reference signals
        # -----------------------------------------------------------------------------------------
        # At this point, the resource grid is built and all control and reference signals are placed within
        ControlInfo = FlexLinkTransmitter.InitializeResourceGrid(ControlInfo, NumOfdmSymbols)


        # -----------------------------------------------------------------------------------------
        # > Define the Signal Field and add it to the resource grid
        # -----------------------------------------------------------------------------------------
        NumDataBlocks = 20
        SignalFieldFormat1 = CSignalField(FEC_Mode   = 1                            # 0/1 - LDPC Coding / Polar Coding
                                        , CBS_A_Flag = 1                            # 0/1/2/3 -> 648/1296/1944/1944 (LDPC) or 256/512/1024/2048 (Polar Coding)
                                        , FEC_A_Flag = 2                            # 0/1/2/3/4/5/6/7 -> 1/2,2/3,3/4,5/6,1/2,1/2,1/2 (LDPC) or 1/4,3/8,1/2,5/8,3/4,7/8,1/4,1/4 (Polar)
                                        , NDB_A      = NumDataBlocks                # 0 to 10**14 - 1 or 16383  The number of data blocks sent in payload A
                                        , RM_A_Flag  = 0                            # 0/1/2/3/4/5/6/7 -> RateMatchingFactor = 0, 0.5, 0.75, 1, 3, 7, 15, 31  Number rate matched bits = CBS * (1 + RateMatchingFactor)
                                        , BPS_A_Flag = 1                            # 0/1/2/3 -> BPSK, QPSK, 16QAM, 64QAM
                                        , NumOfdmSymbols        = NumOfdmSymbols  
                                        , TxReferenceClockCount = 0                 # 0 to 10**14 - 1 or 16383  The number of data blocks sent in payload A
                                        , Client_Flag           = 0)                # 0/1 -> Point-to-Point / Point-to-Multipoint
        
        print(SignalFieldFormat1)

        # Add the signal field
        FlexLinkTransmitter.AddSignalField(SignalFieldFormat1)

        #FlexLinkTransmitter.PlotResourceGrid(0)
        #FlexLinkTransmitter.PlotResourceGrid(1)

        # -----------------------------------------------------------------------------------------
        # > Define the list of data blocks to be transmitted
        # -----------------------------------------------------------------------------------------
        # Let's determine the DataBlockSize
        CodeBlockSize = SignalFieldFormat1.CodeBlockSizeA
        DataWordSize  = int(CodeBlockSize * SignalFieldFormat1.CodingRate)
        DataBlockSize = DataWordSize - 16                              # 16 CRC bits

        DataBlockList = []
        for DataBlockIndex in range(0, NumDataBlocks):
            CurrentDataBlock = np.random.randint(low=0, high=2, size= DataBlockSize, dtype = np.uint8)
            DataBlockList.append(CurrentDataBlock)

        FlexLinkTransmitter.AddPayloadA(DataBlockList)

        #FlexLinkTransmitter.PlotResourceGrid(0)
        #FlexLinkTransmitter.PlotResourceGrid(1)

        # ------------------------------------------------------------------------------------------
        # > Build the packet
        # ------------------------------------------------------------------------------------------
        strPreambleALength = 'short'   # Either 'short' = 50 useconds, or 'long' = 250 useconds
        Output = FlexLinkTransmitter.BuildTxPacket(strPreambleALength)


        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(Output[0,:].real, 'r', Output[0, :].imag, 'b')
        plt.grid('#cccccc')
        plt.title('IQ Waveform of P0 signal')
        plt.tight_layout()
        plt.subplot(2,1,2)
        plt.plot(Output[1,:].real, 'r', Output[1, :].imag, 'b')
        plt.grid('#cccccc')
        plt.title('IQ Waveform of P1 signal')
        plt.tight_layout()
        plt.show()


    # -----------------------------------------------------------------------------------------
    # > Overwrite the information in the first reference signal with all ones to test the OFDM modulator
    # -----------------------------------------------------------------------------------------
    #Copy = FlexLinkTransmitter.ResourceGrid[:,0].copy()
    #FlexLinkTransmitter.ResourceGrid[:, 0] = (1+1j)*np.ones(Copy.shape, np.complex64)

    #Output = FlexLinkTransmitter.OfdmModulate()
    #stop = 1

 