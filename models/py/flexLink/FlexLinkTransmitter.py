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
__version__   = "1.0.0"
__date__      = "April, 20th, 2024"
__copyright__ = 'Andreas Schwarzinger'

# --------------------------------------------------------
# Import Statements
# --------------------------------------------------------
from   FlexLinkParameters import *
from   FlexLinkPhyBase    import CFlexLinkPhyBase
from   FlexLinkCoder      import CCrcProcessor, CLdpcProcessor, CPolarProcessor, InterleaverFlexLink, CLinearFeedbackShiftRegister
from   QamMapping         import CQamMappingIEEE
import Preamble
import numpy              as np
import matplotlib.pyplot  as plt

import os
import sys                               # We use sys to include new search paths of modules
OriginalWorkingDirectory = os.getcwd()   # Get the current directory
DirectoryOfThisFile      = os.path.dirname(__file__)   # FileName = os.path.basename
if DirectoryOfThisFile != '':
    os.chdir(DirectoryOfThisFile)        # Restore the current directory

# There are modules that we will want to access and they sit two directories above DirectoryOfThisFile
sys.path.append(DirectoryOfThisFile + "\\..\\..\\DspComm")
sys.path.append(DirectoryOfThisFile + "\\..\\..\\..\\python\\KoliberEng")

import ModulateDemodulate as md 
import Visualization  as vis
# md = ModulateDemodulate()
# vis = visualization.Visualization()


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
class CFlexLinkTransmitter(CFlexLinkPhyBase):
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
        # Calling Constructor of base class
        # There are a lot of variables that are shared between the Transmitter and Receiver in the base class
        super().__init__(bLteBw, bOverSampling)
        
        # Error checking
        assert isinstance(bLteBw, bool),         'The input argument bLteBw must be a boolean value'
        assert isinstance(bOverSampling, bool),  'The input argument bOverSampling must be a boolean value'  
           
        # Uninitialized parameters 
        # --> self.ResourceGrid is self explanatory
        # --> self.ResourceGridEnum holds an integer representing the type of information present in the resource element (It is in the base class)
        # self.ControlInformation   (See the base class)
        # self.SignalField          (See the base class)
        self.ResourceGrid       = None         # This will be initialized to a complex valued numpy matrix in the InitializeResourceGrid()
        
        # Construct some of the required objects for the transmitter
        self.PolarProcessor     = CPolarProcessor()
        self.LdpcProcessor      = None

        # Debugging Quantities for the Signal Field
        self.EncodedBitsOfSignalField      = None # Signal Field Bits after FEC encoding
        self.InterleavedBitsOfSignalField  = None # Signal Field Bits after Interleaving
        self.RateMatchedBitsOfSignalField  = None # Signal Field Bits after Rate matching
        self.ScrambledBitsOfSignalField    = None # Signal Field Bits after Scrambling
        self.ReSequenceOfSignalFieldP0     = None # The resource element sequence to be mapped into the resource grid for port0
        self.ReSequenceOfSignalFieldP1     = None # The resource element sequence to be mapped into the resource grid for port1



        # Boolean Flags
        self.ResourceGridInitialized = False
        self.SignalFieldWasAdded     = False
        self.PayloadAWasAdded        = False
        self.PayloadBWasAdded        = False
        self.PacketWasBuild          = False



    # --------------------------------------------------------------------------------------------------------------------------------------- #
    #                                                                                                                                         #
    #                                                                                                                                         #
    # > Function: InitializeResourceGrid()                                                                                                    #
    #                                                                                                                                         #
    #                                                                                                                                         #
    # --------------------------------------------------------------------------------------------------------------------------------------- #
    def InitializeResourceGrid(self
                             , ControlInfo: CControlInformation
                             , SignalField: CSignalField):
        '''
        This function will populate the first symbol the resource grid based with control information and reference signals
        '''
        # Error checking
        assert isinstance(ControlInfo, CControlInformation), 'The ControlInfo input argument is of invalid type'
        assert isinstance(SignalField, CSignalField),        'Only use integers for the NumberOfdmSymbols argument'

        NumberOfdmSymbols = SignalField.NumOfdmSymbols
        assert NumberOfdmSymbols > 3,                        'The resource grid must have at least 3 OFDM symbols - 1 Ref Symbol, 1 Signal Field Symbol, 1 PayloadA symbol'

        # -----------------------------------
        # > Call the intializers in the base class. They configure the self.ResourceGridEnum, which is part of the base class
        # -----------------------------------
        self.PreInitReferenceGrid(ControlInfo)
        self.InitReferenceGridEnum(SignalField)

        # ----------------------------------
        # > The resource grid is a matrix of complex values that will be read out to the IFFT
        # ----------------------------------
        self.ResourceGrid       = np.zeros(self.ResourceGridEnum.shape, np.complex64)  
        
   
        # ----------------------------------
        # > Place control information into the first reference symbol
        # ----------------------------------
        ControlBitCount  = 0    # There are some 300 ControlBits that are placed
        for k in range(0, self.NumSubcarriers):        # k is the subcarrier index

            # Can we place a control element at this subcarrier???
            if self.ResourceGridEnum[k, 0, CFlexLinkTransmitter.AntPort0] == EReType.Control.value:
                ControlBitIndex     = ControlBitCount % CControlInformation.NumberOfControlBits
                CurrentControlBit   = ControlInfo.ControlInformationArray[ControlBitIndex]
                ScramblingBit       = self.ScrambingSequence[ControlBitCount % len(self.ScrambingSequence)]
                ScrambledControlBit = (1 + CurrentControlBit + ScramblingBit) % 2 

                # Convert the scrambled control bit into a BPSK symbol
                QamSymbol           = np.complex64(2*int(ScrambledControlBit) - 1)

                # Place the QamSymbol - Only at Antenna port 0 as control information is only places there.
                self.ResourceGrid[k, 0, CFlexLinkTransmitter.AntPort0]    = QamSymbol

                # Increment the control bit count
                ControlBitCount    += 1

        # ----------------------------------
        # > Place the reference signals of antanna port 0 into the all reference symbols
        # ----------------------------------
        if ControlInfo.NumberTxAntennas == 1:
            Scaling = np.complex64(1)                   # BPSK modulation 0/1 -> -1 + j0 / 1 + j0 (No other scaling)
        else:  
            Scaling = np.complex64(1.414)               # Reference signals are BPSK modulated with 3dB boost 
                                                        # Thus bits 0/1 map to -1.414 + j0 / 1.414 + j0

        for l in range(0, self.SignalField.NumOfdmSymbols, self.ControlInformation.ReferenceSymbolPeriodicity):

            RefSignalP0Count = 0                            # There are close to 300 RefSignalP0 values placed
            for k in range(0, self.NumSubcarriers):         # k is the subcarrier index
                
                # Can we place a reference signal P0 at this subcarrier???
                if self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort0] == EReType.RefSignalPort0.value:
                    ScramblingRefBit                                       = self.ScrambingSequence[RefSignalP0Count % len(self.ScrambingSequence)]
                    QamSymbol                                              = np.complex64(2*int(ScramblingRefBit) - 1) # Convert the scrambled control bit into a BPSK symbol
                    self.ResourceGrid[k, l, CFlexLinkTransmitter.AntPort0] = QamSymbol*Scaling                         # Place the QamSymbol               
                    RefSignalP0Count                                      += 1                                         # Increment the reference signal P0 bit count


        # ----------------------------------
        # > Place the reference signals of antanna port 1 into all reference symbols
        # ----------------------------------
        if ControlInfo.NumberTxAntennas == 1:
            Scaling = np.complex64(1)                   # BPSK modulation 0/1 -> -1 + j0 / 1 + j0 (No other scaling)
        else:  
            Scaling = np.complex64(1.414)               # Reference signals are BPSK modulated with 3dB boost 
                                                        # Thus bits 0/1 map to -1.414 + j0 / 1.414 + j0

        for l in range(0, self.SignalField.NumOfdmSymbols, self.ControlInformation.ReferenceSymbolPeriodicity):

            RefSignalP1Count = 0                            # There are close to 300 RefSignalP0 values placed
            for k in range(0, self.NumSubcarriers):         # k is the subcarrier index
                
                # Can we place a reference signal P0 at this subcarrier???
                if self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort1] == EReType.RefSignalPort1.value:
                    ScramblingRefBit                                       = self.ScrambingSequence[RefSignalP1Count % len(self.ScrambingSequence)]
                    QamSymbol                                              = np.complex64(2*int(ScramblingRefBit) - 1)     # Convert the scrambled control bit into a BPSK symbol
                    self.ResourceGrid[k, l, CFlexLinkTransmitter.AntPort1] = QamSymbol*Scaling                             # Place the QamSymbol               
                    RefSignalP1Count                                      += 1                                             # Increment the reference signal P0 bit count


        # ----------------------------------
        # > Place the DC Signals in the correct area of the resource grid
        # ----------------------------------
        Half    = int(ControlInfo.NumberDcSubcarriers / 2)
        DcRange = range(self.CenterSubcarrier - Half, self.CenterSubcarrier + Half + 1)

        for l in range(0, NumberOfdmSymbols):               # l is the OFDM symbol index     
            for k in DcRange:                               # k is the subcarrier index
                if self.ResourceGridEnum [k, l, CFlexLinkTransmitter.AntPort0] == EReType.Empty.value:       # Nothing has yet been placed here
                    self.ResourceGrid    [k, l, CFlexLinkTransmitter.AntPort0] = np.complex64(0)

                if ControlInfo.NumberTxAntennas == 2:
                    if self.ResourceGridEnum [k, l, CFlexLinkTransmitter.AntPort1] == EReType.Empty.value:       # Nothing has yet been placed here
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

        SourceBits                    = np.zeros(N, np.uint16)
        SourceBits[MessageBitIndices] = self.SignalField.SignalFieldArray.copy()
        self.EncodedBitsOfSignalField = self.PolarProcessor.RunPolarEncoder(SourceBits)

        # ----------------------------------------------------------------------------
        # Step 2: Interleave the encoded information (simulation will have to show how useful this step actually is.)
        # ----------------------------------------------------------------------------
        FEC_Mode = 1   # Polar
        if SignalFieldFormat == 1:
            CBS_A_Flag = 0 # Indicates 256 encoded bits
        else:
            CBS_A_Flag = 2 # Indicates 1024 encoded bits

        InterleavingIndices = InterleaverFlexLink(FEC_Mode, CBS_A_Flag)

        self.InterleavedBitsOfSignalField                      = np.zeros(N, np.uint16)
        self.InterleavedBitsOfSignalField[InterleavingIndices] = self.EncodedBitsOfSignalField 

        # ----------------------------------------------------------------------------
        # Step 3: Rate Matching / Repetition
        # ----------------------------------------------------------------------------
        # First we need to find out how many resource elements are available within the number of OFDM
        # symbols that we have available for the signal field
        ReCount           = 0

        for l in range(1, NumberSignalFieldSymbols + 1):  # Remember, symbol l = 0 is reserved for the control information
            for k in range(0, self.NumSubcarriers):
                if self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort0] == EReType.SignalField.value:          

                    # If there are two antennas, then ensure that the Resource element type in the 
                    # AntPort1 resource grid matches what is assigned in the AntPort0 resource grid.
                    # Reassign the resource element type to SignalField
                    if self.ControlInformation.NumberTxAntennas == 2:
                        assert self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort0] == self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort1]
                                              
                    ReCount += 1
                
                

      
        # Compute the number of available bits 
        NumAvailableSignalFieldBits = ReCount * NumBitsPerQamValue

        # Create the rate matched bit array
        self.RateMatchedBitsOfSignalField = np.zeros(NumAvailableSignalFieldBits, np.uint16)                    
        for BitIndex in range(0, NumAvailableSignalFieldBits):
            self.RateMatchedBitsOfSignalField [BitIndex] = self.InterleavedBitsOfSignalField[BitIndex % N]


        # ------------------------------------------------------------------------
        # Step 4: Scramble the Rate Matched bits
        # ------------------------------------------------------------------------
        self.ScrambledBitsOfSignalField = np.zeros(NumAvailableSignalFieldBits, np.uint16)       
        for BitIndex in range(0, NumAvailableSignalFieldBits):
            self.ScrambledBitsOfSignalField[BitIndex] = (self.RateMatchedBitsOfSignalField[BitIndex]  + self.ScrambingSequence[BitIndex % len(self.ScrambingSequence)]) % 2



        # ------------------------------------------------------------------------
        # Step 5: Convert the Rate Matched Bit stream into BPSK or QPSK values and map them into the Resource Grid
        # ------------------------------------------------------------------------
        if self.ControlInformation.NumberTxAntennas == 1:
            self.ReSequenceOfSignalFieldP0     = np.zeros(int(len(self.ScrambledBitsOfSignalField) / NumBitsPerQamValue), np.complex64)
            StartBit = 0
            QamCount = 0
            for l in range(1, NumberSignalFieldSymbols + 1):  # Remember, symbol l = 0 is reserved for the control information
                for k in range(0, self.NumSubcarriers):

                    if self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort0] == EReType.SignalField.value:
                        assert StartBit <= len(self.ScrambledBitsOfSignalField) - NumBitsPerQamValue
                        self.ReSequenceOfSignalFieldP0[QamCount]              = CQamMappingIEEE.Mapping(NumBitsPerQamValue, self.ScrambledBitsOfSignalField[StartBit:StartBit+NumBitsPerQamValue])[0]
                        self.ResourceGrid[k,l, CFlexLinkTransmitter.AntPort0] = self.ReSequenceOfSignalFieldP0[QamCount]    
                        StartBit += NumBitsPerQamValue
                        QamCount += 1

            # Ensure that we have gone through the entire RateMatchedBits array. If not, there is a mapping error
            assert StartBit == len(self.ScrambledBitsOfSignalField), 'A mapping error has occured.'  
            assert QamCount == len(self.ReSequenceOfSignalFieldP0),  'A mapping error has occured.'

        else:   
            # We have two antenna ports and we must use SFBC during the mapping process

            # -------------------- Debugging Arrays
            self.ReSequenceOfSignalFieldP0     = np.zeros(int(len(self.ScrambledBitsOfSignalField) / NumBitsPerQamValue), np.complex64)
            self.ReSequenceOfSignalFieldP1     = np.zeros(int(len(self.ScrambledBitsOfSignalField) / NumBitsPerQamValue), np.complex64)
            QamCount                           = 0    # only exists for the debugging arrays 
            # --------------------

            StartBit                    = 0
            SignalFieldOpportunityCount = 0
            CurrentReLocation           = [0, 1]
            LastReLocation              = [0, 1]
            for l in range(1, NumberSignalFieldSymbols + 1):  # Remember, symbol l = 0 is reserved for the control information
                for k in range(0, self.NumSubcarriers):
                    CurrentReLocation           = [k, l]

                    # Only take action if the current resource element is reserved for the signal field
                    if self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort0] == EReType.SignalField.value:
                        # Ensure that the resource element in the P1 resource grid is also reserved for the signal field                          
                        assert self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort1] == EReType.SignalField.value, \
                                                'For the two antenna case, the ReType in both resource grids must match'
                        
                        SignalFieldOpportunityCount       += 1

                        # Every two signal field opportunities we will want to map the SFCB encoded information
                        if SignalFieldOpportunityCount % 2 == 0:
                            assert StartBit <= len(self.ScrambledBitsOfSignalField) - 2 * NumBitsPerQamValue
                            X0  = CQamMappingIEEE.Mapping(NumBitsPerQamValue, self.ScrambledBitsOfSignalField[StartBit:StartBit+NumBitsPerQamValue])
                            StartBit += NumBitsPerQamValue
                            X1  = CQamMappingIEEE.Mapping(NumBitsPerQamValue, self.ScrambledBitsOfSignalField[StartBit:StartBit+NumBitsPerQamValue])
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

                            # --------- Populate Debugging Arrays
                            self.ReSequenceOfSignalFieldP0[QamCount]  =  Port0First[0]
                            self.ReSequenceOfSignalFieldP1[QamCount]  =  Port1First[0]
                            QamCount += 1
                            self.ReSequenceOfSignalFieldP0[QamCount]  =  Port0Second[0]
                            self.ReSequenceOfSignalFieldP1[QamCount]  =  Port1Second[0] 
                            QamCount += 1
                            # ----------------------------------------

                        LastReLocation = CurrentReLocation

            # Ensure that we have gone through the entire RateMatchedBits array. If not, there is a mapping error
            assert StartBit == len(self.ScrambledBitsOfSignalField), 'A mapping error has occured.'  
            assert QamCount == len(self.ReSequenceOfSignalFieldP0),  'A mapping error has occured.'




        # --------------------------
        # Create the self.ResourceBlockEnum matrix, which we synthezise from the self.ResourceGridEnum matrix 
        # --------------------------
        self.ResourceBlockEnum = EReType.Unassigned.value * np.ones([self.NumResourceBlocks, self.SignalField.NumOfdmSymbols], np.int16)
        
        # Itererate through the resource blocks
        for l in range(0, self.SignalField.NumOfdmSymbols):
            # Iterate through the subcarriers
            for k in range(0, self.NumSubcarriers):
                # Determine in which resource block we are currently located
                kVirtual = k
                if kVirtual > self.CenterSubcarrier:
                    kVirtual = k -1
                CurrentResourceBlockIndex = int(kVirtual / 12)

                # Check each resource element and determine the resource block type. We ignore resource elements that hold
                # reference signals and DC resource elements
                if self.ResourceGridEnum[k, l, 0] == EReType.PayloadAEvenCodeWord.value:
                    self.ResourceBlockEnum[CurrentResourceBlockIndex, l] = EReType.PayloadAEvenCodeWord.value
                if self.ResourceGridEnum[k, l, 0] == EReType.PayloadAOddCodeWord.value:
                    self.ResourceBlockEnum[CurrentResourceBlockIndex, l] = EReType.PayloadAOddCodeWord.value
                if self.ResourceGridEnum[k, l, 0] == EReType.Control.value:
                    self.ResourceBlockEnum[CurrentResourceBlockIndex, l] = EReType.Control.value
                if self.ResourceGridEnum[k, l, 0] == EReType.SignalField.value:
                    self.ResourceBlockEnum[CurrentResourceBlockIndex, l] = EReType.SignalField.value                 
                if self.ResourceGridEnum[k, l, 0] == EReType.UnassignedDataP0 .value:
                    self.ResourceBlockEnum[CurrentResourceBlockIndex, l] = EReType.UnassignedDataP0.value 




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
        assert len(DataBlockList) == self.SignalField.NumberDataBlocksA,    'Unexpected number of data blocks in DataBlockList'
        for ListIndex in range(0, len(DataBlockList)):
            assert isinstance(DataBlockList[ListIndex], np.ndarray),        'Each data block must be a numpy array'
            assert np.issubdtype(DataBlockList[ListIndex].dtype, np.uint16), 'Each data block must a numpy array of type np.uint16'
            
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
            self.LdpcProcessor = CLdpcProcessor('WLAN', None, CodeBlockSize, strEncodingRate)
            N                     = CodeBlockSize
        else:                                        # Configure the Polar encoder
            p                     = 0.55             # The base erasure probability
            N                     = CodeBlockSize    # The number of final encoded bits (code block size)
            K                     = DataWordSize     # The number of bits we want to encode
            ErasureProbabilities  = self.PolarProcessor.FindChannelProbabilities(CodeBlockSize, p)
            SortedIndices         = np.argsort(ErasureProbabilities)
            MessageBitIndices     = SortedIndices[0: K]


        # Determine where we will start to map PayloadA information into the ResourceGrid. We want to declare
        # these variable before the for range loops that follows below, as these variables are updated within the loops.
        StartOfdmSymbolIndex    = self.ControlInformation.NumberSignalFieldSymbols + 1    
        StartResourceBlockIndex = 0    
        StartSubcarrierIndex    = 0
                             

        # To make our life easier let's create an array of 76 entries representing the number of bits
        # per QAM symbol for each resource block. This will facilitate the accounting of available resource
        # when we deteremine the number of bits available in each resource block
        if self.SignalField.SignalFieldFormat == 1:
            BpsEachResourceBlock = self.SignalField.BitsPerQamSymbol * np.ones(76, np.uint16)
        else:
            BpsEachResourceBlock = self.SignalField.BitsPerQamSymbol   # Which must be a vector this time


        # Compute the CRC, Encode, interleave, rate match and map each data block into the resource grid
        for ListIndex in range(0, len(DataBlockList)):
            DataBlock = DataBlockList[ListIndex]

            # -----------------------------------------------------------------------
            # > 1. Compute the CRC and form the data word
            # -----------------------------------------------------------------------
            DataWord                  = np.zeros(DataBlockSize + 16, np.uint16)
            DataWord[0:DataBlockSize] = DataBlock

            CrcOutput = CCrcProcessor.ComputeCrc(CCrcProcessor.Generator16_LTE, DataBlock.tolist())

            # The CRC Bits are inserted after the data block bits
            for BitIndex in range(DataBlockSize, DataBlockSize + 16):
                DataWord[BitIndex] = CrcOutput[BitIndex - DataBlockSize]


            # -----------------------------------------------------------------------
            # > 2. FEC encode the data word into an encoded data word
            # -----------------------------------------------------------------------
            if self.SignalField.FEC_Mode == 0:     # This is LDCP coding
                EncodedDataWord               = self.LdpcProcessor.EncodeBits(DataWord)
            else:
                SourceBits                    = np.zeros(N, np.uint16)
                SourceBits[MessageBitIndices] = DataWord 
                EncodedDataWord               = self.PolarProcessor.RunPolarEncoder(SourceBits)         

            assert len(EncodedDataWord) == CodeBlockSize, 'An error occured during the FEC encoding process'
                
                
            # -----------------------------------------------------------------------
            # > 3. Interleave the encoded data word to form the code block
            # -----------------------------------------------------------------------    
            InterleavingIndices            = InterleaverFlexLink(self.SignalField.FEC_Mode, self.SignalField.CBS_A_Flag)
            CodeBlock                      = np.zeros(N, np.uint16)
            CodeBlock[InterleavingIndices] = EncodedDataWord 


            # -----------------------------------------------------------------------
            # > 4. Determine the minimum size of the code word, which is the rate matched bit stream.
            # -----------------------------------------------------------------------    
            MinimumCodeWordSize            = int(CodeBlockSize * (1 + self.SignalField.RateMatchingFactor))

            # ----------------------------------------------------------------------
            # > 5. Determine the actual number of rate matched bits, which will be >= MinimumCodeWordSize. Remember, the MinimumCodeWordSize
            #      will likely not completely fill up the last resource block. We must find the number of code word bits that will in 
            #      fact fill the last resource block completely. To find the actual number of code word bits, we use self.CapacityTable_RefSymbol
            #      and self.CapacityTable_DataSymbol, which indicate the bit capacity for each resource blocks in an OFDM reference and data symbol
            #      given the BPS information provided in the SignalField. Remember, the SignalField only provides BPS information for payloadA.
            # ----------------------------------------------------------------------
            # We start at coordinates [k, l] in the resource grid and start counting bits
            NumCodeWordBits           = 0                        # Keep track of the number of bits available for PayloadA in the resource grid
            NumResourceBlocksUsed     = 0                        # Keep track of the number of resource blocks used
            CurrentResourceBlockIndex = StartResourceBlockIndex 
            CurrentOfdmSymbolIndex    = StartOfdmSymbolIndex

            while True:
                # Determine whether the current Ofdm Symbol (with index l) is an OFDM reference or data symbol
                bReferenceSymbol = CurrentOfdmSymbolIndex % self.ControlInformation.ReferenceSymbolPeriodicity == 0

                # Grab a reference to the capacity table that we need to use for this OFDM symbol with index l
                if bReferenceSymbol == True:
                    CurrentCapacityTable = self.CapacityTable_RefSymbol
                else:
                    CurrentCapacityTable = self.CapacityTable_DataSymbol
                
                NumCodeWordBits       += np.uint16(CurrentCapacityTable[CurrentResourceBlockIndex])
                NumResourceBlocksUsed += 1

                # Prepare indices for the next iteration through the while loop
                CurrentResourceBlockIndex += 1
                if CurrentResourceBlockIndex == self.NumResourceBlocks:   # Them move to the next OFDM symbol
                    CurrentResourceBlockIndex = 0
                    CurrentOfdmSymbolIndex   += 1

                # Exit the loop in case that the NumCodeWordBits that are located in the NumResourceBlocksUsed is larger
                # than the MinimumCodeWordSize. At that point we have found out how many resource blocks we actually need.
                if NumCodeWordBits >= MinimumCodeWordSize:
                    break

            #print('CodeWord: ' + str(ListIndex) + '  Resource blocks used: ' + str(NumResourceBlocksUsed) + "    CodeWordSize: " + str(NumCodeWordBits))

            # -----------------------------------------------------------------------
            # > 6. With the correct code word size, execute the rate matching operation
            # -----------------------------------------------------------------------
            CodeWordSize     = NumCodeWordBits
            CodeWord         = np.zeros(CodeWordSize, np.uint16)
            for BitIndex in range(0, CodeWordSize):
                CodeWord[BitIndex] = CodeBlock[BitIndex % CodeBlockSize]   # Rate matching / Repetition happens here


            # ------------------------------------------------------------------------
            # > 7. Scramble the Rate Matched bits
            # ------------------------------------------------------------------------
            for BitIndex in range(0, len(CodeWord)):
                CodeWord[BitIndex] = (CodeWord[BitIndex]  + self.ScrambingSequence[BitIndex % len(self.ScrambingSequence)]) % 2


            # -----------------------------------------------------------------------
            # > 8. Map the CodeWord 
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

                # Ensure that we are not extracting information at symbols that don't exist. Remember, that we had set number of symbols in the control
                # information object. Thus before we computed how many symbols we actually need.
                assert l < self.SignalField.NumOfdmSymbols, 'We are attempting to map data into a non-existent OFDM symbol'
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
                    self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort0]   = EReType.Empty.value
                    self.ResourceGrid    [k, l, CFlexLinkTransmitter.AntPort0]   = 0 + 0j
                    if self.ControlInformation.NumberTxAntennas == 2:
                        self.ResourceGridEnum[k, l, CFlexLinkTransmitter.AntPort1] = EReType.Empty.value
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
                            assert NumCodeWordBits <= CodeWordSize - 2 * BPS, 'NumCodeWordBit (' + str(NumCodeWordBits) + ') is larger than CodeWordSize - 2 * BPS (' + \
                                                                                                   str(CodeWordSize - 2 * BPS) + ')'
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
            StartSubcarrierIndex    = k
            StartOfdmSymbolIndex    = l
            StartResourceBlockIndex = CurrentResourceBlockIndex


        # Set the flags indicating what has been done
        self.PayloadAWasAdded        = True
        self.PayloadBWasAdded        = False
        self.PacketWasBuild          = False








    # -------------------------------------------------------------------------------------------------------- #
    #                                                                                                          #
    #                                                                                                          #
    # > Function: BuildTxPacket                                                                                #
    #                                                                                                          #
    #                                                                                                          #
    # -------------------------------------------------------------------------------------------------------- # 
    def BuildTxPacket(self
                    , strPreambleALength: str
                    , bPlot : bool             ) -> np.ndarray:
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
        # AgcBurst = Preamble.GenerateAgcBurstTest(self.SampleRate, True) # lwd working 

        SampleLengthAgcBurst = len(AgcBurst)

        # Build the PreambleA
        PreambleA = Preamble.GeneratePreambleA(self.SampleRate, 1024, strPreambleALength)
        SampleLengthPreambleA = len(PreambleA)
        
        if bPlot:
            Preamble.DetectPreambleA(PreambleA, 20.0e6, bPlot)
            Preamble.ProcessPreambleA(PreambleA, 20.0e6, bHighCinr= False, bShowPlots = True)
            

        # Build the PreambleB
        PreambleB, _ = Preamble.GeneratePreambleB(self.SampleRate)
        SampleLengthPreambleB = len(PreambleB)
        
        PreambleLength        = SampleLengthAgcBurst + SampleLengthPreambleA + SampleLengthPreambleB

        # Ofdm Modulate the Resource grid from Port 0
        OfdmTxOutputP0 = self.OfdmModulate(0, self.ResourceGrid)

        # Concatenate all portions of the Tx Waveform for Port 0
        TxOutputP0     = np.hstack((AgcBurst.astype(np.complex64), PreambleA.astype(np.complex64), PreambleB.astype(np.complex64), OfdmTxOutputP0.astype(np.complex64)))
        TxOutputP1     = np.zeros(len(TxOutputP0), np.complex64)
        
        # Render the waveform for Port 1 if required
        if self.ControlInformation.NumberTxAntennas == 2:
            OfdmTxOutputP1 = self.OfdmModulate(1, self.ResourceGrid)
            assert len(OfdmTxOutputP0) == len(OfdmTxOutputP1), 'The OFDM modulated signals for P0 and P1 must be the same'
            TxOutputP1[PreambleLength:] = OfdmTxOutputP1 

        # Therefore the output is a matrix with two rows, where the first tows represents the Tx Waveform for P0, whereas the second row
        # represents the Tx Waveform for P1
        TxOutput = np.vstack([TxOutputP0, TxOutputP1])

        # Indicate that the packet was built
        self.PacketWasBuild          = True

        return TxOutput







   


# ----------------------------------------------------------------------
# > Testbench
# ----------------------------------------------------------------------
if __name__ == '__main__':
    
    Test = 0
    bPlot = False # True 

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
        NumOfdmSymbols = 20 # 15 # 20
        ControlInfo = CControlInformation(ReferenceSymbolPeriodicityIndex= 3      # 0/1/2/3 - [1, 3,  6, 12]
                                        , ReferenceSignalSpacingIndex    = 0 #was:2      # 0/1/2/3 - [3, 6, 12, 24]
                                        , NumSignalFieldSymbolsIndex     = 1      # 0/1/2/3 - [1, 2,  4, 10]
                                        , SignalFieldModulationFlag      = 1      # 0/1     - BPSK / QPSK
                                        , NumberTxAntennaPortFlag        = 1      # 0/1     - 1TX  / 2TX
                                        , DcSubcarrierFlag               = 0)     # 0/1     - 1/13 Subcarriers
        
        print(ControlInfo)

        # -----------------------------------------------------------------------------------------
        # > We preinitialize the resource grid here. 
        # -----------------------------------------------------------------------------------------
        # In fact, we are simply building three arrays that indicate the resource element types for the subcarriers
        # for the first reference Symbol, the remaining reference symbols and the data symbols.
        # This call is actually not required, the call to InitializeResourceGrid() below, will automatically do the
        # PreInitReferenceGrid() procedure.
        FlexLinkTransmitter.PreInitReferenceGrid(ControlInfo)


        # -----------------------------------------------------------------------------------------
        # > Define the Signal Field and add it to the resource grid
        # -----------------------------------------------------------------------------------------
        NumDataBlocks = 40
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

        # Here we complete build the ResourceGridEnum, which enumerate the type of every resource element withing.
        # At this point, we may plot the ReosurceGridEnum and the capacity tables for PayloadA are available.
        FlexLinkTransmitter.InitializeResourceGrid(ControlInfo, SignalFieldFormat1)



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
            CurrentDataBlock = np.random.randint(low=0, high=2, size= DataBlockSize, dtype = np.uint16)
            DataBlockList.append(CurrentDataBlock)

        FlexLinkTransmitter.AddPayloadA(DataBlockList)

        #FlexLinkTransmitter.PlotResourceGrid(0)
        #FlexLinkTransmitter.PlotResourceGrid(1)

        # ------------------------------------------------------------------------------------------
        # > Build the packet
        # ------------------------------------------------------------------------------------------
        strPreambleALength = 'long' #'short'   # Either 'short' = 50 useconds, or 'long' = 250 useconds
        Output = FlexLinkTransmitter.BuildTxPacket(strPreambleALength, bPlot)


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
        print('end')


    # -----------------------------------------------------------------------------------------
    # > Overwrite the information in the first reference signal with all ones to test the OFDM modulator
    # -----------------------------------------------------------------------------------------
    #Copy = FlexLinkTransmitter.ResourceGrid[:,0].copy()
    #FlexLinkTransmitter.ResourceGrid[:, 0] = (1+1j)*np.ones(Copy.shape, np.complex64)

    #Output = FlexLinkTransmitter.OfdmModulate()
    #stop = 1

 