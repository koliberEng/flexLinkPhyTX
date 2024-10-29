# File:       FlexLinkPhyBase.py
# Notes:      This script contains physical layer code and procedures that are either:
#             1. Shared by both CFlexLinkTransmitter and CFlexLinkReceiver
#                - Resource grid enumeration
#             2. Are complimentary modules and are best included in a single file for easier debugging and maintenance.
#                - SFBC encoding and decoding
#                - OFDM Modulation and Demodulation

__title__     = "FlexLinkPhyBase"
__author__    = "Andreas Schwarzinger"
__status__    = "preliminary"
__date__      = "April, 27th, 2024"
__copyright__ = 'Andreas Schwarzinger'

# --------------------------------------------------------
# Import Statements
# --------------------------------------------------------
from   FlexLinkParameters import *
from   FlexLinkCoder      import CLinearFeedbackShiftRegister
import numpy              as np
import matplotlib.pyplot  as plt
import math



# --------------------------------------------------------
# > Class: CFlexLinkPhyBase
# --------------------------------------------------------
class CFlexLinkPhyBase():
    '''
    brief: This is the base class for both CFlexLinkTransmitter and CFlexLinkReceiver
    '''

    # ----------------------------------
    # > Function: Constructor
    # ----------------------------------
    def __init__(self
               , bLteBw:         bool = True
               , bOverSampling:  bool = True):
        
        # ------------------------------------
        # > Error checking
        # ------------------------------------
        assert isinstance(bLteBw, bool),         'The input argument bLteBw must be a boolean value'
        assert isinstance(bOverSampling, bool),  'The input argument bOverSampling must be a boolean value'  
           
        # --------------------------------------------------
        # > Set parameters based on the oversampling flag
        # --------------------------------------------------
        if bOverSampling == True:               # The IFFT will produce samples at 40MHz sample rate
            self.IfftSize        = 2048         # This is the suggested mode as the DAC should not be running at 20MHz but 40MHz --> 2048/40MHz = 51.2 microseconds
            self.CP_Length       = 232          # 232 / 40MHz = 5.8 microseconds
            self.SampleRate      = 40e6         # This reduces sample and hold distortion produced by the DAC and makes analog filtering easier.
        else:                                   # The IFFT will produce samples at 20MHz sample rate
            self.IfftSize        = 1024         # This mode is mostly meant for debugging and illustration                       --> 1024/20MHz = 51.2 microseconds
            self.CP_Length       = 116          # 116 / 20MHz = 5.8 microseconds
            self.SampleRate      = 20e6   
               
        self.ScSpacing           = self.SampleRate/self.IfftSize                                                               # = 19.53125KHz
        
        # ---------------------------------------------------
        # > Set parameters based on the bandwidth flag (See the resource grid layout figure in the specification)
        # ---------------------------------------------------
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
        self.PosIfftIndices           = np.arange(0,                                     self.CenterSubcarrier + 1, 1, np.int16)   # These are the IFFT input indices, to which we 
        self.NegIfftIndices           = np.arange(self.IfftSize - self.CenterSubcarrier, self.IfftSize,          1, np.int16)      # connect the negative and positive subcarriers

        self.OccupiedBw               = self.NumSubcarriers * self.ScSpacing
        self.PosSubcarrierIndices     = np.arange(self.CenterSubcarrier, self.NumSubcarriers,   1, np.int16)
        self.NegSubcarrierIndices     = np.arange(0,                     self.CenterSubcarrier, 1, np.int16)
        
        # --------------------------------------------------
        # > Create the basic scrambling sequence. 
        # --------------------------------------------------
        # This sequence is used to make bit sequence appear more random in nature. The sequence is 255 bits long and repeats afterwards.
        self.LFSR                     = CLinearFeedbackShiftRegister(bUseMaxLengthTable = True, IndexOrTapConfig = 8)
        self.LFSR.InitializeShiftRegister([0, 0, 0, 0, 0, 0, 0, 1])
        self.ScrambingSequence        = self.LFSR.RunShiftRegister(NumberOutputBits= 255) 

        # -------------------------------------------------
        # > Declare the existence of the following ReType matrices that will hold resource element type information
        # -------------------------------------------------
        # These are matrices with two columns. Column 0 is for Port0 and column 1 is for Port1 
        # Resource element type definitions for the first reference symbol
        self.FirstReferenceSymbolReType   = None
        # Resource element type definitions for the remaining reference symbols
        self.OtherReferenceSymbolReType   = None
        # Resource element type definitions for the data symbol (not a reference symbol)
        self.DataSymbolReType             = None

        # ---------------------------------------------------
        # > Declare the existence of the capcity tables
        # ---------------------------------------------------
        self.DataREsPerResourceBlock_RefSymbol  = None     # Will hold the number of REs in each resource block available for data in an OFDM reference symbol
        self.DataREsPerResourceBlock_DataSymbol = None     # Will hold the number of REs in each resource block available for data in an OFDM data symbol
        self.BPS_Array                          = None     # An array that hold the bits per QAM symbol for each resource block
        self.CapacityTable_RefSymbol            = None     # Will hold the number of bits per resource block in an OFDM reference symbol
        self.CapacityTable_DataSymbol           = None     # Will hold the number of bits per resource block in an OFDM data symbol

        # -------------------------------------------------
        # > Declare variables for the resource grid
        # -------------------------------------------------
        self.ControlInformation    = None
        self.SignalField           = None
        self.ResourceGridEnum      = None    # The magic matrix that tells us what type of information each resource element in the resource grid holds
        self.ResourceBlockEnum     = None    # The magic matrix that tells us what type of information each resource block in the resource grid holds	
        self.PreInitializationDone = False   # Has PreInitReferenceGrid() been called
        self.InitializationDone    = False   # Has InitReferenceGrid()    been called
        







    # --------------------------------------------------------------------------------------------------------------------------------------- #
    #                                                                                                                                         #
    #                                                                                                                                         #
    # > Function: PreInitReferenceGrid()                                                                                                      #
    #                                                                                                                                         #
    #                                                                                                                                         #
    # --------------------------------------------------------------------------------------------------------------------------------------- #
    def PreInitReferenceGrid(self
                            , ControlInfo: CControlInformation):
        '''
        In this function we will enumerate the resource element types for the first reference symbol, other reference symbols as well
        as OFDM data symbols. There is nothing else going on here.
        Fill out: self.FirstReferenceSymbolReType
        Fill out: self.OtherReferenceSymbolReType
        Fill out: self.DataSymbolReType 
        '''
        # Error checking
        assert isinstance(ControlInfo, CControlInformation), 'The ControlInfo input argument is of invalid type'
        self.ControlInformation = ControlInfo

        # If we move back to the PreInitReferenceGrid() function, and the resource grid was previously initialized
        # in the InitReferenceGridEnum() function, then invalidate that initialization.
        self.InitializationDone = False

        # --------------------------------------------------------
        # > The following initialization DOES NOT need information from the CControlInformation object
        # --------------------------------------------------------
        # Initialize the enum vectors for the first reference symbol, the other reference symbols, and the data symbols
        self.FirstReferenceSymbolReType      = EReType.Empty.value * np.ones([self.NumSubcarriers, 2], np.int8)
        # Resource element type definitions for the remaining reference symbols
        self.OtherReferenceSymbolReType      = EReType.UnassignedDataP0.value * np.ones([self.NumSubcarriers, 2], np.int8) 
        # Resource element type definitions for the data symbol (not a reference symbol)
        self.DataSymbolReType                = EReType.UnassignedDataP0.value * np.ones([self.NumSubcarriers, 2], np.int8)

        if self.ControlInformation.NumberTxAntennas == 1:
            self.OtherReferenceSymbolReType[:,1] = EReType.Empty.value * np.ones(self.NumSubcarriers, np.int8) 
            self.DataSymbolReType[:,1]           = EReType.Empty.value * np.ones(self.NumSubcarriers, np.int8)
        else:
            self.OtherReferenceSymbolReType[:,1] = EReType.UnassignedDataP1.value * np.ones(self.NumSubcarriers, np.int8) 
            self.DataSymbolReType[:,1]           = EReType.UnassignedDataP1.value * np.ones(self.NumSubcarriers, np.int8)

        # Let's make sure the center subcarrier is zeros out (These arrays are filled out once CControlInformation is available)
        self.FirstReferenceSymbolReType[self.CenterSubcarrier, 0] = EReType.Empty.value
        self.FirstReferenceSymbolReType[self.CenterSubcarrier, 1] = EReType.Empty.value
        self.OtherReferenceSymbolReType[self.CenterSubcarrier, 0] = EReType.Empty.value
        self.OtherReferenceSymbolReType[self.CenterSubcarrier, 1] = EReType.Empty.value
        self.DataSymbolReType[self.CenterSubcarrier, 0]           = EReType.Empty.value
        self.DataSymbolReType[self.CenterSubcarrier, 1]           = EReType.Empty.value


        # Program the first reference symbol resource element types
        for k in range(0, self.NumSubcarriers):   # k is the subcarrier index

            # Can we place a control element at this subcarrier???
            bResourceElementOpportunity = (k % 3 == 0)
            if bResourceElementOpportunity == True and k != self.CenterSubcarrier:
                self.FirstReferenceSymbolReType[k, 0] = EReType.Control.value
                self.FirstReferenceSymbolReType[k, 1] = EReType.Empty.value

            # Can we place a reference signal for port 0 at this subcarrier???
            bResourceElementOpportunity = ((k-2) % 3 == 0)
            if bResourceElementOpportunity == True and k != self.CenterSubcarrier:
                self.FirstReferenceSymbolReType[k, 0] = EReType.RefSignalPort0.value
                self.FirstReferenceSymbolReType[k, 1] = EReType.Empty.value

            # Can we place a reference signal for port 1 at this subcarrier???
            bResourceElementOpportunity = ((k-1) % 3 == 0)
            if bResourceElementOpportunity == True and k != self.CenterSubcarrier and self.ControlInformation.NumberTxAntennas == 2:
                self.FirstReferenceSymbolReType[k, 1] = EReType.RefSignalPort1.value
                self.FirstReferenceSymbolReType[k, 0] = EReType.Empty.value





        # --------------------------------------------------------
        # > The following initialization DOES need information from the CControlInformation object
        # --------------------------------------------------------
        # Configure self.OtherReferenceSymbolReType 
        for n in range(0, 1000):      
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
                self.OtherReferenceSymbolReType[k, 0] = EReType.RefSignalPort0.value
                self.OtherReferenceSymbolReType[k, 1] = EReType.Empty.value
                                    
        # ----------------------------------
        # > Place the reference signals of antanna port 1 into the remaining reference symbols
        # ----------------------------------
        for n in range(0, 1000):  
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
                # Place the reference signal into the resource grid if we have 2 Tx antennas
                if ControlInfo.NumberTxAntennas == 2:
                    # The reference signal position in the other resource grid of the other antenna port must be empty
                    self.OtherReferenceSymbolReType[k, 1] = EReType.RefSignalPort1.value  
                    self.OtherReferenceSymbolReType[k, 0] = EReType.Empty.value  


        # ----------------------------------
        # > Place the DC Signals in the correct area of self.DataSymbolReType and self.OtherReferenceSymbolReType
        # ----------------------------------
        Half    = int(ControlInfo.NumberDcSubcarriers / 2)
        DcRange = range(self.CenterSubcarrier - Half, self.CenterSubcarrier + Half + 1)

        for k in DcRange:                               # k is the subcarrier index
            # No data may be mapped at the DC subcarriers
            self.DataSymbolReType[k, 0] = EReType.Empty.value
            self.DataSymbolReType[k, 1] = EReType.Empty.value

            # No data may be mapped at the DC subcarriers but reference signal may be mapped on top of the DC subcarriers
            # Therefore, we will only force these resource elements to Empty if they are still unassigned and haven't
            # been reserve for reference signal yet
            if self.OtherReferenceSymbolReType[k, 0] != EReType.RefSignalPort0.value:         # This RE must not have been assigned yet
                self.OtherReferenceSymbolReType[k, 0] = EReType.Empty.value
                 
            if ControlInfo.NumberTxAntennas == 2:
                if self.OtherReferenceSymbolReType[k, 1] != EReType.RefSignalPort1.value:     # This RE must not have been assigned yet
                    self.OtherReferenceSymbolReType[k, 1] = EReType.Empty.value


        # -------------------------------------------------
        # > Define Capacity Tables (Data RE's per resource block 0 through 69 / 75)
        # -------------------------------------------------
        self.DataREsPerResourceBlock_RefSymbol  = np.zeros(self.NumResourceBlocks, np.uint16)
        self.DataREsPerResourceBlock_DataSymbol = np.zeros(self.NumResourceBlocks, np.uint16)

        for k in range(0, self.NumSubcarriers):
            # Determine the current Resource Block
            kVirtual = k
            if k > self.CenterSubcarrier:
                kVirtual = k -1
            
            CurrentResourceBlock = int(kVirtual / 12)

            # Is this resource element available for data (is it still unassigned) ??
            if self.OtherReferenceSymbolReType[k, 0] == EReType.UnassignedDataP0.value:
                self.DataREsPerResourceBlock_RefSymbol[CurrentResourceBlock] += 1

            if self.DataSymbolReType[k, 0] == EReType.UnassignedDataP0.value:
                self.DataREsPerResourceBlock_DataSymbol[CurrentResourceBlock] += 1


        self.PreInitializationDone = True











    # --------------------------------------------------------------------------------------------------------------------------------------- #
    #                                                                                                                                         #
    #                                                                                                                                         #
    # > Function: InitReferenceGridEnum()                                                                                                     #
    #                                                                                                                                         #
    #                                                                                                                                         #
    # --------------------------------------------------------------------------------------------------------------------------------------- #
    def InitReferenceGridEnum(self
                            , SignalField: CSignalField):
        '''
        In this function, we initialize the self.ResourceGridEnum matrix, which indicates what type of information resides
        at the resource elements. Once we have access to the signal field, we will be able to fill out this matrix all the way
        through the PayloadA. For payloadB, we will have to know what the information regarding the MAC headers embedded in payloadA.
        -> Initialize self.ResourceGridEnum
        -> Fill self.ResourceGridEnum with self.FirstReferenceSymbolReType, self.OtherReferenceSymbolReType, self.DataSymbolReType
        '''

        # Error checking
        assert isinstance(SignalField, CSignalField), 'The SignalField input argument is of improper type'
        assert self.PreInitializationDone == True, 'You must call PreInitReferenceGrid() before executing this function'

        # Start setup
        self.SignalField           = SignalField
        NumberOfAntennas           = self.ControlInformation.NumberTxAntennas
        NumberOfSignalFieldSymbols = self.ControlInformation.NumberSignalFieldSymbols 
        NumberOfdmSymbols          = self.SignalField.NumOfdmSymbols
        ReferenceSymbolPeriodicity = self.ControlInformation.ReferenceSymbolPeriodicity

        # ----------------------------------
        # > Create the resource grid enumeration matrix and populate the reference and data symbols
        # ----------------------------------
        MaxNumberTxAntennas     = 2
        self.ResourceGridEnum   = EReType.Unassigned.value * np.ones([self.NumSubcarriers, NumberOfdmSymbols, MaxNumberTxAntennas], np.int16)
        
        for SymbolIndex in range(0, NumberOfdmSymbols):
            # Place the first OFDM referene symbol from the previously computed variable self.FirstReferenceSymbolReType
            if SymbolIndex == 0:
                self.ResourceGridEnum[:, SymbolIndex, :] = self.FirstReferenceSymbolReType 
                continue

            # Place the Remaining reference symbols from the previously computed variable self.OtherReferenceSymbolReType
            if SymbolIndex % ReferenceSymbolPeriodicity == 0:
                self.ResourceGridEnum[:, SymbolIndex, :] = self.OtherReferenceSymbolReType 
            
            # Place data symbols from the previously computed variable self.DataSymbolReType
            else:
                self.ResourceGridEnum[:, SymbolIndex, :] = self.DataSymbolReType 


        # -----------------------------------
        # > Now add the resource type for the Signal Field
        # -----------------------------------     
        for l in range(1, NumberOfSignalFieldSymbols + 1):
            for k in range(0, self.NumSubcarriers):
                ResourceElementTypeP0 = self.ResourceGridEnum[k, l, 0]
                ResourceElementTypeP1 = self.ResourceGridEnum[k, l, 1]

                if(ResourceElementTypeP0 == EReType.UnassignedDataP0.value):
                    self.ResourceGridEnum[k, l, 0] = EReType.SignalField.value
                if(ResourceElementTypeP1 == EReType.UnassignedDataP1.value):
                    self.ResourceGridEnum[k, l, 1] = EReType.SignalField.value


        # -----------------------------------
        # > Now add the resource element type for the PayloadA
        # -----------------------------------    
        # From the Signal field information, we can work backwards to find the data word size 
        # and data block size
        CodeBlockSize       = self.SignalField.CodeBlockSizeA
        DataWordSize        = int(CodeBlockSize * self.SignalField.CodingRate)
        DataBlockSize       = DataWordSize - 16                              # 16 CRC bits
        MinimumCodeWordSize = int(CodeBlockSize * (1 + self.SignalField.RateMatchingFactor))
        
        # Let's ensure that in both cases (signal field format 1 and 2), the BPS quantity is
        # a vector that covers all resource blocks
        if self.ControlInformation.SignalFieldFormat == 1:
            assert isinstance(self.SignalField.BitsPerQamSymbol, int), 'The bits per QAM symbol for signal field format 1 must be a single integer'
            self.BPS_Array = self.SignalField.BitsPerQamSymbol * np.ones(self.NumResourceBlocks, np.uint16)
        else:
            self.BPS_Array = self.SignalField.BitsPerQamSymbol

        assert len(self.BPS_Array) == self.NumResourceBlocks

        # We can now compute the final capacity table for PayloadA
        self.CapacityTable_RefSymbol            =  self.BPS_Array * self.DataREsPerResourceBlock_RefSymbol
        self.CapacityTable_DataSymbol           =  self.BPS_Array * self.DataREsPerResourceBlock_DataSymbol

        CodeWordCount    = 0
        CodeWordBitCount = 0
        bFirstPass        = True
        for l in range(NumberOfSignalFieldSymbols + 1, self.SignalField.NumOfdmSymbols):    # l - OFDM symbol index
            for k in range(0, self.NumSubcarriers):                                         # k - Subcarrier index
                # Determine in which resource block we are currently located
                kVirtual = k
                if kVirtual > self.CenterSubcarrier:
                    kVirtual = k -1

                CurrentResourceBlockIndex   = int(kVirtual / 12)            # As there are 12 subcarriers in a resource block
                BPS                         = self.BPS_Array[CurrentResourceBlockIndex]

                # Did we completely fill a resource block during the last iteration????
                # The very first k = 0 would trigger the bLastResourceBlockFull without the bFirstPass check
                bLastResourceBlockFull = kVirtual % 12 == 0 and bFirstPass == False
                bFirstPass = False

                # If the resource block was filled during the last iteration, then check to see whether we have used all bits in 
                # the code work. If so, let's increment the code 
                if bLastResourceBlockFull == True:
                    # Note, the actual number of bits that we use will likely be slightly larger than the
                    # MinimumCodeWordSize. Remember, that we want to fill up an integer number of resource blocks
                    # The MinimumCodeWordSize may not be enough bits to do that. 
                    #print(NumCodeWordBits/NumResourceBlocksUsed)
                    if CodeWordBitCount >= MinimumCodeWordSize:
                        CodeWordCount    += 1
                        CodeWordBitCount  = 0
                
                # If we have finished processing all data blocks, then return from this function
                if CodeWordCount == self.SignalField.NumberDataBlocksA:
                    self.InitializationDone = True
                    return

                # Check the resource element type of the current k, l pair
                ResourceElementTypeP0 = self.ResourceGridEnum[k, l, 0]
                ResourceElementTypeP1 = self.ResourceGridEnum[k, l, 1]

                # ---------------------
                if(ResourceElementTypeP0 == EReType.UnassignedDataP0.value):
                    if CodeWordCount % 2 == 0:   # Even Code block
                        self.ResourceGridEnum[k, l, 0] = EReType.PayloadAEvenCodeWord.value
                    else:                         # Odd Code block
                        self.ResourceGridEnum[k, l, 0] = EReType.PayloadAOddCodeWord.value

                    CodeWordBitCount += BPS

                # ----------------------
                if(ResourceElementTypeP1 == EReType.UnassignedDataP1.value):
                    if CodeWordCount % 2 == 0:   # Even Code block
                        self.ResourceGridEnum[k, l, 1] = EReType.PayloadAEvenCodeWord.value
                    else:                         # Odd Code block
                        self.ResourceGridEnum[k, l, 1] = EReType.PayloadAOddCodeWord.value


    
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

                if Value == 3 or Value == -3:  # UnassignedData --> Yellow
                    Z[k,l] = 0.40

                if Value == 0:   # Empty --> White / Light Grey
                    Z[k,l] = 1 

                if Value == 2:   # RefSignalPort0 --> Grey
                    Z[k,l] = 0.90

                if Value == -2:   # RefSignalPort1 --> Red
                    Z[k,l] = 0.28

                if Value == 4:   # Control information --> Blue
                    Z[k,l] = 0.08 

                if Value == 5:   # Signal Field --> Green
                    Z[k,l] = 0.5

                if Value == 6:   # PayloadA Even Code Word
                    Z[k,l] = 0.65

                if Value == 7:   # PayloadA Odd  Code Word --> Blue
                    Z[k,l] = 0.70

                # Here I have to establish the min and max values of the CMAP. These two resource elements do not
                # represent any content.
                if k == NumberSubcarriers-1 and l == NumberOfdmSymbols - 1:
                    Z[k,l] = 1.0    # This is the maximum color value
                
                if k == NumberSubcarriers - 2 and l == NumberOfdmSymbols - 1:
                    Z[k,l] = 0.08   # This is the minimum color value

        X, Y            = np.mgrid[0:NumberSubcarriers+1:1, 0:NumberOfdmSymbols+1:1]

        if AntennaPort == 0:
            fig = plt.figure('Resource Grid Port 0')
        else:
            fig = plt.figure('Resource Grid Port 1') 

        plt.pcolor(Y, X, Z, cmap="tab20c", )
        plt.title('Resource Grid\n\n(Blue - Control / Dark Grey - Ref Signal P0 / Red - Ref Signal P1 / Dark Purple - PayloadA Even CodeWord)\n( Light Purple - PayloadA Odd CodeWord / Orange - Available for Data / Light Gray - Empty = 0+0j)')
        plt.ylabel('Subcarriers') 
        plt.xlabel('Ofdm Symbols') 
        plt.grid(color='#999999') 
        plt.show()



















    # ---------------------------------------------------------------------------------------------------------------- #
    #                                                                                                                  #
    #                                                                                                                  #
    # > Function: OfdmModulate() - This function transforms the resource grid into the time domain OFDM symbols        #
    #                                                                                                                  #
    #                                                                                                                  #
    # ---------------------------------------------------------------------------------------------------------------- #
    def OfdmModulate(self
                   , AntennaPort: int 
                   , ResourceGrid: np.ndarray) -> np.ndarray:
        '''
        This function OFDM modulates the QAM values in the resource grid
        '''
        assert isinstance(AntennaPort, int),         'The AntennaPort input argument is of invalid type' 
        assert AntennaPort == 0 or AntennaPort == 1, 'The AntennaPort input argument is invalid'
        if AntennaPort == 1 and self.ControlInformation.NumberTxAntennas == 1:
            assert False, 'Attemping to render non-existing antenna port 1 signal'

        ResourceGrid = ResourceGrid[:, :, AntennaPort]

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

            #print(np.var(Scaling * IfftOutputBuffer))

            # Increment the OutputSequenceIndex
            OutputSequenceIndex += OfdmSymbolLength

        return OfdmTxOutput
    






    # ---------------------------------------------------------------------------------------------------------------- #
    #                                                                                                                  #
    #                                                                                                                  #
    # > Function: OfdmDemodulator() - This function transforms the time domain signal into an OFDM resource grid       #
    #                                                                                                                  #
    #                                                                                                                  #
    # ---------------------------------------------------------------------------------------------------------------- #
    def OfdmDemodulate(self
                     , InputSequence:    np.ndarray
                     , FirstStartSample: int = 0) -> np.ndarray:
        '''
        This function OFDM demodulates the IQ input sequence into a resource grid
        - InputSequence:     The input sequence to be demodulated
        - FirstStartSample:  The sample index of the first sample of the CP portion of OFDM symbol of the stongest path
                             (Provided via correlation against preamble B)
        '''
        assert isinstance(InputSequence, np.ndarray),                        'The input argument InputSequence is of invalid type'
        assert isinstance(FirstStartSample, int),                            'The input argument FirstStartSample is of invalid type'
        
        # ---------------------------------------------------
        # The OFDM Demodulator may only run at the 20MHz sample rate
        FftSize           = 1024                  # This mode is mostly meant for debugging and illustration  --> 1024/20MHz = 51.2 microseconds
        CP_Length         = 116                   # 116 / 20MHz = 5.8 microseconds
        OFDM_SymbolLength = FftSize + CP_Length 
        SampleRate        = 20e6   
        ScSpacing         = SampleRate/FftSize    # = 19.53125KHz

        # By definition, we will always begin the FFT operation 1 microsecond before the FirstStartSample
        # This is to avoid the effect of procursor images (early paths) that may be present in the signal
        TimeAdvance = 1e-6
        StartSample = FirstStartSample + CP_Length - int(SampleRate * TimeAdvance)
        assert StartSample >= 0, 'The StartSample is invalid'

        # Determine the number of OFDM symbols in the input sequence
        NumberOfdmSymbols  = int((InputSequence.size - StartSample + CP_Length) / OFDM_SymbolLength)

        # We define the subcarriers in terms of IEEE tones as this makes the demodulator easier
        StartTone    = -int((self.NumSubcarriers - 1) / 2)
        StopTone     = abs(StartTone) 
        TonesIEEE    = np.arange(StartTone, StopTone + 1, 1, np.int16)
        Compensation = np.exp(1j * 2 * np.pi * TimeAdvance * TonesIEEE * ScSpacing) 

        # Start the OFDM Demodulation process
        ResourceGrid = np.zeros([self.NumSubcarriers, NumberOfdmSymbols], np.complex64)


        Scaling =  np.sqrt(self.NumSubcarriers / FftSize) / np.sqrt(FftSize)  # The 1/sqrt(2) factor is due to the fact that the FFT is not normalized
        for l in range(0, NumberOfdmSymbols):   # l is the OFDM symbol index
            # Fetch the OFDM symbol from the input sequence
            StartIndex = StartSample + l * OFDM_SymbolLength
            StopIndex  = StartIndex + FftSize
            FftInput   = InputSequence[StartIndex:StopIndex]

            # Perform the FFT operation
            FftOutput = np.fft.fft(FftInput) * Scaling

            # Place the FFT Output into the Resource Grid
            ResourceGrid[self.PosSubcarrierIndices, l] = FftOutput[self.PosIfftIndices] 
            ResourceGrid[self.NegSubcarrierIndices, l] = FftOutput[self.NegIfftIndices]
            ResourceGrid[:, l] *= Compensation

        return ResourceGrid
    








    # --------------------------------------------------------------------------------------------------------------------------------------- #
    # > Function: SFBC_Encode()                                                                                                               #
    # --------------------------------------------------------------------------------------------------------------------------------------- #
    @staticmethod
    def SFBC_Encode(X0: np.complex64
                  , X1: np.complex64) -> np.array:
        '''
        This function return the SFBC encoded information for a pair of input QAM values.
        '''
        Symbol0Port0  = X0
        Symbol1Port0  = X1
        Symbol0Port1  = -np.conj(X1)
        Symbol1Port1  =  np.conj(X0)
        Output        = np.array([Symbol0Port0, Symbol1Port0, Symbol0Port1, Symbol1Port1])
        
        return Output







    # --------------------------------------------------------------------------------------------------------------------------------------- #
    # > Function: SFBC_Decode()                                                                                                               #
    # --------------------------------------------------------------------------------------------------------------------------------------- #
    @staticmethod
    def SFBC_Decode(Y0:  np.complex64
                  , Y1:  np.complex64 
                  , H00: np.complex64
                  , H10: np.complex64
                  , H01: np.complex64
                  , H11: np.complex64) -> tuple:
        '''
        This function return the SFBC decoded information for a pair of observed QAM values and four channel coefficients.
        - Y0:  The first observed QAM value
        - Y1:  The second observed QAM value
        - H00: The channel coefficient between the resource element of encoded Symbol 0, Antenna 0 and observation Y0 at the single receive antenna
        - H10: The channel coefficient between the resource element of encoded Symbol 1, Antenna 0 and observation Y1 at the single receive antenna
        - H01: The channel coefficient between the resource element of encoded Symbol 0, Antenna 1 and observation Y0 at the single receive antenna
        - H11: The channel coefficient between the resource element of encoded Symbol 1, Antenna 1 and observation Y1 at the single receive antenna
        Notes: We are using method 2 in the MatLab code of Section 9.2.2. 'Digital Signal Processing in Modern Communication Systems Edition 3
        '''
        assert isinstance(Y0,  np.complex64), 'The input argument Y0  is of invalid type'
        assert isinstance(Y1,  np.complex64), 'The input argument Y1  is of invalid type'
        assert isinstance(H00, np.complex64), 'The input argument H00 is of invalid type'
        assert isinstance(H10, np.complex64), 'The input argument H10 is of invalid type'
        assert isinstance(H01, np.complex64), 'The input argument H01 is of invalid type'
        assert isinstance(H11, np.complex64), 'The input argument H11 is of invalid type'

        # Define the observation vector Y
        Y    = np.array([Y0, np.conj(Y1)])

        C    = 1/(H00*np.conj(H10) + np.conj(H11)*H01)
        Temp = np.array([[np.conj(H10), H01], [-np.conj(H11), H00]])
        X    = C * np.matmul(Temp, Y)
        
        return X[0], np.conj(X[1])










# --------------------------------------------------------
# > Main
# --------------------------------------------------------
if __name__ == "__main__":
    Test = 1                       # 0 - Test the OFDM Modulator and Demodulator
                                   # 1 - Test the SFBC Encoder and Decoder 

    # Test the OFDM Modulator and Demodulator
    if Test == 0:
        bLteBw = False
        bOverSampling = False

        PhyBase = CFlexLinkPhyBase(bLteBw, bOverSampling)

        # Test the OFDM Modulator
        ResourceGridTx = np.ones([PhyBase.NumSubcarriers, 2, 2], np.complex64)
        OfdmTxOutput   = PhyBase.OfdmModulate(0, ResourceGridTx)

        ResourceGridRx = PhyBase.OfdmDemodulate(OfdmTxOutput)

        # Verify that the Resource Grids are the same
        MaxError = np.max(np.abs(ResourceGridTx[:,:,0] - ResourceGridRx))
        assert MaxError < 1e-6, 'The OFDM Modulator and Demodulator failed to produce the same resource grid'
        print('The OFDM Modulator and Demodulator produced the same resource grid')




    # Test the SFBC Encoder and Decoder
    if Test == 1:
        # Define the transmitted symbols X0, X1
        X0 = np.complex64(1    +   1j)
        X1 = np.complex64(-0.5 - 2.0j)

        # SFBC Encode the symbols
        Symbol0Port0, Symbol1Port0, Symbol0Port1, Symbol1Port1 = CFlexLinkPhyBase.SFBC_Encode(X0, X1)

        # Define the channel coefficients
        H00 = np.complex64(1  + 0.2j)    # The channel coefficient between the resource element of encoded Symbol 0, Antenna 0 and observation Y0 at the single receive antenna
        H10 = np.complex64(1  -   1j)    # The channel coefficient between the resource element of encoded Symbol 1, Antenna 0 and observation Y1 at the single receive antenna
        H01 = np.complex64(2  +   1j)    # The channel coefficient between the resource element of encoded Symbol 0, Antenna 1 and observation Y0 at the single receive antenna
        H11 = np.complex64(-3 - 1.2j)    # The channel coefficient between the resource element of encoded Symbol 1, Antenna 1 and observation Y1 at the single receive antenna

        # The observations at the single receiver antenna are called Y0, Y1
        Y0  = H00*Symbol0Port0 + H01*Symbol0Port1    
        Y1  = H10*Symbol1Port0 + H11*Symbol1Port1 

        # SFBC Decode the observations
        X0_Estimate, X1_Estimate = CFlexLinkPhyBase.SFBC_Decode(Y0, Y1, H00, H10, H01, H11)

        # Verify that the output is the same as the input
        assert np.abs(X0 - X0_Estimate) < 0.00001, 'The SFBC Decoder failed to produce the same X0 value'
        assert np.abs(X1 - X1_Estimate) < 0.00001, 'The SFBC Decoder failed to produce the same X1 value'
        print('The SFBC Encoder and Decoder produced the same output')
        