# File:       FlexLinkPhyBase.py
# Notes:      This script contains physical layer code and procedures that are shared by
#             both CFlexLinkTransmitter and CFlexLinkReceiver

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
        # -------------------------------------------------
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
        # > Define the type of information that needs to go into the first reference symbol
        # -------------------------------------------------
        # These are matrices with two columns. Column 0 is for Port0 and column 1 is for Port1 
        # Resource element type definitions for the first reference symbol
        self.FirstReferenceSymbolReType   = EReType.Empty.value      * np.ones([self.NumSubcarriers, 2], np.int8)
        # Resource element type definitions for the remaining reference symbols
        self.OtherReferenceSymbolReType   = EReType.Unassigned.value * np.ones([self.NumSubcarriers, 2], np.int8) 
        # Resource element type definitions for the data symbol (not a reference symbol)
        self.DataSymbolReType             = EReType.Unassigned.value * np.ones([self.NumSubcarriers, 2], np.int8)

        for k in range(0, self.NumberSubcarriers):   # k is the subcarrier index

            # Can we place a control element at this subcarrier???
            bControlElementOpportunity = (k % 3 == 0)
            if bControlElementOpportunity == True or k != self.CenterSubcarrier:
                self.FirstReferenceSymbolReType[k, 0] = EReType.Control.value

            # Can we place a reference signal for port 0 at this subcarrier???
            bControlElementOpportunity = ((k-2) % 3 == 0)
            if bControlElementOpportunity == True or k != self.CenterSubcarrier:
                self.FirstReferenceSymbolReType[k, 0] = EReType.RefSignalPort0.value

            # Can we place a reference signal for port 1 at this subcarrier???
            bControlElementOpportunity = ((k-1) % 3 == 0)
            if bControlElementOpportunity == True or k != self.CenterSubcarrier:
                self.FirstReferenceSymbolReType[k, 1] = EReType.RefSignalPort1.value

        # Let's make sure the center subcarrier is zeros out
        self.OtherReferenceSymbolReType[self.CenterSubcarrier, 0] = EReType.Empty.value
        self.OtherReferenceSymbolReType[self.CenterSubcarrier, 1] = EReType.Empty.value
        self.DataSymbolReType[self.CenterSubcarrier, 0]           = EReType.Empty.value
        self.DataSymbolReType[self.CenterSubcarrier, 1]           = EReType.Empty.value


        # -------------------------------------------------
        # > Declara variables for the resource grid
        # -------------------------------------------------
        self.ResourceGrid    = None
        self.ResourceGridEnu = None



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
        This function creates a resource element type array for an OFDM symbol with / without Reference signals 
        '''
        # Error checking
        assert isinstance(ControlInfo, CControlInformation), 'The ControlInfo input argument is of invalid type'

        # Configure self.OtherReferenceSymbolReTypeP0
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
                assert self.OtherReferenceSymbolReType[k, 0] == EReType.Unassigned.value, 'This RE must be unassigned'
                self.OtherReferenceSymbolReType[k, 0] = EReType.RefSignalPort0.value
                                    
                if ControlInfo.NumberTxAntennas == 2:
                    # The reference signal position in the other resource grid of the other antenna port must be empty
                    assert self.OtherReferenceSymbolReType[k, 1] == EReType.Unassigned.value, 'This RE must be unassigned'
                    self.OtherReferenceSymbolReType[k, 1] = EReType.Emtpy.value  

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
                # Place the reference signal into the resource grid
                assert self.OtherReferenceSymbolReType[k, 1] == EReType.Unassigned.value, 'This RE must be unassigned'
                self.OtherReferenceSymbolReType[k, 1] = EReType.RefSignalPort1.value
                                    
                if ControlInfo.NumberTxAntennas == 2:
                    # The reference signal position in the other resource grid of the other antenna port must be empty
                    assert self.OtherReferenceSymbolReType[k, 0] == EReType.Unassigned.value, 'This RE must be unassigned'
                    self.OtherReferenceSymbolReType[k, 0] = EReType.Emtpy.value  


        # ----------------------------------
        # > Place the DC Signals in the correct area of the resource grid
        # ----------------------------------
        Half    = int(ControlInfo.NumberDcSubcarriers / 2)
        DcRange = range(self.CenterSubcarrier - Half, self.CenterSubcarrier + Half + 1)

        for k in DcRange:                               # k is the subcarrier index
            # No data may be mapped at the DC subcarriers
            self.DataSymbolReType[k, 0] = EReType.Emtpy.value
            self.DataSymbolReType[k, 1] = EReType.Emtpy.value

            # No data may be mapped at the DC subcarriers but reference signal may be mapped on top of the DC subcarriers
            # Therefore, we will only force these resource elements to Empty if they are still unassigned and haven't
            # been reserve for reference signal yet
            if self.OtherReferenceSymbolReType[k, 0] == EReType.Unassigned.value:         # Nothing has yet been placed here
                self.OtherReferenceSymbolReType[k, 0] = EReType.Emtpy.value
                 
            if ControlInfo.NumberTxAntennas == 2:
                if self.OtherReferenceSymbolReType[k, 1] == EReType.Unassigned.value:     # Nothing has yet been placed here
                    self.OtherReferenceSymbolReType[k, 1] = EReType.Emtpy.value





        # --------------------------------------------------------------------------------------------------------------------------------------- #
        #                                                                                                                                         #
        #                                                                                                                                         #
        # > Function: InitReferenceGrid()                                                                                                         #
        #                                                                                                                                         #
        #                                                                                                                                         #
        # --------------------------------------------------------------------------------------------------------------------------------------- #
        def InitReferenceGrid(self
                           , SignalField: CSignalField):
            pass