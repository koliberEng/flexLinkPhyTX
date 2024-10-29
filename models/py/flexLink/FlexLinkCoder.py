# File:     FlexLinkCoder.py
# Notes:    This file provides CRC, and FEC encoding and decoding services
# Refences: Digital Signal Processing in Modern Communication Systems (Edition 2) Sections 5.6.1 and 5.6.6
#           IEEE Std 802.11n-2009 (Oct 2009) Amendment 5: Enhancements For Higher Throughput, New York, NY 
#           IEEE Std 802.11-2012
#
# Content:  1. CCrcProcessor                -> A class that implements cyclic redundancy checks for error detection
#           2. CLinearFeedbackShiftRegister -> A class that implement linear feedback shift registers for generation of random bits 
#           3. CLdpcProcessor               -> A class that encodes and decodes the Low Density Parity Check codes used in the 802.11 standard
#           4. CPolarProcessor              -> A class that implements generic polar encoding and decoding
#           5. CInterleaverFlexLink         -> A class that enables interleaving for the FlexLink standard

__title__     = "FlexLinkCoder"
__author__    = "Andreas Schwarzinger"
__status__    = "released Version 1.0"
__date__      = "Jan, 4rd, 2023"
__copyright__ = 'Andreas Schwarzinger'

import os
import sys                               # We use sys to include new search paths of modules
OriginalWorkingDirectory = os.getcwd()   # Get the current directory
DirectoryOfThisFile      = os.path.dirname(__file__)   # FileName = os.path.basename
if DirectoryOfThisFile != '':
    os.chdir(DirectoryOfThisFile)        # Restore the current directory

# There are modules that we will want to access and they sit two directories above DirectoryOfThisFile
sys.path.append(DirectoryOfThisFile + "\\..\\..\\DspComm")

import numpy as np
from   SignalProcessing import *
import DebugUtility
import matplotlib.pyplot as plt


















# ------------------------------------------------------------------
# > CCrcProcessor Class
# ------------------------------------------------------------------
class CCrcProcessor():
    """
    This class provides general cyclic redundancy check services.
    """
    # Length 8 CRC used in different standards 
    # See https://en.wikipedia.org/wiki/Cyclic_redundancy_check for more sequences 
    Generator08_LTE       = np.array([1, 1,0,0,1, 1,0,1,1], np.uint16)
    Generator08_Bluetooth = np.array([1, 1,0,1,0, 0,1,1,1], np.uint16)
    Generator08_GSM       = np.array([1, 0,1,0,0, 1,0,0,1], np.uint16)

    # Length 10 CRC  
    Generator10_GSM       = np.array([1, 0,1, 0,1,1,1, 0,1,0,1])   # 0x175
    Generator10_CDMA2000  = np.array([1, 1,1, 1,1,0,1, 1,0,0,1])   # 0x3D9

    # Length 12 CRC  
    Generator12_UMTS      = np.array([1, 1,0,0,0, 0,0,0,0, 1,1,1,1])
    
    # Length 16 is used to protect the signal field (LTE gcrc16)
    Generator16_LTE       = np.array([1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1], dtype = np.uint16)

    # Length 24 is used to protect each transport block in the payload (LTE gcrc24a)
    Generator24_LTEA      = np.array([1,1,0,0,0,0,1,1,0,0,1,0,0,1,1,0,0,1,1,1,1,1,0,1,1], dtype = np.uint16)
    Generator24_LTEB      = np.array([1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1], dtype = np.uint16)


    # -------------------------------------------------
    @classmethod
    def ComputeCrc(cls
                 , GeneratorFunction
                 , InputBitVector) -> np.ndarray:

        # -------------------------------------------------------
        # Error checking
        assert isinstance(GeneratorFunction, np.ndarray)
        assert all([( x == 1 or x == 0 ) for x in GeneratorFunction]), 'The generator function must be binary'
        assert all([( x == 1 or x == 0)  for x in InputBitVector]),    'The InputBitVector must be binary'

        if isinstance(InputBitVector, list):
            InputBitVector = np.array(InputBitVector, dtype = np.uint16)
        else:
            assert isinstance(InputBitVector, np.ndarray), 'The InputBitVector must be either a list or of type np.ndarray'
            InputBitVector = InputBitVector.astype(np.uint16)

        # -----------------------------------------------------
        # Compute the cyclic redundancy check
        GeneratorSize = len(GeneratorFunction)-1
        
        TempMessage = np.hstack([InputBitVector, np.zeros(GeneratorSize, dtype = np.uint16)])
        for Index in range(0, len(InputBitVector)):
            Range = np.arange(Index, Index + GeneratorSize+1, 1, dtype = np.int32)
            if TempMessage[Index] == 1:
                TempMessage[Range] = np.mod(TempMessage[Range] + GeneratorFunction, 2)
            if np.sum(TempMessage) == 0:
                break
 
        return TempMessage[-GeneratorSize:]         









# ------------------------------------------------------------------
# > The Linear Feedback Shift Register Class
# ------------------------------------------------------------------
class CLinearFeedbackShiftRegister():
    """
    This class implements a linear feedback shift register. 
    """    
    # The following dictionary provides a map of taps configurations of maximal-length feedback polynomials for shift 
    # registers up to 16 bits. The feedback returns to the first delay element at the left of the shift register. There
    # are several tap configuration for each length. The dictionary below lists just one tap configuration that works.
    #
    # i.e.: A tap configuration of [1, 0, 1, 0, 0] => x^5 + x^3 + 1
    #
    # Which looks as follows:
    #           0   1   2   3   4     <- Shift Register index when implemented as np.zeros(Length, np.uint16)
    #           1   2   3   4   5     <- Connection Positions as shown in the tap configuration
    #          ___ ___ ___ ___ ___
    #      -> |___|___|___|___|___|
    #      |            |       | 
    #       ----------- + <-----
    #
    # The period at which the bits repeat is equal to 2^NumTaps - 1
    # Thus for the example above, x^5 + x^3 + 1, the sequence repeats every 2^5 - 1 = 31 clocks
    #
    TapDict = { 2  : [1, 1],                                                \
                3  : [1, 1, 0],                                             \
                4  : [1, 1, 0, 0],                                          \
                5  : [1, 0, 1, 0, 0],                                       \
                6  : [1, 1, 0, 0, 0, 0],                                    \
                7  : [1, 1, 0, 0, 0, 0, 0],                                 \
                8  : [1, 0, 1, 1, 1, 0, 0, 0],                              \
                9  : [1, 0, 0, 0, 1, 0, 0, 0, 0],                           \
                10 : [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],                        \
                11 : [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],                     \
                12 : [1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],                  \
                13 : [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],               \
                14 : [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],            \
                15 : [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],         \
                16 : [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] }     
                    
    # --------------------------------------
    # > The constructor
    # --------------------------------------
    def __init__(self
               , bUseMaxLengthTable: bool = True
               , IndexOrTapConfig:   int  = 5):     # Follows example above -> x^5 + x^3 + 1
        '''
        This function sets up a linear feedback shift register.
        '''

        assert isinstance(bUseMaxLengthTable, bool),                 'The bUseMaxLengthTable argument must be a bool'
        if bUseMaxLengthTable == True:
            assert isinstance(IndexOrTapConfig, int),                'The IndexOrTapConfig argument must be an integer'
            assert IndexOrTapConfig >= 2 and IndexOrTapConfig <= 16, 'The IndexOrTapConfig argument must be >= 2 and <= 16'

            self.TapConfiguration = CLinearFeedbackShiftRegister.TapDict[IndexOrTapConfig]
        else:
            assert isinstance(IndexOrTapConfig, list),               'The IndexOrTapConfig argument must be a list'
            assert all([isinstance(x, int) and (x == 0 or x == 1) for x in IndexOrTapConfig]), 'The IndexOrTapConfig argument must be composed of 0s and 1s'
            assert IndexOrTapConfig[0] == 1,                         'The first element in indexOrTapConfig must be a 1'

            self.TapConfiguration = IndexOrTapConfig

        
        self.ShiftRegister = np.zeros(len(self.TapConfiguration), np.uint16)

        # The tap configuration is a list of bits. We need a connection mask such that
        # OutputBit = np.sum(self.ShiftRegister * ConnectionMask) % 2
        self.ConnectionMask = np.flipud(np.array(self.TapConfiguration, np.uint16))

        self.IsInitialized = False


    # ----------------------------------------
    # > Initialize the shift register with values 
    # ----------------------------------------
    def InitializeShiftRegister(self
                              , BitList: list):
        '''
        This function will initialize the shift register with bits
        ''' 
        assert isinstance(BitList, list),                                         'The BitList argument must be a list'
        assert all([isinstance(x, int) and (x == 0 or x == 1) for x in BitList]), 'The BitList argument must be composed of 0s and 1s'
        assert any(x == 1 for x in BitList),                                      'At least one element in the bit list must be a 1'
        assert len(BitList) == len(self.ShiftRegister),                           'Initialization vector is not of right length'

        for Index in range(0, len(BitList)):
            self.ShiftRegister[Index] = np.uint16(BitList[Index])

        self.IsInitialized = True


    # ----------------------------------------
    # > Run Shift Register
    # ----------------------------------------
    def RunShiftRegister(self
                       , NumberOutputBits: int) -> np.ndarray:
        '''
        This function will run the shift registers for NumberOutputBits number of clocks.
        '''
        assert isinstance(NumberOutputBits, int), 'The NumberOutputBits argument must be an integer'
        assert NumberOutputBits > 0,              'The NumberOutputBits argument must be larger than 0'
        assert self.IsInitialized,                'The shift register is not initialized'

        Output = np.zeros(NumberOutputBits, np.uint16)

        for Index in range(0, NumberOutputBits):
            # Compute bit and store in output
            Bit                    = np.sum(self.ShiftRegister * self.ConnectionMask) % 2
            Output[Index]          = Bit

            # Update shift register
            self.ShiftRegister[1:] = self.ShiftRegister[0:-1]
            self.ShiftRegister[0]  = Bit

        return Output
    

















# ------------------------------------------------------------------------------------------------- #
#                                                                                                   #
#                                                                                                   #
# > CLdpcProcessor Class                                                                            #
#                                                                                                   #
#                                                                                                   #
# ------------------------------------------------------------------------------------------------- #
class CLdpcProcessor():
    """
    brief: This class provides LDPC encoding and decoding services for the FlexLink Specification 
    notes: See IEEE Std 802.11-2012 Annex F
    """
    # -------------------------------
    # Matrix Prototypes for codeword block length n = 648 bits, with subblock size Z = 27 bits
    # -------------------------------
    PrototypeM648_1_2 = np.array([[ 0, -1, -1, -1,  0,  0, -1, -1,  0, -1, -1,  0,  1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                                  [22,  0, -1, -1, 17, -1,  0,  0, 12, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                                  [ 6, -1,  0, -1, 10, -1, -1, -1, 24, -1,  0, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1], 
                                  [ 2, -1, -1,  0, 20, -1, -1, -1, 25,  0, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1], 
                                  [23, -1, -1, -1,  3, -1, -1, -1,  0, -1,  9, 11, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1], 
                                  [24, -1, 23,  1, 17, -1,  3, -1, 10, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1], 
                                  [25, -1, -1, -1,  8, -1, -1, -1,  7, 18, -1, -1,  0, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1], 
                                  [13, 24, -1, -1,  0, -1,  8, -1,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1], 
                                  [ 7, 20, -1, 16, 22, 10, -1, -1, 23, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1], 
                                  [11, -1, -1, -1, 19, -1, -1, -1, 13, -1,  3, 17, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1], 
                                  [25, -1,  8, -1, 23, 18, -1, 14,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0], 
                                  [ 3, -1, -1, -1, 16, -1, -1,  2, 25,  5, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0]], dtype = np.int8)

    PrototypeM648_2_3 = np.array([[25, 26, 14, -1, 20, -1,  2, -1,  4, -1, -1,  8, -1, 16, -1, 18,  1,  0, -1, -1, -1, -1, -1, -1], 
                                  [10,  9, 15, 11, -1,  0, -1,  1, -1, -1, 18, -1,  8, -1, 10, -1, -1,  0,  0, -1, -1, -1, -1, -1], 
                                  [16,  2, 20, 26, 21, -1,  6, -1,  1, 26, -1,  7, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1], 
                                  [10, 13,  5,  0, -1,  3, -1,  7, -1, -1, 26, -1, -1, 13, -1, 16, -1, -1, -1,  0,  0, -1, -1, -1], 
                                  [23, 14, 24, -1, 12, -1, 19, -1, 17, -1, -1, -1, 20, -1, 21, -1,  0, -1, -1, -1,  0,  0, -1, -1], 
                                  [ 6, 22,  9, 20, -1, 25, -1, 17, -1,  8, -1, 14, -1, 18, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1], 
                                  [14, 23, 21, 11, 20, -1, 24, -1, 18, -1, 19, -1, -1, -1, -1, 22, -1, -1, -1, -1, -1, -1,  0,  0], 
                                  [17, 11, 11, 20, -1, 21, -1, 26, -1,  3, -1, -1, 18, -1, 26, -1,  1, -1, -1, -1, -1, -1, -1,  0]], dtype = np.int8)

    PrototypeM648_3_4 = np.array([[16, 17, 22, 24,  9,  3, 14, -1,  4,  2,  7, -1, 26, -1,  2, -1, 21, -1,  1,  0, -1, -1, -1, -1], 
                                  [25, 12, 12,  3,  3, 26,  6, 21, -1, 15, 22, -1, 15, -1,  4, -1, -1, 16, -1,  0,  0, -1, -1, -1], 
                                  [25, 18, 26, 16, 22, 23,  9, -1,  0, -1,  4, -1,  4, -1,  8, 23, 11, -1, -1, -1,  0,  0, -1, -1], 
                                  [ 9,  7,  0,  1, 17, -1, -1,  7,  3, -1,  3, 23, -1, 16, -1, -1, 21, -1,  0, -1, -1,  0,  0, -1], 
                                  [24,  5, 26,  7,  1, -1, -1, 15, 24, 15, -1,  8, -1, 13, -1, 13, -1, 11, -1, -1, -1, -1,  0,  0], 
                                  [ 2,  2, 19, 14, 24,  1, 15, 19, -1, 21, -1,  2, -1, 24, -1,  3, -1,  2,  1, -1, -1, -1, -1,  0]], dtype = np.int8)

    PrototypeM648_5_6 = np.array([[17, 13,  8, 21,  9,  3, 18, 12, 10,  0,  4, 15, 19,  2,  5, 10, 26, 19, 13, 13,  1,  0, -1, -1],
                                  [ 3, 12, 11, 14, 11, 25,  5, 18,  0,  9,  2, 26, 26, 10, 24,  7, 14, 20,  4,  2, -1,  0,  0, -1],
                                  [22, 16,  4,  3, 10, 21, 12,  5, 21, 14, 19,  5, -1,  8,  5, 18, 11,  5,  5, 15,  0, -1,  0,  0],
                                  [ 7,  7, 14, 14,  4, 16, 16, 24, 24, 10,  1,  7, 15,  6, 10, 26,  8, 18, 21, 14,  1, -1, -1,  0]], dtype = np.int8)


    # -------------------------------
    # Matrix Prototypes for codeword block length n = 1296 bits, with subblock size Z = 54 bits
    # -------------------------------
    PrototypeM1296_1_2 = np.array([[40, -1, -1, -1, 22, -1, 49, 23, 43, -1, -1, -1,  1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                                   [50,  1, -1, -1, 48, 35, -1, -1, 13, -1, 30, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                                   [39, 50, -1, -1,  4, -1,  2, -1, -1, -1, -1, 49, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1], 
                                   [33, -1, -1, 38, 37, -1, -1,  4,  1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1], 
                                   [45, -1, -1, -1,  0, 22, -1, -1, 20, 42, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1], 
                                   [51, -1, -1, 48, 35, -1, -1, -1, 44, -1, 18, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1], 
                                   [47, 11, -1, -1, -1, 17, -1, -1, 51, -1, -1, -1,  0, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1], 
                                   [ 5, -1, 25, -1,  6, -1, 45, -1, 13, 40, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1], 
                                   [33, -1, -1, 34, 24, -1, -1, -1, 23, -1, -1, 46, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1], 
                                   [ 1, -1, 27, -1,  1, -1, -1, -1, 38, -1, 44, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1], 
                                   [-1, 18, -1, -1, 23, -1, -1,  8,  0, 35, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0], 
                                   [49, -1, 17, -1, 30, -1, -1, -1, 34, -1, -1, 19,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0]], dtype = np.int8)

    PrototypeM1296_2_3 = np.array([[25, 52, 41,  2,  6, -1, 14, -1, 34, -1, -1, -1, 24, -1, 37, -1, -1,  0,  0, -1, -1, -1, -1, -1], 
                                   [43, 31, 29,  0, 21, -1, 28, -1, -1,  2, -1, -1,  7, -1, 17, -1, -1, -1,  0,  0, -1, -1, -1, -1], 
                                   [20, 33, 48, -1,  4, 13, -1, 26, -1, -1, 22, -1, -1, 46, 42, -1, -1, -1, -1,  0,  0, -1, -1, -1], 
                                   [45,  7, 18, 51, 12, 25, -1, -1, -1, 50, -1, -1,  5, -1, -1, -1,  0, -1, -1, -1,  0,  0, -1, -1], 
                                   [35, 40, 32, 16,  5, -1, -1, 18, -1, -1, 43, 51, -1, 32, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1], 
                                   [ 9, 24, 13, 22, 28, -1, -1, 37, -1, -1, 25, -1, -1, 52, -1, 13, -1, -1, -1, -1, -1, -1,  0,  0], 
                                   [32, 22,  4, 21, 16, -1, -1, -1, 27, 28, -1, 38, -1, -1, -1,  8,  1, -1, -1, -1, -1, -1, -1,  0]], dtype = np.int8)

    PrototypeM1296_3_4 = np.array([[39, 40, 51, 41,  3, 29,  8, 36, -1, 14, -1,  6, -1, 33, -1, 11, -1,  4,  1,  0, -1, -1, -1, -1], 
                                   [48, 21, 47,  9, 48, 35, 51, -1, 38, -1, 28, -1, 34, -1, 50, -1, 50, -1, -1,  0,  0, -1, -1, -1], 
                                   [30, 39, 28, 42, 50, 39,  5, 17, -1,  6, -1, 18, -1, 20, -1, 15, -1, 40, -1, -1,  0,  0, -1, -1], 
                                   [29,  0,  1, 43, 36, 30, 47, -1, 49, -1, 47, -1,  3, -1, 35, -1, 34, -1,  0, -1, -1,  0,  0, -1], 
                                   [ 1, 32, 11, 23, 10, 44, 12,  7, -1, 48, -1,  4, -1,  9, -1, 17, -1, 16, -1, -1, -1, -1,  0,  0], 
                                   [13,  7, 15, 47, 23, 16, 47, -1, 43, -1, 29, -1, 52, -1,  2, -1, 53, -1,  1, -1, -1, -1, -1,  0]], dtype = np.int8)
     
    PrototypeM1296_5_6 = np.array([[48, 29, 37, 52,  2, 16,  6, 14, 53, 31, 34,  5, 18, 42, 53, 31, 45, -1, 46, 52,  1,  0, -1, -1],
                                   [17,  4, 30,  7, 43, 11, 24,  6, 14, 21,  6, 39, 17, 40, 47,  7, 15, 41, 19, -1, -1,  0,  0, -1],
                                   [ 7,  2, 51, 31, 46, 23, 16, 11, 53, 40, 10,  7, 46, 53, 33, 35, -1, 25, 35, 38,  0, -1,  0,  0],
                                   [19, 48, 41,  1, 10,  7, 36, 47,  5, 29, 52, 52, 31, 10, 26,  6,  3,  2, -1, 51,  1, -1, -1,  0]], dtype = np.int8)

    # -------------------------------
    # Matrix Prototypes for codeword block length n = 1944 bits, with subblock size Z = 81 bits
    # -------------------------------
    PrototypeM1944_1_2 = np.array([[57, -1, -1, -1, 50, -1, 11, -1, 50, -1, 79, -1,  1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                                   [ 3, -1, 28, -1,  0, -1, -1, -1, 55,  7, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                                   [30, -1, -1, -1, 24, 37, -1, -1, 56, 14, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1], 
                                   [62, 53, -1, -1, 53, -1, -1,  3, 35, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1], 
                                   [40, -1, -1, 20, 66, -1, -1, 22, 28, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1], 
                                   [ 0, -1, -1, -1,  8, -1, 42, -1, 50, -1, -1,  8, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1], 
                                   [69, 79, 79, -1, -1, -1, 56, -1, 52, -1, -1, -1,  0, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1], 
                                   [65, -1, -1, -1, 38, 57, -1, -1, 72, -1, 27, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1], 
                                   [64, -1, -1, -1, 14, 52, -1, -1, 30, -1, -1, 32, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1], 
                                   [-1, 45, -1, 70,  0, -1, -1, -1, 77,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1], 
                                   [ 2, 56, -1, 57, 35, -1, -1, -1, -1, -1, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0], 
                                   [24, -1, 61, -1, 60, -1, -1, 27, 51, -1, -1, 16,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0]], dtype = np.int8)

    PrototypeM1944_2_3 = np.array([[61, 75,  4, 63, 56, -1, -1, -1, -1, -1, -1,  8, -1,  2, 17, 25,  1,  0, -1, -1, -1, -1, -1, -1], 
                                   [56, 74, 77, 20, -1, -1, -1, 64, 24,  4, 67, -1,  7, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1], 
                                   [28, 21, 68, 10,  7, 14, 65, -1, -1, -1, 23, -1, -1, -1, 75, -1, -1, -1,  0,  0, -1, -1, -1, -1], 
                                   [48, 38, 43, 78, 76, -1, -1, -1, -1,  5, 36, -1, 15, 72, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1], 
                                   [40,  2, 53, 25, -1, 52, 62, -1, 20, -1, -1, 44, -1, -1, -1, -1,  0, -1, -1, -1,  0,  0, -1, -1], 
                                   [69, 23, 64, 10, 22, -1, 21, -1, -1, -1, -1, -1, 68, 23, 29, -1, -1, -1, -1, -1, -1,  0,  0, -1], 
                                   [12,  0, 68, 20, 55, 61, -1, 40, -1, -1, -1, 52, -1, -1, -1, 44, -1, -1, -1, -1, -1, -1,  0,  0], 
                                   [58,  8, 34, 64, 78, -1, -1, 11, 78, 24, -1, -1, -1, -1, -1, 58,  1, -1, -1, -1, -1, -1, -1,  0]], dtype = np.int8)

    PrototypeM1944_3_4 = np.array([[48, 29, 28, 39,  9, 61, -1, -1, -1, 63, 45, 80, -1, -1, -1, 37, 32, 22,  1,  0, -1, -1, -1, -1], 
                                   [ 4, 49, 42, 48, 11, 30, -1, -1, -1, 49, 17, 41, 37, 15, -1, 54, -1, -1, -1,  0,  0, -1, -1, -1], 
                                   [35, 76, 78, 51, 37, 35, 21, -1, 17, 64, -1, -1, -1, 59,  7, -1, -1, 32, -1, -1,  0,  0, -1, -1], 
                                   [ 9, 65, 44,  9, 54, 56, 73, 34, 42, -1, -1, -1, 35, -1, -1, -1, 46, 39,  0, -1, -1,  0,  0, -1], 
                                   [ 3, 62,  7, 80, 68, 26, -1, 80, 55, -1, 36, -1, 26, -1,  9, -1, 72, -1, -1, -1, -1, -1,  0,  0], 
                                   [26, 75, 33, 21, 69, 59,  3, 38, -1, -1, -1, 35, -1, 62, 36, 26, -1, -1,  1, -1, -1, -1, -1,  0]], dtype = np.int8)

    PrototypeM1944_5_6 = np.array([[13, 48, 80, 66,  4, 74,  7, 30, 76, 52, 37, 60, -1, 49, 73, 31, 74, 73, 23, -1,  1,  0, -1, -1],
                                   [69, 63, 74, 56, 64, 77, 57, 65,  6, 16, 51, -1, 64, -1, 68,  9, 48, 62, 54, 27, -1,  0,  0, -1],
                                   [51, 15,  0, 80, 24, 25, 42, 54, 44, 71, 71,  9, 67, 35, -1, 58, -1, 29, -1, 53,  0, -1,  0,  0],
                                   [16, 29, 36, 41, 44, 56, 59, 37, 50, 24, -1, 65,  4, 65, 52, -1,  4, -1, 73, 52,  1, -1, -1,  0]], dtype = np.int8)
    
    # --------------------------------------------------------------------------------
    # > This function expands the prototype matrix into a parity check matrix
    # --------------------------------------------------------------------------------
    @classmethod
    def CreateParityCheckMatrix(cls
                              , iBlockLength: int = 648
                              , strRate:      str = '1/2') -> np.ndarray:
        """
        brief: This function will create a parity check matrix from one of the available prototype matrices
        param: iBlockLength  - Currently we have a choice of 648, 1296 and 1944
        param: strRate       - Currently we have a choice of '1/2', '2/3', '3/4, '5/6'
        """
        
        # Error checking
        assert isinstance(iBlockLength, int),                                       'The iBlockLength input type is not int.'
        assert isinstance(strRate, str),                                            'The strRate input typ is not str.'
        assert iBlockLength == 648 or iBlockLength == 1296 or iBlockLength == 1944, 'The block length must be 648 or 1296 or 1944.'
        assert strRate == '1/2' or strRate == '2/3' or strRate == '3/4' or strRate == '5/6', \
                                                                           'The strRate input arguement must be 1/2, 2/3, 3/4 or 5/6'

        # Determine current prototype matrix
        match iBlockLength:
            case 648:
                if strRate == '1/2': PrototypeMatrix = cls.PrototypeM648_1_2
                if strRate == '2/3': PrototypeMatrix = cls.PrototypeM648_2_3
                if strRate == '3/4': PrototypeMatrix = cls.PrototypeM648_3_4
                if strRate == '5/6': PrototypeMatrix = cls.PrototypeM648_5_6
                SubmatrixSize = 27
            case 1296:
                if strRate == '1/2': PrototypeMatrix = cls.PrototypeM1296_1_2
                if strRate == '2/3': PrototypeMatrix = cls.PrototypeM1296_2_3
                if strRate == '3/4': PrototypeMatrix = cls.PrototypeM1296_3_4
                if strRate == '5/6': PrototypeMatrix = cls.PrototypeM1296_5_6
                SubmatrixSize = 54
            case 1944:
                if strRate == '1/2': PrototypeMatrix = cls.PrototypeM1944_1_2
                if strRate == '2/3': PrototypeMatrix = cls.PrototypeM1944_2_3
                if strRate == '3/4': PrototypeMatrix = cls.PrototypeM1944_3_4
                if strRate == '5/6': PrototypeMatrix = cls.PrototypeM1944_5_6
                SubmatrixSize = 81
            case _:
                assert False, 'An error has occured.'

        # Build the parity check matrix, H
        Rows, Columns = PrototypeMatrix.shape
        H = np.zeros([Rows * SubmatrixSize, Columns * SubmatrixSize], dtype = np.uint16)

        for row in range(0, Rows):
            for column in range(0, Columns):
                # The cyclic shift
                iCyclicShift = PrototypeMatrix[row, column]
                
                if iCyclicShift == -1:    # Then do not insert the eye matrix
                    continue

                # Create the eye matrix and cyclically shift it
                Eye        = np.eye(SubmatrixSize, dtype = np.uint16)
                EyeShifted = np.roll(Eye, iCyclicShift, axis = 1)   # Cyclic shift of each row

                # Figure out the row/column ranges inside the H matrix where we want to insert the Eye matrix
                RowStart    = row    * SubmatrixSize
                RowStop     = RowStart + SubmatrixSize
                ColumnStart = column * SubmatrixSize 
                ColumnStop  = ColumnStart + SubmatrixSize

                # Insert the Eye matrix
                H[RowStart:RowStop, ColumnStart:ColumnStop] = EyeShifted

        # Return the parity check matrix
        return H




    # --------------------------------------------------------------------------------
    # > This function transform the parity check matrix via the Gauss-Jordan Elimination
    # --------------------------------------------------------------------------------
    @staticmethod
    def TransformParityCheckMatrix(H) -> np.ndarray:
        """
        Transform the parity check matrix into a form that can be used to create the generator matrix
        """
        H_New = H.copy()
        rows, columns = H_New.shape
        N             = columns   # The number of encoded bits
        L             = rows      # The number of parity bits
        K             = N - L     # The number of message bits

        # ------------------------------------
        # Step 1: The forward elimination step
        for column in range(K, N):
            # [RowOfInterest, column] are the coordinates of the diagonal of the LxL square matrix all
            # the way to the right of H_New. The point of the exercise is to place 1's in these positions. 
            # If we don't find a 1 there, we need to look to see whether one of the rows below has a
            # one in this column and then pivot.
            RowOfInterest = column - K    # Increments as follows: 0, 1, 2, 3, 4, ... L - 1

            # 1a. Execute a pivot step if we need it
            if H_New[RowOfInterest, column] != 1:
                # We will now look at the rows below until we find a 1 in this column.
                # Rather than switching the current row with the one containing the 1,
                # we add the row below to the current on. This is fine as we are
                # doing modulo 2 addition. We could have switched them as well.
                RemainingRowsBelow = L - RowOfInterest -1
                bPivotSuccessful  = False
                if RemainingRowsBelow > 0:
                    # We attempt to pivot rows
                    for row in range(RowOfInterest+1, L):
                        if H_New[row, column] == 1:
                            # Modulo two addition of RowOfInterest and row
                            H_New[RowOfInterest, :] = np.remainder(H_New[RowOfInterest, :] + H_New[row,:], 2)
                            bPivotSuccessful = True
                            break
                else:
                    assert bPivotSuccessful, 'The Pivot operation failed. Supply a proper H matrix.' 

            # 1b. Execute the forward elimination step
            #     At the point, we have ensured that the diagonal of the square matrix on the right 
            #     of H_New features ones everywhere
            for row in range(RowOfInterest+1, L):
                if H_New[row, column] == 1:
                    # Modulo two addition of RowOfInterest and row
                    H_New[row, :] = np.remainder(H_New[RowOfInterest, :] + H_New[row,:], 2)

        # ------------------------------------
        # Step 2: The backward elimination step
        for column in range(N-1, K, -1):
            RowOfInterest = column - K   # Decrement as follows: L, L-1, L-2, L-3 ... 2
            # If any of the rows above feature a 1 in this column, get rid of it by adding
            for row in range(0, RowOfInterest):
                if H_New[row, column] == 1:
                    # Modulo two addition of RowOfInterest and row
                    H_New[row, :] = np.remainder(H_New[RowOfInterest, :] + H_New[row,:], 2)

        return H_New



    # ----------------------------------------------------------------------------
    # > This function will compute the generator matrix G
    # ----------------------------------------------------------------------------
    @staticmethod
    def ComputeGeneratorMatrix(H) -> np.ndarray:
        """
        This function computes the generator matrix, G, of a paritycheck matrix, H. 
        """
        rows, columns = H.shape
        N             = columns   # The number of encoded bits
        L             = rows      # The number of parity bits
        K             = N - L     # The number of message bits

        H_Modified = CLdpcProcessor.TransformParityCheckMatrix(H)
        P          = H_Modified[0:L, 0:K]
        PT         = np.transpose(P)
        G          = np.hstack([np.eye(K, dtype = P.dtype), PT])

        # We can verify that the generator matrix was properly computed
        GHT        = np.matmul(G, np.transpose(H))
        GHT_MOD2   = np.remainder(GHT, 2)
        TotalSum   = np.sum(GHT_MOD2[:])
        assert TotalSum == 0, 'The generator was not computed properly.'

        return G



    # ----------------------------------------------------------------------------
    # > This function will compute the SISO Single Parity Check
    # ----------------------------------------------------------------------------
    @staticmethod
    def SISO_SPC_Decoder(rn: np.ndarray) -> np.ndarray:
        '''
        brief: This function implements the single input single output single parity check operation.
        param: rn - This is the vector of intrinsic beliefs at the bit nodes
        var:   l  - This is the vector of extrinsic beliefs for the single paraity check node at index x
        '''
        # Type checking
        assert isinstance(rn, np.ndarray), 'The input rn must be of type nd.ndarray'
        assert np.issubdtype(rn.dtype, np.floating), 'The entries in rn must be floating point numbers'

        # Run the decoder
        l       = np.zeros(len(rn), dtype = rn.dtype) 
        for Index, r in enumerate(rn):
            r_other    = np.delete(rn.copy(), Index)                # Remove the current intrinsic belief from r
            sign_other = -np.sign(np.prod(-r_other))     
            mag_other  = np.min(abs(r_other))
            l[Index]   = sign_other * mag_other            
            
        # Return the new beliefs
        return l





    # ----------------------------------------------------------------------------
    # > The constuctor for a CLdpcProcessor object
    # ----------------------------------------------------------------------------
    def __init__(self
               , strMode:          str = 'WLAN'                         # 'WLAN' or 'CUSTOM'
               , H_Custom:         np.ndarray = np.array([0], np.uint16)       
               , OutputBlockSize:  int = 648
               , strRate:          str = '1/2') -> np.ndarray:

        # Error checking
        assert strMode.lower() == 'wlan' or strMode.lower() == 'custom', 'The strMode argument must be either "wlan" or "custom"'

        if strMode.lower() == 'custom':
            # At this point we only care about the H_Custom argument
            assert isinstance(H_Custom, np.ndarray), 'The H_Custom input argument must be an np.ndarray'
            self.H       = H_Custom
        else:
            # Error checking
            assert isinstance(OutputBlockSize, int),                                             'The iBlockLength input type is not int.'
            assert isinstance(strRate, str),                                                     'The strRate input typ is not str.'
            assert OutputBlockSize == 648 or OutputBlockSize == 1296 or OutputBlockSize == 1944, 'The block length must be 648 or 1296 or 1944.'
            assert strRate == '1/2' or strRate == '2/3' or strRate == '3/4',                     'The strRate input arguement must be 1/2, 2/3, or 3/4.'

            # Record Input arguments
            match strRate:
                case '1/2':
                    self.Rate = 0.5
                case '2/3':  
                    self.Rate = 2/3
                case '3/4':
                    self.Rate = 3/4
                case '5/6':
                    self.Rate = 5/6

            # The number of input bits in a single transport block. Those input bits that can be encoded as one group.
            self.TransportBlockSize = int(OutputBlockSize * self.Rate)

            # The number of output (encoded) bits resulting from the bits in one transport block
            self.OutputBlockSize    = OutputBlockSize
        
            # Create the Parity Check and Generator Matrices 
            self.H                  = CLdpcProcessor.CreateParityCheckMatrix(OutputBlockSize, strRate)
        
        
        # Create the generator matrix
        self.G           = CLdpcProcessor.ComputeGeneratorMatrix(self.H)
        
        rows, columns = self.H.shape
        self.N = self.NumEncodedBits = columns             # The number of encoded bits
        self.L = self.NumParityBits  = rows                # The number of parity bits
        self.K = self.NumMessageBits = self.N - self.L     # The number of message bits

        


    # ----------------------------------------------------------------------------
    # > This function will LDPC encode a vector of input data bits
    # ----------------------------------------------------------------------------
    def EncodeBits(self
                 , InputBits:         np.ndarray
                 , strSoftbitMapping: str  = 'hard'                
                 , bAllowZeroPadding: bool = False) -> np.ndarray:
        """
        brief: This function will create a parity check matrix from one of the available prototype matrices
        param: InputBits         - The input bit vector 
        param: strSoftbitMapping - 'hard' -> 0/1, 'inverting' -> maps 0/1 to 1/-1, 'non-inverting' -> maps 0/1 to -1/1
        param: bAllowZeroPadding - If true, then the input bit vector may be of a size != self.BlockLength 
        """
        
        # ------------------------------
        # Error Checking
        assert isinstance(InputBits, np.ndarray),          'The InputBits input argument must be of type np.ndarray'
        assert np.issubdtype(InputBits.dtype, np.integer), 'The InputBits input argument must be an array of integers'
        assert isinstance(strSoftbitMapping, str)        , 'The strBitMode input argument must be a string'
        assert strSoftbitMapping.lower() == 'hard' or strSoftbitMapping.lower() == 'inverting' \
                                                   or strSoftbitMapping.lower() == 'non-inverting', \
                                                           'The strBitMode input argument is invalid'
        assert isinstance(bAllowZeroPadding, bool),        'The bAllowZeroPadding input arbument must be of type bool'
        assert len(InputBits.shape) == 1 or len(InputBits.shape) == 2, 'The shape of the InputBits input argument is invalid.'

        # The input should be a simple array with shape (NumInputBits, ), not a matrix of dimensions with shape (1, NumInputBits)
        if len(InputBits.shape) == 2:   # Then we must convert the input matrix into a simple array
            InputBits = InputBits[0, :]

        bProperBits = all([((x==0 or x==1)) for x in InputBits])
        assert bProperBits,   'The input bits must be either 0s or 1s.'

        if bAllowZeroPadding == False and len(InputBits) % self.TransportBlockSize != 0:
            assert False, 'The InputBits vector size must be a integer multiple of the transport block size.'
        
        
        # ---------------------------------------------------------------------------
        # Determine number of transport blocks (The number of input bits per block)
        NumberOfInputBits          = len(InputBits)
        if NumberOfInputBits % self.TransportBlockSize > 0:
            NumPaddingBits         = self.TransportBlockSize - NumberOfInputBits % self.TransportBlockSize
        else:
            NumPaddingBits         = 0
            
        ZeroPaddedInputBits        = np.append(InputBits, np.zeros(NumPaddingBits, dtype = InputBits.dtype))
        FinalNumberOfInputBits     = len(ZeroPaddedInputBits)

        assert FinalNumberOfInputBits % self.TransportBlockSize == 0, 'Zero Padding was unsuccessful'
        self.NumberTransportBlocks = int(FinalNumberOfInputBits/self.TransportBlockSize)

        # Allocate memory for the output vector
        NumberOfOutputBits         = self.NumberTransportBlocks * self.OutputBlockSize
        OutputBits                 = np.zeros([1, NumberOfOutputBits], np.int8)

        # Encode the Input Bits
        # A place holder for the transport block
        TransportBlock      = np.zeros([1, self.TransportBlockSize], dtype = ZeroPaddedInputBits[0].dtype)
        for BlockIndex in range(0, self.NumberTransportBlocks):
            InputStartIndex     = BlockIndex * self.TransportBlockSize
            TransportBlock[0,:] = ZeroPaddedInputBits[InputStartIndex:InputStartIndex + self.TransportBlockSize]
            OutputBitBlock      = np.mod(np.matmul(TransportBlock, self.G), 2)

            OutputStartIndex                                                        = BlockIndex * self.OutputBlockSize
            OutputBits[0, OutputStartIndex:OutputStartIndex + self.OutputBlockSize] = OutputBitBlock


        if strSoftbitMapping == 'hard':
            return OutputBits[0]
        elif strSoftbitMapping == 'inverting':         # Here we map bit 0 to +1 and bit 1 to -1
            InvertedOutputBits = -(2*OutputBits[0] - 1)
            return InvertedOutputBits
        else:                                          # Here we map bit 0 to -1 and bit 1 to +1
            NonInvertedOutputBits = 2*OutputBits[0] - 1 
            return NonInvertedOutputBits





    # ----------------------------------------------------------------------------
    # > This is the LDPC Message Passing Decoder procedure
    # ----------------------------------------------------------------------------
    def DecodeBits(self
                 , InputBeliefs:      np.ndarray
                 , NumberIterations:  int = 8
                 , strSoftbitMapping: str = 'non-inverting') -> np.ndarray:
        
        '''
        brief: This is the LDPC Message Passing Decoder
        param: InputBeliefs      -  A vector of LLR bit beliefs
        param: NumberIteration   -  Self explanatory
        param: strSoftbitMapping -  Softbit to hardbit mapping ('non-inverting' = -1/1 maps to 0/1   --- 'inverting' = -1/1 maps to 1/0)
                                    IEEE - is 'non-inverting', 3GPP - is 'inverting'
        '''
        # --------------------------------------
        # Type checking 
        assert isinstance(InputBeliefs, np.ndarray),           'The InputBeliefs input argument must be of type np.ndarray'
        assert np.issubdtype(InputBeliefs.dtype, np.floating), 'The InputBeliefs input argument must be an array of floating point values'
        assert isinstance(NumberIterations, int),              'The NumberIterations input argument must be an integer.'
        assert isinstance(strSoftbitMapping, str),             'The "strSoftbitMapping" input argument must be of type str'
        
        # --------------------------------------
        # Error checking
        assert strSoftbitMapping == 'non-inverting' or strSoftbitMapping == 'inverting', 'The strSoftbitMapping input argument is invalid'
        assert len(InputBeliefs) % self.NumEncodedBits == 0,   'The number of input bits must be an integer multiple of self.NumEncodedBits'

        # --------------------------------------
        # Copy and reformat the input beliefs if necessary
        if strSoftbitMapping == 'inverting':
            IntrinsicBeliefs = -InputBeliefs.copy()
        else:
            IntrinsicBeliefs =  InputBeliefs.copy()

        # ---------------------------------------
        # Find the number of LDPC decoding operations are necessary for the entire input belief vector
        NumLdpcRepetitions = int(len(InputBeliefs) / self.NumEncodedBits)
        RxBitEstimates     = np.zeros(NumLdpcRepetitions * self.NumMessageBits, np.uint16)

        # Iterate through each Ldpc Process
        for Repetition in range(0, NumLdpcRepetitions):
            StartIndexBeliefs   = Repetition*self.NumEncodedBits
            StartIndexMessage   = Repetition*self.NumMessageBits
            IntrinsicBeliefTemp = IntrinsicBeliefs[StartIndexBeliefs:StartIndexBeliefs + self.NumEncodedBits]
            CurrentBeliefs      = IntrinsicBeliefTemp

            E                = np.zeros(self.H.shape, InputBeliefs.dtype)
            for Interation in range(0, NumberIterations):
                # --------------------------
                # Step 1 and 4
                M = np.ones(self.H.shape, InputBeliefs.dtype)
                for row in range(0, self.H.shape[0]):
                    M[row, :] *= CurrentBeliefs

                M = M * self.H                 # Zero out positions that are not interesting to us
                                               # This helps when we add values column wise to find the total extrinsic belief
                M = M - E                      # During the first pass E == 0 (Step 1)
                                               # During later passes,  E != 0 (Step 4)

                # --------------------------
                # Step 2 and 5
                for CheckNodeIndex in range(0, self.L):
                    # Find the indices of non-zero entries in row 'CheckNodeIndex' of the parity check matrix H
                    CheckNodeConnectionIndices = np.nonzero(self.H[CheckNodeIndex,:])
                    # Fetch the extrinsic beliefs at that row and send to SISO_SPC_Decoder
                    rn     = M[CheckNodeIndex, CheckNodeConnectionIndices]
                    l      = CLdpcProcessor.SISO_SPC_Decoder(rn[0])
                    # This matrix remembers the intrinsic beliefs that we need to subtract later
                    E[CheckNodeIndex, CheckNodeConnectionIndices] = l
                

                # Step 4:
                # Find the sum of the extrinsic beliefs for a particular bit node.
                TotalExtrinsicBeliefs = np.sum(E, axis=0)
       
                # Add that sum to the intrinsic belief (original received bit beliefs) to get
                # the new updated intrinsic belief = r[x]new in the text.
                CurrentBeliefs      = IntrinsicBeliefTemp + TotalExtrinsicBeliefs

            # Map the received beliefs back to bits
            RxBitEstimates[StartIndexMessage:StartIndexMessage + self.NumMessageBits] = 0.5 * (np.sign(CurrentBeliefs[0:self.K]) + 1)

        # Recast to a reasonable type
        RxBitEstimates = RxBitEstimates.astype(np.uint16)
        return RxBitEstimates


































# ------------------------------------------------------------------------------------------------- #
#                                                                                                   #
#                                                                                                   #
# > CPolarProcessor Class                                                                           #
#                                                                                                   #
#                                                                                                   #
# ------------------------------------------------------------------------------------------------- #
class CPolarProcessor():
    """
    brief: This class provides polar encoding and decoding services for the FlexLink Specification 
    """

    # ------------------------------------------
    # FindChannelProbabilities()
    # ------------------------------------------
    @staticmethod
    def FindChannelProbabilities( N: int
                                , p: float)->np.ndarray:
        """
        brief: This function finds the probability of correct decoding for each input bit of the Polar code
        param: N - The number of bits in the polar code (N = 2**n)
        param: p - The original erasure probability of a single unencoded bit
        """
        # -------------------------------------
        # Error checking
        # -------------------------------------
        assert isinstance(N, int),   'The number of bits in the polar code must be an integer value.'
        K = np.log2(N) 
        assert np.fmod(K, 1.0) < 1e-4, 'The number of bits N must be equal to 2**n, where n is an integer > 0.'
        assert isinstance(p, float), 'The original erasure probability must be a floating point number.'
        
        # -------------------------------------
        # Start
        # -------------------------------------
        NumberOfStages = int(np.log10(N) / np.log10(2))

        # The ChannelErasureProbabilities Matrix will hold the erasure probabilities, where the
        # last column holds the erasure probabilities of the original 2^N channels W, whereas
        # column 1 will contain the erasure probabilities of the final polarized channels.  
        ChannelErasureProbabilities = p * np.ones([N, NumberOfStages+1], np.float32);

        # Iterate through each stage
        for StageIndex in range(NumberOfStages-1,-1,-1):
            for ChannelIndex in range(0, N):
                pPreviousChannel  = ChannelErasureProbabilities[ChannelIndex, StageIndex+1]
                MakeChannelBetter = int(math.floor(ChannelIndex/ (2**StageIndex))) % 2 == 1 
                if(MakeChannelBetter == True):
                    ChannelErasureProbabilities[ChannelIndex, StageIndex] = pPreviousChannel**2
                else:
                    ChannelErasureProbabilities[ChannelIndex, StageIndex] = 2*pPreviousChannel - pPreviousChannel**2

        return ChannelErasureProbabilities[:, 0].flatten()        
    


    # -------------------------------------------------------------------------------
    # RunPolarEncoder()
    # -------------------------------------------------------------------------------
    @staticmethod
    def RunPolarEncoder(InputVector: np.ndarray) -> np.ndarray:
        '''
        brief: This function executes the polar encoding process
        param: InputVector - An np.ndarray of input bits
        '''
        # -------------------------------------
        # Error checking
        # -------------------------------------
        assert isinstance(InputVector, np.ndarray),          'InputVector must be a numpy array.'
        assert len(InputVector.shape) == 1,                  'InputVector must be a one dimensional array'
        assert np.issubdtype(InputVector.dtype, np.integer), 'InputVector must be an integer array'
        bProperBits = all([(x == 1 or x == 0) for x in InputVector]) 
        assert bProperBits, 'The InputVector must be composed of 1s and 0s'
        N = len(InputVector)
        K = np.log2(N) 
        assert np.fmod(K, 1.0) < 1e-4, 'The number of bits in the InputVetor must be equal to 2**n, where n is an integer > 0.'
        
        # -------------------------------------
        # Start
        # -------------------------------------
        NumberOfStages = int(np.log10(N) / np.log10(2))
        
        # The matrix, M, shows the progression of the input bits through the Polar structure.
        M = np.zeros([N, NumberOfStages + 1], np.uint16)
        M[:, 0] = InputVector;                               # Load input vector into first column 
        
        # The Polar Encoding Process
        for StageIndex in range(1, NumberOfStages+1):
            NumberInputPerNode = 2**(StageIndex)              #    2,   4,   8,   16
            NumberOfNodes      = int(N / NumberInputPerNode)  #  N/2, N/4, N/8, N/16
            for NodeIndex in range(0, NumberOfNodes):
                U1_StartIndex               = 2**(StageIndex) * NodeIndex 
                U1_StopIndex                = U1_StartIndex + int(NumberInputPerNode/2)
                U2_StartIndex               = U1_StopIndex 
                U2_StopIndex                = U2_StartIndex + int(NumberInputPerNode/2)
                U1_Indices                  = np.arange(U1_StartIndex, U1_StopIndex, 1, np.int32)
                U2_Indices                  = np.arange(U2_StartIndex, U2_StopIndex, 1, np.int32)
                U1                          = M[U1_Indices, StageIndex-1]
                U2                          = M[U2_Indices, StageIndex-1]
                X1                          = np.remainder(U1+U2, 2)
                X2                          = U2
                M[U1_Indices, StageIndex]   = X1
                M[U2_Indices, StageIndex]   = X2

        return M[:, NumberOfStages].flatten()







    # ---------------------------------------------------------------------------------------
    # RunPolarDecoder()
    # ---------------------------------------------------------------------------------------
    @staticmethod
    def RunPolarDecoder(InputSequence: np.ndarray
                      , FrozenIndices: np.ndarray  ) -> np.ndarray:
        '''
        brief: This function runs the polar decoder
        param: InputVector   - The input vector of received beliefs of size 2**n, where n is a non-zero positive integer
        param: FrozenIndices - This vector indicates at which locations the frozen zeros are located. 
        '''
        assert isinstance(InputSequence, np.ndarray)
        assert np.issubdtype(InputSequence.dtype, np.floating)
        assert isinstance(FrozenIndices, np.ndarray)
        assert np.issubdtype(FrozenIndices.dtype, np.integer)
        assert len(InputSequence.shape) == 1, 'The InputSequence must be a 1D vector.'
        assert len(FrozenIndices.shape) == 1, 'The FrozenIndices must be a 1D vector.'


        # ---------------------------------------------------------
        # A lambda (local function)
        # ---------------------------------------------------------
        def MinSumDecoder(Y1, Y2):
            '''
            brief: This function computes the output believe of the MinSum decoder
            param: Y1 - One of two input beliefs (needs to be a 1D numpy array)
            param: Y2 - One of two input beliefs (needs to be a 1D numpy array)
            '''
            assert isinstance(Y1, np.ndarray)
            assert isinstance(Y2, np.ndarray) 
            assert len(Y1.shape) == 1, 'The Y1 input argument must be a 1D numpy array'
            assert len(Y2.shape) == 1, 'The Y2 input argument must be a 1D numpy array'
            assert len(Y1) == len(Y2), 'The Y1 and Y2 inputs must be of the same length'
            assert np.issubdtype(Y1.dtype, np.floating), 'The Y1 input argument must be a numpy floating point variable'
            assert np.issubdtype(Y2.dtype, np.floating), 'The Y2 input argument must be a numpy floating point variable'

            OutputBelief = np.zeros(len(Y1), np.float32)
            for Index in range(0, len(Y1)):
                OutputBelief[Index] = -np.sign(Y1[Index]) * np.sign(Y2[Index]) * np.fmin(np.abs(Y1[Index]), np.abs(Y2[Index]))
            return OutputBelief
        

        # ---------------------------------------------------------
        # A lambda (local function)
        # ---------------------------------------------------------
        def RepetitionDecoder(Y1, Y2, U):
            '''
            brief: This function computes the belief produced by the repetition decoder
            param: Y1 - One of two input beliefs
            param: Y2 - One of two input beliefs
            param: U  - A hard decision bit computed during the MinSum decoding
            '''
            assert isinstance(Y1, np.ndarray)
            assert isinstance(Y2, np.ndarray)
            assert isinstance(U,  np.ndarray)
            assert len(Y1.shape) == 1, 'The Y1 input argument must be a 1D numpy array'
            assert len(Y2.shape) == 1, 'The Y2 input argument must be a 1D numpy array'
            assert len(U.shape)  == 1, 'The  U input argument must be a 1D numpy array'
            assert len(Y1) == len(Y2), 'The Y1 and Y2 inputs must be of the same length'
            assert len(Y1) == len(U),  'The Y1 and U2 inputs must be of the same length'
            assert np.issubdtype(Y1.dtype, np.floating), 'The Y1 input argument must be a numpy floating point variable'
            assert np.issubdtype(Y2.dtype, np.floating), 'The Y2 input argument must be a numpy floating point variable'
            assert np.issubdtype(U.dtype,  np.integer),  'The  U input argument must be a numpy integer point variable'

            OutputBelief = np.zeros(len(U), np.float32)
            for Index in range(0, len(U)):
                if(U[Index] == 0):
                    OutputBelief[Index] = Y2[Index] + Y1[Index]
                else:
                    OutputBelief[Index] = Y2[Index] - Y1[Index]

            return OutputBelief


        # ---------------------------------------------------------
        # A lambda (local function)
        # ---------------------------------------------------------
        def GetNextNode(CurrentNode:  int
                      , CurrentStage: int
                      , bMovingLeft:  bool                   # Moving left in the tree  
                      , bMinSumDone:  bool) -> tuple:        # True / False - MinSum (has / has not) yet been computed for this node
            '''
            brief: This function provides the next stage and node in tree diagram given the input arguments
            param: bMovingLeft - We are either moving left in the tree (minsum or repetition decoding will be done) or right for re-encoding
            param: bMinSumDone - True (MinSum is done, do repetition decoding) / False (nothing has been done, do minsum decoding)
            '''
            assert isinstance(CurrentNode, int)
            assert isinstance(CurrentStage, int)
            assert isinstance(bMovingLeft, bool)
            assert isinstance(bMinSumDone, bool)
            # If bMovingLeft, then we are decoding and computing beliefs
            if(bMovingLeft == True):               # Either minsum or repetition decoding will be done
                NextStage = CurrentStage -1
                if(bMinSumDone == False):          # No processing of any kind has been done for this node
                    NextNode  = CurrentNode * 2
                else:                              # MinSum processing is done for this node
                    NextNode  = CurrentNode * 2 + 1
            else:                                  # re-encoding will be done
                NextStage = CurrentStage + 1
                NextNode  = int(math.floor(CurrentNode / 2))

            return (NextStage, NextNode)


        # -------------------------------------------------
        # A lambda (local function)
        # -------------------------------------------------
        def GetPortIndices(CurrentNode:  int
                         , CurrentStage: int) -> tuple:
            '''
            brief: This function determines the port indices for the given node and stage
            '''
            assert isinstance(CurrentNode, int)
            assert isinstance(CurrentStage, int)

            NumberOfPorts  = 2**CurrentStage
            StartPort      = CurrentNode * NumberOfPorts
            AllIndices     = np.arange(StartPort, StartPort + NumberOfPorts)
            UpperIndices  = AllIndices[:int(NumberOfPorts/2)]
            LowerIndices  = AllIndices[int(NumberOfPorts/2):]

            return (AllIndices, UpperIndices, LowerIndices)
        


        # -------------------------------------------------
        # A lambda (local function)
        # -------------------------------------------------
        def Debugging(NodeStateList: list
                    , L: np.ndarray
                    , B: np.ndarray):
            
            assert isinstance(NodeStateList, list)
            assert isinstance(L, np.ndarray)
            assert isinstance(B, np.ndarray)

            print('NodeStateList: '); print(NodeStateList)
            print('Matrix L:');       print(L)
            print('Matrix B:');       print(B)
        

        # ---------------------------------------------------------------------------------------------------------------------
        # Set up the Simplified Successive Decoding Process
        # ---------------------------------------------------------------------------------------------------------------------
        N        = len(InputSequence)
        K        = np.log2(N) 
        assert np.fmod(K, 1.0) < 1e-4, 'The number of bits in the InputVetor must be equal to 2**n, where n is an integer > 0.'
        NumberOfStages = int(np.log10(N) / np.log10(2))

        # Here we set the state of each node. Notice that there are more nodes as we move left
        # 0 - No processing has been done
        # 1 - MinSum     decoding has been done
        # 2 - Repetition decoding has been done
        # 3 - Re-encoding has been done
        NodeStateList = []
        for StageIndex in range(1, NumberOfStages + 1):
            NodeStateList.append(np.zeros(2**(NumberOfStages - StageIndex), np.uint16))

        # Define the belief matrix L and place the received beliefs into the right-most column
        L       = np.zeros([N, NumberOfStages + 1], np.float32)
        L[:,-1] = InputSequence

        # Define the bit matrix and place the known frozen zeros into the left most column
        B                   = -10*np.ones([N, NumberOfStages + 1], np.int8)
        B[FrozenIndices, 0] = np.zeros(len(FrozenIndices), np.int8)

        # ----------------------------------------------------------
        # Start Simplified Successive Decoding Process
        # ----------------------------------------------------------
        CurrentNode  = 0
        CurrentStage = NumberOfStages

        while True:
            # Check the state of the current node to see what needs to be done
            NodeState = NodeStateList[CurrentStage - 1][CurrentNode]
            # If 0 -> We need to run the MinSum Decoding (Moving right to left)
            # If 1 -> We need to run the repetition decoder (Moving right to left)
            # If 2 -> Re-encode Bit information (Moving left to right)
            # If 3 -> The node is done. Current configuration should never be reached.
            # Get the ports for the current node and stage
            _, CurrentUpperIndices, CurrentLowerIndices = GetPortIndices(CurrentNode, CurrentStage)

            match(NodeState):
                case 0:    # Let's do the MinSum Operation 
                    UpperBeliefs    =  L[CurrentUpperIndices, CurrentStage]   # Current Stage is 1 based index
                    LowerBeliefs    =  L[CurrentLowerIndices, CurrentStage]   

                    # If we are in stage 1, then we need to make hard decisions. Careful, stage 1 is the first stage (1based index)
                    if(CurrentStage == 1):
                        # 1. Run the MinSumDecoder and deposit the result in L. 
                        #    As this is stage 1, there is only one CurrentUpperIndices and therefore one output belief
                        OutputBelief                          = MinSumDecoder(UpperBeliefs, LowerBeliefs)
                        CurrentUpperIndex                     = CurrentUpperIndices[0]
                        CurrentLowerIndex                     = CurrentLowerIndices[0]
                        L[CurrentUpperIndex, CurrentStage -1] = OutputBelief[0] 

                        # 2. Make hard decision unless we have reached a frozen bit
                        if(B[CurrentUpperIndex, CurrentStage - 1] != 0):                          # If this is not a frozen bit
                            HardDecision = int(np.round(0.5 * (np.sign(OutputBelief[0]) + 1)))    # map from -1/+1 float to 0/1 integers
                            B[CurrentUpperIndices, CurrentStage -1] = HardDecision 
                        
                        U = B[CurrentUpperIndices, CurrentStage -1]                               # Assign U

                        # 3. Now that we know U, let's run the repetition decoder
                        OutputBelief     = RepetitionDecoder(UpperBeliefs, LowerBeliefs, U)
                        L[CurrentLowerIndices, CurrentStage - 1] = OutputBelief

                        # 4. Make a hard decision unless we have reached a frozen bit
                        if(B[CurrentLowerIndices, CurrentStage - 1] != 0):                        # If this is not a frozen bit
                            HardDecision = int(np.round(0.5 * (np.sign(OutputBelief[0]) + 1)))    # map from -1/+1 float to 0/1 integers
                            B[CurrentLowerIndices, CurrentStage -1] = HardDecision      
        
                        # 5. Run the re-encoding step
                        B[CurrentUpperIndex, 1] = (B[CurrentUpperIndex, 0] + B[CurrentLowerIndex, 0]) % 2 
                        B[CurrentLowerIndex, 1] =  B[CurrentLowerIndex, 0]

                        # Both MinSum and Repetition decoding is done
                        NodeStateList[CurrentStage - 1][CurrentNode] = 2

                    # The CurrentStage > 1
                    # Run the MinSumDecoder and deposit the result in L
                    else:
                        OutputBeliefs                                = MinSumDecoder(UpperBeliefs, LowerBeliefs) 
                        L[CurrentUpperIndices, CurrentStage -1]      = OutputBeliefs 
                        NodeStateList[CurrentStage - 1][CurrentNode] = 1  # MinSum decoding done     

                # Here we run the repetition decoder for CurrentStage > 1
                case 1:
                    UpperBeliefs    =  L[CurrentUpperIndices, CurrentStage]   # Current Stage is 1 based index
                    LowerBeliefs    =  L[CurrentLowerIndices, CurrentStage]   
                    U               =  B[CurrentUpperIndices, CurrentStage - 1]
                    OutputBeliefs   =  RepetitionDecoder(UpperBeliefs, LowerBeliefs, U)
                    L[CurrentLowerIndices, CurrentStage - 1]     = OutputBeliefs
                    NodeStateList[CurrentStage - 1][CurrentNode] = 2  # Repetition decoding done     

                # We  have finished the repetition decoding and need to re-encode
                case 2:
                    UpperBits                                    = B[CurrentUpperIndices, CurrentStage - 1]
                    LowerBits                                    = B[CurrentLowerIndices, CurrentStage - 1]
                    B[CurrentUpperIndices, CurrentStage]         = np.mod(UpperBits + LowerBits, 2)
                    B[CurrentLowerIndices, CurrentStage]         = LowerBits
                    NodeStateList[CurrentStage - 1][CurrentNode] = 3  # We are completely done with this node     

                    # If we have reached the last stage, then exit the loop
                    if CurrentStage == NumberOfStages:
                        break

                case _:
                    assert False, 'We should not get here.'    


            # Let's determine the upcoming node and stage
            # If we are in stage 1, then we move from right to left
            MovingLeft = False
            if (NodeState == 0 or NodeState == 1) and CurrentStage != 1:
                MovingLeft = True

            CurrentStage, CurrentNode = GetNextNode(CurrentNode, CurrentStage, MovingLeft, bool(NodeState != 0))

            # The following code shows the NodeStateList, and the matrices L and B.
            # When debugging this code, it is helpful to place these statements
            Debugging = False
            if Debugging == True:
                Debugging(NodeStateList, L, B)

        return B[:, 0]






























# ------------------------------------------------------------------------------------------------- #
#                                                                                                   #
#                                                                                                   #
# > Binary Convolutional Encoding Class                                                             #
#                                                                                                   #
#                                                                                                   #
# ------------------------------------------------------------------------------------------------- #
class CBinaryConvolutionalCoder():
    '''
    brief: This class provides binary convolutional encoding services for the FlexLink Specification   
    '''
    def __init__(self
                , GeneratorPolynomialsOct:   list                    # For LTE -> [133, 171, 165]  (Octal representation)
                , bTailBiting:               bool     = False        # False - Needs padding zeros at end of message / True - No padding zeros
                , strMode:                   str      = 'SoftNonInverting') -> np.ndarray:
                                            # 0/1 = hard -> 0/1, softnoninverting -> -1/+1, softinverting -> +1/-1
        # ---------------------
        # Error checking
        # ---------------------
        assert isinstance(GeneratorPolynomialsOct, list),   'The GeneratorPolynomialsOct input argument must be a list.'
        for Polynomial in GeneratorPolynomialsOct:
            assert isinstance(Polynomial, str),             'The GeneratorPolynomialsOct input argument must be a list of integers.'
        assert isinstance(bTailBiting, bool),               'The bTailBiting input argument must be a boolean.'
        assert isinstance(strMode, str),                    'The strMode input argument must be a string.'
        assert strMode.lower() == 'hard' or strMode.lower() == 'softnoninverting' or strMode.lower() == 'softinverting', \
                                                            'The strMode input argument is invalid.'

        # Save off the member variables
        self.bTailBiting = bTailBiting
        self.strMode     = strMode

        # ---------------------
        # Set the class variables
        # ---------------------
        self.GeneratorPolynomialsBin  = []
        self.ConstraintLength         = 0

        # Translate the octal polynomials into binary vectors - The constraint length is inherent in the binary polynomial form.
        # The constraint length is the maximum bit length of the binary polynomial vectors, which is 7 in the example below.
        # i.e. 133oct -> 1'011'011, 171oct -> 1'111'001, 164oct -> 1'110'100
        # Had we used 133oct, 71oct, 64oct, the constraint length would still be 7 as the 133oct is still 7 bits in length.
        GeneratorPolynomialsBinTemp   = []
        ConstraintLength = 0
        for Polynomial in GeneratorPolynomialsOct:
            BinaryString = "{0:b}".format(int(Polynomial, 8))
            BinaryVector = [np.int8(d) for d in BinaryString]
            if len(BinaryVector) > ConstraintLength:
                ConstraintLength = len(BinaryVector)
            GeneratorPolynomialsBinTemp.append(BinaryVector)

        # It is possible that the binary vectors are of different lengths, as would have been the case for 133oct, 71oct, 64oct.
        # We need to determine the constraint length and ensure that all binary vectors are of that length.
        for BinaryVector in GeneratorPolynomialsBinTemp:
            if len(BinaryVector) < ConstraintLength:
                NumMissingZeros = ConstraintLength - len(BinaryVector)
                BinaryVector    = [0]*NumMissingZeros + BinaryVector

            self.GeneratorPolynomialsBin.append(BinaryVector)

        self.ConstraintLength = len(self.GeneratorPolynomialsBin[0])





    # --------------------------------------------------------------
    # > EncodeBits()
    # --------------------------------------------------------------
    def EncodeBits(self
                 , InputBits:   np.ndarray) -> np.ndarray:
        '''
        brief: This method encodes the input bits using the binary convolutional encoder
        '''
        # ---------------------
        # Error checking
        # ---------------------
        assert isinstance(InputBits, np.ndarray),           'The InputBits input argument must be a numpy array.'
        assert np.issubdtype(InputBits.dtype, np.integer),  'The InputBits input argument must be an integer array.'
        assert len(InputBits.shape) == 1,                   'The InputBits input argument must be a one dimensional array.'
        
        # ---------------------
        # Is the convolutional encoder tail biting?
        # ---------------------
        if self.bTailBiting == True:
            # The initial state of the shift register must be equal to the final projected state
            Reg = np.flipud(InputBits[-self.ConstraintLength + 1:]).astype(np.int8)    # Switch to np.int8 to avoid overflow later on
        else:
            # The initial state of the shift register is all zeros
            Reg = np.zeros(self.ConstraintLength - 1, np.uint8).astype(np.int8)        # Switch to np.int8 to avoid overflow later on


        # ---------------------
        # Run the Encoder
        # ---------------------
        NumOutputBitsPerInputBit = len(self.GeneratorPolynomialsBin) 
        Output                   = np.zeros(len(InputBits)*len(self.GeneratorPolynomialsBin), np.int8)
        
        InputVector = np.zeros(self.ConstraintLength, np.int8)
        for Index in range(0, len(InputBits)):
            # Get the input bit and place it properly  
            InputBit        = InputBits[Index]
            InputVector[0]  = InputBit
            InputVector[1:] = Reg
            # Compute the output bits
            for PolynomialIndex, Polynomial in enumerate(self.GeneratorPolynomialsBin):
                # Compute the output bit
                OutputBit  = np.remainder(np.sum(InputVector * Polynomial), 2)
                # Store the output bit
                Output[Index*NumOutputBitsPerInputBit + PolynomialIndex] = OutputBit

            # Update the shift register
            Reg = np.roll(InputVector, 1)[1:]


        # ---------------------
        # Return the encoded bits
        # ---------------------
        if self.strMode.lower() == 'softnoninverting':
            Output = 2*Output - 1
        
        if self.strMode.lower() == 'softinverting':
            Output = 1 - 2*Output
            
        return Output






    # --------------------------------------------------------------
    # > Viterbi Decoder()
    # --------------------------------------------------------------
    def ViterbiDecoder(self
                     , EncodedBits: np.array)-> np.array:
        '''
        This function executes the Viterbi decoder to recover the convolutionally encoded message bits
        '''
        # -------------------------------------
        # > Error checking
        # -------------------------------------
        assert isinstance(EncodedBits, np.ndarray),            'The EncodedBits argument must be an np.array'
        assert np.issubdtype(EncodedBits.dtype, np.floating) \
            or np.issubdtype(EncodedBits.dtype, np.integer),   'The EncodedBits must be numeric in nature'

        # -------------------------------------
        # > Convert the EncodedSoftBits to SoftNonInverting in case they are not formated as such
        # -------------------------------------
        if   self.strMode.lower() == 'softinverting': # Convert from hard to non-inverting softbits (+1/-1 -> -1/+1)
            InputBeliefs = -EncodedBits
        elif self.strMode.lower() == 'hard':
            InputBeliefs = 2*EncodedBits - 1        # Convert from hard to non-inverting softbits (0/1 -> -1/+1)
        else:
            InputBeliefs = EncodedBits.copy()

        # In case of Tail biting, we do twice the work, but avoid transmission of zero pads, which would guarantee
        # a final state of 0 in the decoder.
        if self.bTailBiting == True:
            InputBeliefs = np.hstack([InputBeliefs, InputBeliefs])

        # ------------------------------------
        # > Setup the Viterbi Decoder
        # ------------------------------------
        NumberOfStates           = 2 ** (self.ConstraintLength -1)
        NumOutputBitsPerInputBit = len(self.GeneratorPolynomialsBin)

        # Build two matrices each featuring dimensions NumberOfStates by NumOutputBItsPerInputBit
        # The first matrix holds the encoder output for state i with input bit = 0.
        # The second matrix holds the encoder output for state i with input bit = 1.
        # We pre-build these matrices for speed so that during the Viterbi loop we don't have to keep recalculating the same values.
        EncoderOutputForInput0 = np.zeros([NumberOfStates, NumOutputBitsPerInputBit], np.float32)
        EncoderOutputForInput1 = np.zeros([NumberOfStates, NumOutputBitsPerInputBit], np.float32)
        for State in range(0, NumberOfStates):
            strStateBinary = (bin(State)[2:]).zfill(self.ConstraintLength-1)
            BinaryList     = [int(d) for d in strStateBinary]
            BinaryVector0  = np.array([0] + BinaryList, np.int8)
            BinaryVector1  = np.array([1] + BinaryList, np.int8)
            for BitIndex in range(NumOutputBitsPerInputBit):
                # The *2 - 1 factor converts to noninverting softbits
                EncoderOutputForInput0[State, BitIndex] = (np.sum(BinaryVector0 * self.GeneratorPolynomialsBin[BitIndex]) % 2) * 2 - 1
                EncoderOutputForInput1[State, BitIndex] = (np.sum(BinaryVector1 * self.GeneratorPolynomialsBin[BitIndex]) % 2) * 2 - 1

        # Build the traceback matrix
        NumDecoderOutputBits = int(len(EncodedBits) / NumOutputBitsPerInputBit)
        DecodedBits          = np.zeros(NumDecoderOutputBits, np.uint8)
        TraceBackUnit        = np.zeros([NumberOfStates, NumDecoderOutputBits], np.int8)

        # Define the PathMetric Array. The path with the largest path metric is the winner. The Viterbi
        # decoder always assumes that the encoder (at the transmitter) started in state 0. For Tail biting
        # mode, this is not true. We will find a work around for this mode.
        PathMetricArray       = -1000*np.ones(NumberOfStates, np.float32)
        PathMetricArray[0]    = 0                                    #  We favor state 0 (the first element in the vector)
        PathMetricCopy        = PathMetricArray.copy()               #  We need a copy of the array for update purposes
        PathMetricMatrix      = np.zeros([NumberOfStates, NumDecoderOutputBits + 1], np.float32); # For debugging purposes only.
        PathMetricMatrix[:,0] = PathMetricArray                      # It's nice to see how the path metrics progress in time.

        # --------------------------------------
        # > Run the Viterbi Decoder
        # --------------------------------------
        # The outer loop
        for OutputBitIndex in range(0, NumDecoderOutputBits):
            # Grab a set of output bits from the InputBeliefs
            StartIndex   = OutputBitIndex * NumOutputBitsPerInputBit
            StopIndex    = StartIndex + NumOutputBitsPerInputBit
            ReceivedBits = InputBeliefs[StartIndex : StopIndex]

            # Run through each state (The inner loop)
            for StateAsInteger in range(0, int(NumberOfStates/2)):
                # We will process two states per loop interation. StateA and StateB
                # No matter what the previous state before StateA was, it's input bit was a 0.
                StateA_AsInteger = StateAsInteger
                # No matter what the previous state before StateB was, it's input bit was a 1.
                StateB_AsInteger = StateA_AsInteger + int(NumberOfStates/2)

                # StateA and StateB can only be reached from two previous states. What are they?
                PreviousLowerStateAsInteger = 2 * StateAsInteger
                PreviousUpperStateAsInteger = 2 * StateAsInteger + 1

                # ---------------------------
                # > Compute branch and path metrics for StateA
                # ---------------------------
                #   Let's first find the encoder outputs generated during the transition from these two previous states to StateA.
                #   Remember, because StateAAsInt is always < NumberOfStates/2, the input bit to get to StateA must have been a 0.
                EncoderOutputFromLowerState = EncoderOutputForInput0[PreviousLowerStateAsInteger, :]
                EncoderOutputFromUpperState = EncoderOutputForInput0[PreviousUpperStateAsInteger, :]

                # The new path metric = old path metric + branch metric
                BranchMetricLower  = np.sum(ReceivedBits * EncoderOutputFromLowerState)
                BranchMetricUpper  = np.sum(ReceivedBits * EncoderOutputFromUpperState)
                NewPathMetricLower = PathMetricArray[PreviousLowerStateAsInteger] + BranchMetricLower
                NewPathMetricUpper = PathMetricArray[PreviousUpperStateAsInteger] + BranchMetricUpper

                #print([BranchMetricLower, BranchMetricUpper, NewPathMetricLower, NewPathMetricUpper, PreviousLowerStateAsInteger, PreviousUpperStateAsInteger])

                # And the survivor is???
                if NewPathMetricLower >= NewPathMetricUpper:
                    SurvivorPathMetric             = NewPathMetricLower
                    SurvivorPreviousStateAsInteger = PreviousLowerStateAsInteger
                else:
                    SurvivorPathMetric             = NewPathMetricUpper
                    SurvivorPreviousStateAsInteger = PreviousUpperStateAsInteger

                TraceBackUnit[StateA_AsInteger, OutputBitIndex] = SurvivorPreviousStateAsInteger
                PathMetricCopy[StateA_AsInteger]                = SurvivorPathMetric


                # ---------------------------
                # > Compute branch and path metrics for StateB
                # ---------------------------
                #   Let's first find the encoder outputs generated during the transition from these two previous states to StateB.
                #   Remember, because StateBAsInt is always >= NumberOfStates/2, the input bit to get to StateB must have been a 1.
                EncoderOutputFromLowerState = EncoderOutputForInput1[PreviousLowerStateAsInteger, :]
                EncoderOutputFromUpperState = EncoderOutputForInput1[PreviousUpperStateAsInteger, :]

                # The new path metric = old path metric + branch metric
                BranchMetricLower  = np.sum(ReceivedBits * EncoderOutputFromLowerState)
                BranchMetricUpper  = np.sum(ReceivedBits * EncoderOutputFromUpperState)
                NewPathMetricLower = PathMetricArray[PreviousLowerStateAsInteger] + BranchMetricLower
                NewPathMetricUpper = PathMetricArray[PreviousUpperStateAsInteger] + BranchMetricUpper

                #print([BranchMetricLower, BranchMetricUpper, NewPathMetricLower, NewPathMetricUpper, PreviousLowerStateAsInteger, PreviousUpperStateAsInteger])

                # And the survivor is???
                if NewPathMetricLower >= NewPathMetricUpper:
                    SurvivorPathMetric             = NewPathMetricLower
                    SurvivorPreviousStateAsInteger = PreviousLowerStateAsInteger
                else:
                    SurvivorPathMetric             = NewPathMetricUpper
                    SurvivorPreviousStateAsInteger = PreviousUpperStateAsInteger

                TraceBackUnit[StateB_AsInteger, OutputBitIndex] = SurvivorPreviousStateAsInteger
                PathMetricCopy[StateB_AsInteger]                = SurvivorPathMetric

            # Copy the updated path metrics into the original array
            PathMetricArray = PathMetricCopy.copy()
            PathMetricMatrix[:, OutputBitIndex + 1] = PathMetricArray

        # -------------------------------
        # > Work your way backwards through the trace back unit
        # -------------------------------
        # If the transmitted bit stream has padding bits that forced the encoder into the zero state
        # then we know to start the trace back from state 0. If we don't know what the finat state was, 
        # then begin the traceback from the state with the largest (best) path metric.
        FinalStateAsInteger = 0
        if self.bTailBiting == True:
            FinalStateAsInteger = np.argmax(PathMetricArray)

        # Start the traceback
        CurrentStateAsInteger = FinalStateAsInteger
        for CurrentOutputBitIndex in range(NumDecoderOutputBits-1,-1,-1):   # Decrement from NumberDecoderOutputBits -1 to 0
            if CurrentStateAsInteger < int(NumberOfStates / 2):
                LastBitEnteringEncoder = 0
            else:
                LastBitEnteringEncoder = 1

            DecodedBits[CurrentOutputBitIndex] = LastBitEnteringEncoder

            # The CurrentStateAsInteger is now the previous state as indicated by the trace back unit
            CurrentStateAsInteger = TraceBackUnit[CurrentStateAsInteger, CurrentOutputBitIndex]

        return DecodedBits













# ------------------------------------------------------------------------------------------------------------------ #
#                                                                                                                    #
# > FlexLink Interleaver()                                                                                           #
#                                                                                                                    #
# ------------------------------------------------------------------------------------------------------------------ #
def InterleaverFlexLink(FEC_Mode:   int
                      , CBS_A_Flag: int) -> tuple:
    '''
    This method computes the InterleavingIndices[] vector.
    During the interleaving and deinterleaving operations, we have an input bit vector and an output bit vector.
    During the   interleaving operation, the input  bit at index x is moved to the output vector at an index = InterleavingIndices[x]
    During the deinterleaving operation, the output bit at index x is fetched from the input vector at index = InterleavingIndices[x]
    '''
    # Error checking
    assert isinstance(FEC_Mode, int)   and FEC_Mode >= 0   and FEC_Mode <= 1
    assert isinstance(CBS_A_Flag, int) and CBS_A_Flag >= 0 and CBS_A_Flag <= 3

    # Establish code block size
    CBS_LDPC  = [648, 1296, 1944, 1944]  # The possible code block sizes for LDPC  Coding
    CBS_Polar = [256,  512, 1024, 2048]  # The possible code block sizes for Polar Coding

    if FEC_Mode == 0:
        CBS = CBS_LDPC[CBS_A_Flag]
    else:
        CBS = CBS_Polar[CBS_A_Flag]

    # Establish the indices before interleaving = 0, 1, 2, 3, 4, ... , CBS - 1
    OriginalIndices = np.arange(0, CBS, 1, np.int16)
    NumRows       = 61
    NumColumns    = int(math.ceil(CBS/NumRows))
    NumElements   = NumRows * NumColumns

    # In the index matrix below, a -1 represents a NA value in the interleaving matrix
    # We usually can not fill in the entire matrix, and those positions at which
    # a -1 remains will be skipped when we read out of the matrix
    IndexMatrix   = -1 * np.ones([NumRows, NumColumns], np.int16)

    # Place the indices into the columns of the IndexMatrix
    for Index in OriginalIndices:
        Row                         = Index % NumRows
        Column                      = math.floor(Index / NumRows)
        IndexMatrix[Row, Column]    = Index 

    # We now read out the information one row at a time. The order of the rows 
    # is shown in the next vector.
    RowOrder   = np.zeros(61, np.uint16)
    OrderCount = 0 
    Offsets    = np.array([0, 6, 3, 9, 1, 11, 2, 10, 4, 8, 5, 7])
    for StartIndex in Offsets:
        for StepIndex in range(0, 61, 12):
            PotentialIndex       = StartIndex + StepIndex
            if PotentialIndex <= 60:
                RowOrder[OrderCount] = StartIndex + StepIndex 
                OrderCount          += 1

    # Read out the indices, skipping the -1 = NA positions
    InterleavingIndices = np.zeros(CBS, np.uint16)
    Count              = 0
    for RowIndex in RowOrder:
        for ColumnIndex in range(0, NumColumns):
            if IndexMatrix[RowIndex, ColumnIndex] == -1:
                continue
            InterleavingIndices[Count] = IndexMatrix[RowIndex, ColumnIndex]
            Count                     += 1
    assert Count == CBS, 'The process of creating the interleaving matrix failed'

    # Return both indexing vectors
    return InterleavingIndices
























# --------------------------------------------------------------------------------
# > Test bench
# --------------------------------------------------------------------------------
if __name__ == '__main__':

    # ---------------------------------
    # Verifying the CRC Computation
    # ---------------------------------
    if(False):
        # -> The following test vector is also used in the MatLab code: GenerateCRC_tb.m
        Message = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, \
                            1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, \
                            1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int8)
        
        # -------------------------------------------------------------------------------------------
        # Compute and check the 16 bit CRC. The correct sequence below comes from the MatLab test bench
        CorrectCrcOutput16 = [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0]
        CrcOutput16 = CCrcProcessor.ComputeCrc(CCrcProcessor.Generator16_LTE, Message)
        Errors      = np.sum(np.remainder(CorrectCrcOutput16 + CrcOutput16, 2))
        assert Errors == 0, 'The 16 bit CRC test failed.'

        # -------------------------------------------------------------------------------------------
        # Compute and check the 24 bit CRC. The correct sequence below comes from the MatLab test bench
        CorrectCrcOutput24 = [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0]
        CrcOutput24 = CCrcProcessor.ComputeCrc(CCrcProcessor.Generator24_LTEA, Message)
        Errors      = np.sum(np.remainder(CorrectCrcOutput24 + CrcOutput24, 2))
        assert Errors == 0, 'The 24 bit CRC test failed.'



    # ---------------------------------
    # Verifying the CLinearFeedbackShiftRegister class
    # ---------------------------------
    if(True):
        # The following are identical scrambler built two different ways. They need to deliver the same results
        Scrambler1 = CLinearFeedbackShiftRegister(bUseMaxLengthTable = True,  IndexOrTapConfig = 5)
        Scrambler2 = CLinearFeedbackShiftRegister(bUseMaxLengthTable = False, IndexOrTapConfig = [1, 0, 1, 0, 0])

        Scrambler1.InitializeShiftRegister([0, 0, 0, 0, 1])
        Scrambler2.InitializeShiftRegister([0, 0, 0, 0, 1])

        Output1   = Scrambler1.RunShiftRegister(NumberOutputBits= 100) 
        Output2   = Scrambler2.RunShiftRegister(NumberOutputBits= 100) 

        Errors    = np.sum((Output1 + Output2) % 2)

        assert Errors == 0, 'Scrambler1 and Scrambler2 should function identically, but they do not.'

        Scrambler3 = CLinearFeedbackShiftRegister(bUseMaxLengthTable = True,  IndexOrTapConfig = 8)
        Scrambler4 = CLinearFeedbackShiftRegister(bUseMaxLengthTable = False, IndexOrTapConfig = [1, 0, 1, 1, 1, 0, 0, 0, ])

        Scrambler3.InitializeShiftRegister([0, 0, 0, 0, 0, 0, 0, 1])
        Scrambler4.InitializeShiftRegister([0, 0, 0, 0, 0, 0, 0, 1])

        Output3   = Scrambler3.RunShiftRegister(NumberOutputBits= 300) 
        Output4   = Scrambler4.RunShiftRegister(NumberOutputBits= 300) 

        Errors    = np.sum((Output3 + Output4) % 2)

        assert Errors == 0, 'Scrambler3 and Scrambler4 should function identically, but they do not.'

    # ------------------------------------------- 
    # Verifying the LDPC Processor
    # ------------------------------------------- 
    if(False):
        Test = 6           # 0 - Test TransformParityCheckMatrix() and ComputeGeneratorMatrix() using some small
                        #     test example matrices 
                        # 1 - Transform a Prototype matrix into a ParityCheckMatrix using CreateParityCheckMatrix()
                        # 2 - Test all three functions using all Prototype Matrices. 
                        # 3 - Testing LDPC encoding
                        # 4 - Testing the SISO_SPC_Decoder
                        # 5 - Testing the message passing decoder with the book example
                        # 6 - Testing all WLAN variant of the LDPC processor


        LdpcProcessor = CLdpcProcessor()
        # ------------
        if Test == 0:
            # -> Here we test the Ldpc functionality on some very simple matrices
            #    We will check the TransformParityCheckMatrix() function and the
            #    ComputeGeneratorMatrix() function.
            H = np.array([[1, 1, 1, 1, 0, 0],
                        [1, 0, 0, 1, 1, 0],
                        [0, 0, 1, 1, 0, 1]], dtype = np.int8)

            H = np.array([[1, 1, 1, 1, 0, 1, 0],
                        [1, 0, 0, 1, 1, 0, 1], 
                        [0, 0, 1, 1, 0, 1, 1],
                        [1, 0, 0, 1, 0, 0, 1]], dtype = np.int8)
        
            H_New = CLdpcProcessor.TransformParityCheckMatrix(H)

            # Use the following function to show the matrix in Notepad++
            DebugUtility.ShowMatrix(H_New)

            # The generator matrix is verified internally and would assert upon failure
            G     = CLdpcProcessor.ComputeGeneratorMatrix(H)

            

        # ------------
        if Test == 1:
            H = LdpcProcessor.CreateParityCheckMatrix(648, '3/4')
            DebugUtility.ShowMatrix(CLdpcProcessor.PrototypeM648_3_4, 'PrototypeM648_3_4.txt')
            DebugUtility.ShowMatrix(H)

        # -----------
        if Test == 2:
            print('This test takes several seconds.')
            H = LdpcProcessor.CreateParityCheckMatrix(648, '1/2')
            G = CLdpcProcessor.ComputeGeneratorMatrix(H) 

            H = LdpcProcessor.CreateParityCheckMatrix(648, '2/3')
            G = CLdpcProcessor.ComputeGeneratorMatrix(H)

            H = LdpcProcessor.CreateParityCheckMatrix(648, '3/4')
            G = CLdpcProcessor.ComputeGeneratorMatrix(H)

            H = LdpcProcessor.CreateParityCheckMatrix(648, '5/6')
            G = CLdpcProcessor.ComputeGeneratorMatrix(H)

            H = LdpcProcessor.CreateParityCheckMatrix(1296, '1/2')
            G = CLdpcProcessor.ComputeGeneratorMatrix(H) 

            H = LdpcProcessor.CreateParityCheckMatrix(1296, '2/3')
            G = CLdpcProcessor.ComputeGeneratorMatrix(H)

            H = LdpcProcessor.CreateParityCheckMatrix(1296, '3/4')
            G = CLdpcProcessor.ComputeGeneratorMatrix(H)

            H = LdpcProcessor.CreateParityCheckMatrix(1296, '5/6')
            G = CLdpcProcessor.ComputeGeneratorMatrix(H)

            H = LdpcProcessor.CreateParityCheckMatrix(1944, '1/2')
            G = CLdpcProcessor.ComputeGeneratorMatrix(H) 

            H = LdpcProcessor.CreateParityCheckMatrix(1944, '2/3')
            G = CLdpcProcessor.ComputeGeneratorMatrix(H)

            H = LdpcProcessor.CreateParityCheckMatrix(1944, '3/4')
            G = CLdpcProcessor.ComputeGeneratorMatrix(H)

            H = LdpcProcessor.CreateParityCheckMatrix(1944, '5/6')
            G = CLdpcProcessor.ComputeGeneratorMatrix(H)
        
            print('The test has passed.')


        # ------------
        if Test == 3:
            H = LdpcProcessor.CreateParityCheckMatrix(648, '1/2')
            G = CLdpcProcessor.ComputeGeneratorMatrix(H) 

            NumBits = 324
            InputBits = np.random.randint(low=0, high = 2, size = (1, NumBits), dtype = np.uint16)
            EncodedBits = LdpcProcessor.EncodeBits(InputBits, 'hard', True)
            Stop = 1

        # ------------
        if Test == 4:
            r = np.array([-0.1, -1, -0.9, 1.2], dtype = np.float32)
            print(CLdpcProcessor.SISO_SPC_Decoder(r))


        # -----------
        if Test == 5:
            # Test repeats the book example in section 5.6.5.3 of the book
            H = np.array([[1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],\
                        [0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0],\
                        [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],\
                        [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],\
                        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],\
                        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],\
                        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]], dtype = np.int8)

            LdpcProcessor1 = CLdpcProcessor('custom', H)

            # TxBipolarBits = [ 1  -1   1   -1  1  -1   1     -1   -1    1  -1    1   1  -1
            # Define Rx bit beliefs forcing errors are in positions 2, 7, 12
            RxBeliefs  = np.array([2, .1,  1.5,  -1,  1,  -1,  -0.2,  -0.8,  -0.8,  1, -1,  -0.3,  1,  -1], np.float32)

            # Run the message passing decoder
            OutputBits = LdpcProcessor1.DecodeBits(RxBeliefs 
                                                , NumberIterations = 10
                                                , strSoftbitMapping = 'non-inverting') 

            ProperAnswer = np.array([1, 0, 1, 0, 1, 0, 1], np.uint16)
            assert all(OutputBits == ProperAnswer), 'Test 5 has failed'
            print('Test 5 has passed.')


        # -----------
        if Test == 6:
            # Parameter for bit encoding
            NumInputBits      = 324 * 100
            strBitMode        = 'non-inverting'
            bAllowZeroPadding = False
            InputBits         = np.random.randint(low = 0, high = 2, size = NumInputBits, dtype = np.uint16)

            SnrdB_List         = [0, 1, 2, 3, 4]
            NumEncodedBitsList = [648, 1296, 1944] 
            RateList           = ['1/2', '2/3', '3/4', '5/6']
            BER_List           = []

            # Iterate through the LDPC sizes
            BerListIndex       = 0
            for NumEncodedBits in NumEncodedBitsList:
                # Iterate through the rate
                for Rate in RateList:
                    # Increment list index
                    BER_List.append([0] * len(SnrdB_List))
                    
                    # Run the test for Number of Encoded bit = 648 and rate 1/2
                    print('Building LdpcProcess for ' + str(NumEncodedBits) + ' bits at rate ' + Rate + '.')
                    LdpcProcessor1 = CLdpcProcessor('WLAN', None, 648, '1/2')
                    EncodedBits = LdpcProcessor1.EncodeBits(InputBits
                                                        , strBitMode                
                                                        , bAllowZeroPadding)

                    # Iterate through the SINR
                    for SnrIndex, SnrdB in enumerate(SnrdB_List):
                        print('Decoding at Snr = ' + str(SnrdB) + 'dB.')
                        RxBeliefs = AddAwgn(SnrdB, EncodedBits)

                        # Run the message passing decoder
                        OutputBits = LdpcProcessor1.DecodeBits(RxBeliefs 
                                                            , NumberIterations = 10
                                                            , strSoftbitMapping = 'non-inverting') 

                        # Compute the BER
                        NumErrors = np.sum(np.mod(InputBits + OutputBits, 2))
                        BER       = NumErrors / NumInputBits
                        BER_List[BerListIndex][SnrIndex] = BER
                        
                        print('The BER = ' + str(BER))

                    BerListIndex += 1

        plt.figure(1)
        plt.subplot(4,1,1)
        plt.semilogy(SnrdB_List, BER_List[0], 'r')
        plt.semilogy(SnrdB_List, BER_List[1], 'b')
        plt.semilogy(SnrdB_List, BER_List[2], 'k')
        plt.semilogy(SnrdB_List, BER_List[3], 'k:')
        plt.grid(True)
        plt.tight_layout()
        plt.title('BER for Encoded Bit Size = 648')
        plt.xlabel('SNR')
        plt.xlabel('BER')
        plt.legend(['1/2', '2/3', '3/4', '5/6'])
        plt.subplot(4,1,2)
        plt.semilogy(SnrdB_List, BER_List[4], 'r')
        plt.semilogy(SnrdB_List, BER_List[5], 'b')
        plt.semilogy(SnrdB_List, BER_List[6], 'k')
        plt.semilogy(SnrdB_List, BER_List[7], 'k:')
        plt.grid(True)
        plt.tight_layout()
        plt.title('BER for Encoded Bit Size = 1296')
        plt.xlabel('SNR')
        plt.xlabel('BER')
        plt.legend(['1/2', '2/3', '3/4', '5/6'])
        plt.subplot(4,1,3)
        plt.semilogy(SnrdB_List, BER_List[8], 'r')
        plt.semilogy(SnrdB_List, BER_List[9], 'b')
        plt.semilogy(SnrdB_List, BER_List[10], 'k')
        plt.semilogy(SnrdB_List, BER_List[11], 'k:')
        plt.grid(True)
        plt.tight_layout()
        plt.title('BER for Encoded Bit Size = 1944')
        plt.xlabel('SNR')
        plt.xlabel('BER')
        plt.legend(['1/2', '2/3', '3/4', '5/6'])
        plt.show()



    # ------------------------------------------- 
    # Verifying the Polar Processor
    # ------------------------------------------- 
    if(False):

        Test = 4        # 0 - Check the channel erasure probabilities
                        # 1 - Run the simple N = 4 Encoder / Decoder Example from the book
                        # 2 - Run the the general encoder / decoder test
                        # 3 - The Polar Encoder for the Signal Field Format 1
        
        D = CLinearFeedbackShiftRegister()

        # ----------------------------------------------------
        # 0. Look at the channel erase probability
        # ----------------------------------------------------
        if Test == 0:
            ErasureProbability = CPolarProcessor.FindChannelProbabilities(4, .3)
            print(ErasureProbability)


        # ----------------------------------------------------
        # 1. Run the simple N = 4 Encoder / Decoder example from the book
        # ----------------------------------------------------
        if Test == 1:
            PolarProcessor   = CPolarProcessor()

            N                =  4      # The number of total encoded bits
            K                =  2      # The number of message bits
            NumberFrozenBits =  N - K
            EncoderRate      =  K/N

            SourceBits       = np.array([0, 0, 1, 0], np.int32)
            EncodedBits      = PolarProcessor.RunPolarEncoder(SourceBits)
            print('Source bits:    ' + str(SourceBits))
            print('Encoded bits:   ' + str(EncodedBits))
           
            RxBeliefs        = np.array([0.9, -1.1, 0.8, -1.0], np.float32)
            FrozenIndices    = np.array([0, 1], np.int32) 
            DecodedBits      = PolarProcessor.RunPolarDecoder(RxBeliefs, FrozenIndices)
            print('Decoded bits:   ' + str(DecodedBits))


        # ----------------------------------------------------
        # 2. Run the the general encoder / decoder test
        # ----------------------------------------------------
        if Test == 2:
            PolarProcessor   = CPolarProcessor()

            # Let's set up the test
            SNRdB            = 50
            N                = 512               # The number of total encoded bits
            K                = 256               # The number of message bits
            NumberFrozenBits = N - K             # The number of bits we will force to 0
            EncoderRate      = K/N               # The ratio of message to total bits
            MessageBits      = np.random.randint(low=0, high=2, size=(K,), dtype = np.int8)
            
            # The erasure probability is linked to the CINR that we measured
            p                    = 0.3
            ErasureProbabilities = CPolarProcessor.FindChannelProbabilities(N, p)

            # Low probabilities (good decoding) are first and high erasure probabilities are last
            SortedProbabilities  = np.sort(ErasureProbabilities)
            SortedIndices        = np.argsort(ErasureProbabilities)
            MessageBitIndices    = SortedIndices[0: K]
            FrozenBitIndices     = SortedIndices[K:]

            # Encode the message bits
            SourceBits                    = np.zeros(N, np.int8)
            SourceBits[MessageBitIndices] = MessageBits
            EncodedBits                   = PolarProcessor.RunPolarEncoder(SourceBits)
            EncodedBipolarBits            = 2*EncodedBits.astype(np.int8) - 1

            # Add noise to the message bits
            SNR_Linear      = 10**(SNRdB/10)
            SignalPower     = 1
            NoisePower      = SignalPower/SNR_Linear
            Noise           = np.random.randn(N) * np.sqrt(NoisePower)
            NPower          = np.mean(Noise * Noise)
            RxSoftBits      = EncodedBipolarBits + Noise
            RxHardBits      = np.round(0.5 * np.sign(RxSoftBits) + 0.5)

            # Run the successive cancellation polar decoder
            OutputBits      = PolarProcessor.RunPolarDecoder(RxSoftBits, FrozenBitIndices)
            DecodedBits     = OutputBits[MessageBitIndices] 

            # Gather statistics
            # First, let's determine the raw bit error rate. How many coded bits where flipped in polarity by the noise.
            NumberRawErrors             = np.sum(np.mod(EncodedBits + RxHardBits,  2))
            BER_Raw                     = NumberRawErrors / N
            # Second, let's determine the message bit error rate. How many decoded message bits are in error.
            NumberMessageErrors         = np.sum(np.mod(DecodedBits + MessageBits, 2))
            BER_Message                 = NumberMessageErrors / K
            
            print('BER Raw:     ' + str(100*BER_Raw)      + '%')
            print('BER_Message: ' + str(100*BER_Message)  + '%')



        # ----------------------------------------------------
        # 3. Test the Polar encoder for the Signal Field 1 (This is a 1/4 rate encoder with erasure probability = 0.55)
        # ----------------------------------------------------
        # In this test, we sweep the erasure probabilities and check to see how well the polar encoder performs for
        # different erasure probabilites. The erasure probability simply determines the positioning of the information bits
        # and frozen bits. We have already determined that we want a 1/4 rate encoder. The simulation shows that the
        # erasure probability should be set to between 0.5 to 0.65. We will use 0.55.
        if Test == 3:
            PolarProcessor   = CPolarProcessor()

            # Let's set up the test
            N                = 256                # The number of total encoded bits
            K                = 64                 # The number of message bits
            NumberFrozenBits = N - K              # The number of bits we will force to 0
            EncoderRate      = K/N                # The ratio of message to total bits
            MessageBits      = np.random.randint(low=0, high=2, size=(K,), dtype = np.int8)

            NumErasureProbabilities = 10
            NumSnrdBIterations      = 5
            BER_Matrix              = np.zeros([NumErasureProbabilities, NumSnrdBIterations], np.float32)

            ProbStart = 0.25
            ProbStep  = 0.05
            for ProbIndex in range(0, NumErasureProbabilities):
                # The erasure probability is linked to the CINR that we measured
                p                    = ProbStart + ProbStep * ProbIndex
                ErasureProbabilities = CPolarProcessor.FindChannelProbabilities(N, p)

                # Low probabilities (good decoding) are first and high erasure probabilities are last
                SortedProbabilities  = np.sort(ErasureProbabilities)
                SortedIndices        = np.argsort(ErasureProbabilities)
                MessageBitIndices    = SortedIndices[0: K]
                FrozenBitIndices     = SortedIndices[K:]

                # Encode the message bits
                SourceBits                    = np.zeros(N, np.int8)
                SourceBits[MessageBitIndices] = MessageBits
                EncodedBits                   = PolarProcessor.RunPolarEncoder(SourceBits)
                EncodedBipolarBits            = 2*EncodedBits.astype(np.int8) - 1

                # Add noise to the message bits
                SNRdBStart = -4
                SNRdBStep  = 1
                for SnrIndex in range(0, NumSnrdBIterations):
                    SNRdB           = SNRdBStart + SNRdBStep * SnrIndex
                    SNR_Linear      = 10**(SNRdB/10)
                    SignalPower     = 1
                    NoisePower      = SignalPower/SNR_Linear
                    Noise           = np.random.randn(N) * np.sqrt(NoisePower)
                    NPower          = np.mean(Noise * Noise)
                    RxSoftBits      = EncodedBipolarBits + Noise
                    RxHardBits      = np.round(0.5 * np.sign(RxSoftBits) + 0.5)

                    # Run the successive cancellation polar decoder
                    OutputBits      = PolarProcessor.RunPolarDecoder(RxSoftBits, FrozenBitIndices)
                    DecodedBits     = OutputBits[MessageBitIndices] 

                    # Gather statistics
                    # First, let's determine the raw bit error rate. How many coded bits where flipped in polarity by the noise.
                    NumberRawErrors             = np.sum(np.mod(EncodedBits + RxHardBits,  2))
                    BER_Raw                     = NumberRawErrors / N
                    # Second, let's determine the message bit error rate. How many decoded message bits are in error.
                    NumberMessageErrors         = np.sum(np.mod(DecodedBits + MessageBits, 2))
                    BER_Message                 = NumberMessageErrors / K
                    
                    BER_Matrix[ProbIndex, SnrIndex] = BER_Message

                    #print('BER Raw:     ' + str(100*BER_Raw)      + '%')
                    #print('BER_Message: ' + str(100*BER_Message)  + '%')

                print('Erasure Probability: ' + str(p))
                print('Bit Error Rates:     ' + str(BER_Matrix[ProbIndex, :]))
            


    # ------------------------------------------- 
    # Verifying the Interleaver
    # ------------------------------------------- 
    if False:
        FEC_Mode   = 1
        CBS_A_Flag = 1
        InterleavingIndices = Interleaver(FEC_Mode, CBS_A_Flag)

        Stop = 1



    # -------------------------------------------
    # Verifying the BinaryConvolutionalCoder
    # -------------------------------------------
    if True:
        ConstraintLength        = 7
        GeneratorPolynomialsOct = ['133', '171', '165']
        ModeString              = 'softnoninverting'
        TailBitingFlag          = True

        ConvolutionalCoder = CBinaryConvolutionalCoder(GeneratorPolynomialsOct, TailBitingFlag, ModeString)

        #InputBits    = np.array([1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], np.int8)
        #InputBits   = np.array([0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0], np.uint8)

        # Number of Input bits
        NumBits = 10000
        InputBits    = np.random.randint(0, 2, NumBits, np.uint8)

        bTailBiting = False
        strMode     = 'hard'

        # Convolutionally encode the input bits
        EncodedBits = ConvolutionalCoder.EncodeBits(InputBits)

        # Set up a loop to determine the BER
        SnrList_dB = np.arange(0, 10, 1, np.int32)

        np.random.seed(19680801)
        NormalNoise = np.random.randn(len(EncodedBits))    # Gaussian noise with unity variance

        for SnrdB in SnrList_dB:
            # Viterbi decode the received bits
            SnrLinear  = 10**(SnrdB/10)
            NoisePower = 1 / SnrLinear 

            DecodedBits = ConvolutionalCoder.ViterbiDecoder(EncodedBits + NormalNoise * np.sqrt(NoisePower) * NormalNoise) 

            NumErrors   = np.sum( (InputBits + DecodedBits) % 2)

            BER = NumErrors / len(DecodedBits)

            print('BER at ' + str(SnrdB) + 'dB SINR = ' + str(BER))

        if NumErrors == 0:
            print('The test has passed.')
        else:
            print('The test has failed.')

        Stop = 1