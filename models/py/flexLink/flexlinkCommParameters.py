# File:   commParameters.py
# Author: Andreas Schwarzinger, Leonard Dieguez
# Notes:  This module contains important communication constants that we need for other Modules
# contains parameters for flexlink OFDM waveform

__title__     = "Parameters"
__author__    = "Andreas Schwarzinger, Leonard Dieguez"
__status__    = "preliminary"
__version__   = "0.2.0.1"
__date__      = "Sept, 15, 2022"
__license__   = 'MIT'

# imports 
from   enum import unique, Enum  
import numpy as np

# common number of samples for a CP this is multiplied by a sample rate multiplier based on 
# Bandwidth
baseNormalCP   = np.array([5, 5, 5, 5, 5], dtype = np.int16) # len of array = N symbols per resource block
baseExtendedCP = np.array([5, 5, 5, 5, 5], dtype = np.int16) # extended CP is not used. 
nsubcarrInRB   = 12 # number of subcarriers in a resource block, 12 for LTE
# ------------------------------------------------------------------------------------------------
# > Declaration of the flexlinkNRB enum class   (Handles LTE bandwidths)
# ------------------------------------------------------------------------------------------------
# creating enumeration for the Lte Bandwidth
# resource element is one sub-carrier x one symbol (Tg+Tu=Ts)
# NRB = number of resource blocks. 
# for LTE a resource block is a grid of 12 subcarriers * 7 symbols = 84 resource elements for 
# normal cp (12 * 6 = 72 for extended cp)
# for flexlink a resource block is a grid of 12 subcarriers * 5 symbols = 60 resource elements
# rf bw : sub-carrier bw * n-sb-per-rb * n-rb
# for rfbw = 16MHz flex link is 20,000 * 12 * 

@unique  # Ensure that no name can have the same value
class flexlinkNRB(Enum):
    BwUnknown   = 0
    BwIs2p5MHz  = 1 # This variant will feature an FFT size of 128  and a sample rate of  2.56MHz
    BwIs5MHz    = 2 # This variant will feature an FFT size of 256  and a sample rate of  5.12MHz
    BwIs10MHz   = 3 # This variant will feature an FFT size of 512  and a sample rate of 10.24MHz
    BwIs20MHz   = 4 # This variant will feature an FFT size of 1024 and a sample rate of 20.48MHz
    BwIs40MHz   = 5 # place holder for now
    
    # This function checks to see that provided value is valid
    @classmethod
    def CheckValidBandwidth(cls, EnumInput) -> bool:
        for Member in flexlinkNRB:
            if EnumInput == Member and EnumInput.value != 0:
                return True
        return False

    # This function checks to see that provided number of resource blocks is valid
    @staticmethod
    def CheckNumberOfResourceBlocks(NRB) -> bool:
        for Member in flexlinkNRB:
            if Member.value == NRB:
                return True
        return False

    # This function checks to see whether the number of subcarriers provided is legal
    def GetNumSubcarriers(self) -> int:
        return self.value * nsubcarrInRB
         

# ------------------------------------------------------------------------------------- 
# > Declaration of FFT Parameters such as FFT_Length, SampleRate, and CP Lengths
# -------------------------------------------------------------------------------------
# creating enumeration for the Lte FFT Sizes
@unique  # Ensure that no name can have the same value
class flexlinkfftSizesCP(Enum):
    fftSize128   = {'fftSize' : 1*128, 'SampleRate' :  1*1.92e6,
                   'CPNormal' :  1*baseNormalCP,   'CPExtended':  1*baseExtendedCP}
    fftSize256   = {'fftSize' : 2*128, 'SampleRate' :  2*1.92e6,
                    'CPNormal':  2*baseNormalCP,   'CPExtended':  2*baseExtendedCP}
    fftSize512   = {'fftSize' : 4*128, 'SampleRate' :  4*1.92e6, 
                    'CPNormal': 4*baseNormalCP,   'CPExtended':  4*baseExtendedCP}
    fftSize1024  = {'fftSize' : 8*128, 'SampleRate' :  8*1.92e6, 
                    'CPNormal':  8*baseNormalCP,   'CPExtended':  8*baseExtendedCP}
    fftSize2048  = {'fftSize' : 16*128,'SampleRate' : 16*1.92e6, 
                    'CPNormal': 16*baseNormalCP,   'CPExtended': 16*baseExtendedCP}
    
    # This function checks to see that provided value is valid
    @classmethod
    def CheckValidEnumeration(cls, EnumInput) -> bool:
        for Member in flexlinkfftSizesCP:
            if EnumInput == Member:
                return True
        return False

# ------------------------------------------------------------------------------------------------
# > Declaration of the eCP enum class (Handles cyclic prefix modes)
# ------------------------------------------------------------------------------------------------
# creating enumeration for the cyclic prefix
@unique  # Ensure that no name can have the same value
class flexlinkCP(Enum):
    CPExtended               = 0
    CPNormal                 = 1
    CPUnknown                = 2

    # This function checks to see that provided value is valid
    @classmethod
    def CheckValidCyclicPrefix(cls, EnumInput) -> bool:
        for Member in flexlinkCP:
            if EnumInput == Member and EnumInput.name != 'CPUnknown':
                return True
        return False

    # Compute the number of OFDM symbols per slot
    def GetNumSymbolsPerSlot(self):
        if self.name == 'CPUnknown':
            return 0
        else:
            return 6 + self.value

# ------------------------------------------------------------------------------------------------
# > Declaration of the eFrameStructure class (handles FDD, TDD modes)
# ------------------------------------------------------------------------------------------------
# creating enumeration for the frame structure type
@unique
class EFrameStructure(Enum):
    IsUnknown   = 0
    IsFdd       = 1
    IsTdd       = 2

    # This function checks to see that provided value is valid
    @classmethod
    def CheckValidFrameStructure(cls, EnumInput) -> bool:
        for Member in EFrameStructure:
            if EnumInput == Member and EnumInput.name != 'IsUnknown':
                return True
        return False

# ----------------------------------------------------------------------------------------------
# > Class holding only static variables
# ----------------------------------------------------------------------------------------------
class Params():
    LTE_IFFT_PERIOD              =  66.6666666666666e-6   # Length of IFFT portion of OFDM symbol
    LTE_SLOT_PERIOD              =   0.5e-3               # seconds
    LTE_SUBFRAME_PERIOD          =     1e-3               # seconds
    LTE_RADIOFRAME_PERIOD        =    10e-3               # seconds
    LTE_NUM_SLOTS_PER_SUBFRAME   =     2                  
    LTE_NUM_SLOTS_PER_FRAME      =    20
    LTE_NUM_SUBFRAMES_PER_FRAME  =    10
    LTE_NUM_FRAMES_IN_HYPERFRAME =  1024
    LTE_NUM_RE_PER_RB            = nsubcarrInRB                                           
    LTE_SC_SPACING               = 15000                  # Hz                     


# ---------------------------------------------------------
# Function: The test bench
# ---------------------------------------------------------

if __name__ == "__main__":
    # Testing the flexlinkNRB enum class
    A1 = flexlinkNRB.BwIs10MHz
    B1 = flexlinkNRB.CheckValidBandwidth(A1)  

    # Testing the EFrameStructure enum class
    A2 = EFrameStructure.IsUnknown
    B2 = EFrameStructure.IsFdd
    CheckA2 = EFrameStructure.CheckValidFrameStructure(A2)
    CheckB2 = EFrameStructure.CheckValidFrameStructure(B2)
    Hier = Params.LTE_NUM_RE_PER_RB
     
    # Testing flexlinkCP enum
    A3 = flexlinkCP.CPUnknown
    B3 = flexlinkCP.CPNormal
    CheckA3 = flexlinkCP.CheckValidCyclicPrefix(A3)
    CheckB3 = flexlinkCP.CheckValidCyclicPrefix(B3)

    # Testing flexlinkfftSizesCP enum
    FftParamsDict  = flexlinkfftSizesCP.fftSize1536.value
    Stop = 1
