# File:       CommParameters.py
# Author:     Andreas Schwarzinger (8USNT - Rohde & Schwarz USA)
# Notes:      This module contains all important communiction constants that we need

__title__     = "Parameters"
__author__    = "Andreas Schwarzinger <8USNT - Rohde & Schwarz USA>"
__status__    = "preliminary"
__version__   = "0.2.0.1"
__date__      = "March, 10, 2022"
__copyright__ = 'Copyright 2022 by Rohde & Schwarz USA'

from   enum import unique, Enum  
import numpy as np

BaseNormalCp   = np.array([10, 9, 9, 9, 9, 9, 9], dtype = np.int16)
BaseExtendedCp = np.array([32, 32, 32, 32, 32], dtype = np.int16)

# ------------------------------------------------------------------------------------------------
# > Declaration of the ELteNRB enum class   (Handles LTE bandwidths)
# ------------------------------------------------------------------------------------------------
# creating enumeration for the Lte Bandwidth
@unique  # Ensure that no name can have the same value
class ELteNRB(Enum):
    BwUnknown   = 0
    BwIs1_4MHz  = 6      # This variant will feature an FFT size of 128  and a sample rate of  1.92MHz
    BwIs3MHz    = 15     # This variant will feature an FFT size of 256  and a sample rate of  3.84MHz
    BwIs5MHz    = 25     # This variant will feature an FFT size of 512  and a sample rate of  7.68MHz
    BwIs10MHz   = 50     # This variant will feature an FFT size of 1024 and a sample rate of 15.36MHz
    BwIs15MHz   = 75     # This variant will feature an FFT size of 1408 and a sample rate of 21.12MHz (TSME Limit)
    BwIs20MHz   = 100    # This variant will feature an FFT size of 1408 and a sample rate of 21.12MHz (TSMR Limit)
    
    # This function checks to see that provided value is valid
    @classmethod
    def CheckValidBandwidth(cls, EnumInput) -> bool:
        for Member in ELteNRB:
            if EnumInput == Member and EnumInput.value != 0:
                return True
        return False

    # This function checks to see that provided number of resource blocks is valid
    @staticmethod
    def CheckNumberOfResourceBlocks(NRB) -> bool:
        for Member in ELteNRB:
            if Member.value == NRB:
                return True
        return False

    # This function checks to see whether the number of subcarriers provided is legal
    def GetNumSubcarriers(self) -> int:
        return self.value * 12  
         




# ------------------------------------------------------------------------------------- 
# > Declaration of FFT Parameters such as FFT_Length, SampleRate, and CP Lengths
# -------------------------------------------------------------------------------------
# creating enumeration for the Lte FFT Sizes
@unique  # Ensure that no name can have the same value
class ELteFftSizesCp(Enum):
    FftSize128   = { 'FftSize' : 1*128,  'SampleRate' :  1*1.92e6, 'CpNormal' :  1*BaseNormalCp,   'CpExtended':  1*BaseExtendedCp}
    FftSize256   = { 'FftSize' : 2*128,  'SampleRate' :  2*1.92e6, 'CpNormal' :  2*BaseNormalCp,   'CpExtended':  2*BaseExtendedCp}
    FftSize512   = { 'FftSize' : 4*128,  'SampleRate' :  4*1.92e6, 'CpNormal' :  4*BaseNormalCp,   'CpExtended':  4*BaseExtendedCp}
    FftSize1024  = { 'FftSize' : 8*128,  'SampleRate' :  8*1.92e6, 'CpNormal' :  8*BaseNormalCp,   'CpExtended':  8*BaseExtendedCp}
    FftSize1408  = { 'FftSize' : 11*128, 'SampleRate' : 11*1.92e6, 'CpNormal' : 11*BaseNormalCp,   'CpExtended': 11*BaseExtendedCp}
    FftSize1536  = { 'FftSize' : 12*128, 'SampleRate' : 12*1.92e6, 'CpNormal' : 12*BaseNormalCp,   'CpExtended': 12*BaseExtendedCp}
    FftSize2048  = { 'FftSize' : 16*128, 'SampleRate' : 16*1.92e6, 'CpNormal' : 16*BaseNormalCp,   'CpExtended': 16*BaseExtendedCp}
    
    # This function checks to see that provided value is valid
    @classmethod
    def CheckValidEnumeration(cls, EnumInput) -> bool:
        for Member in ELteFftSizesCp:
            if EnumInput == Member:
                return True
        return False

    



# ------------------------------------------------------------------------------------------------
# > Declaration of the eCP enum class (Handles cyclic prefix modes)
# ------------------------------------------------------------------------------------------------
# creating enumeration for the cyclic prefix
@unique  # Ensure that no name can have the same value
class ELteCP(Enum):
    CpExtended               = 0
    CpNormal                 = 1
    CpUnknown                = 2

    # This function checks to see that provided value is valid
    @classmethod
    def CheckValidCyclicPrefix(cls, EnumInput) -> bool:
        for Member in ELteCP:
            if EnumInput == Member and EnumInput.name != 'CpUnknown':
                return True
        return False

    # Compute the number of OFDM symbols per slot
    def GetNumSymbolsPerSlot(self):
        if self.name == 'CpUnknown':
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





# ------------------------------------------------------------------------------------------------
# > Class holding only static variables
# ------------------------------------------------------------------------------------------------
class Params():
    LTE_IFFT_PERIOD              =  66.6666666666666e-6   # The length of the IFFT portion of the OFDM symbol
    LTE_SLOT_PERIOD              =   0.5e-3               # seconds
    LTE_SUBFRAME_PERIOD          =     1e-3               # seconds
    LTE_RADIOFRAME_PERIOD        =    10e-3               # seconds
    LTE_NUM_SLOTS_PER_SUBFRAME   =     2                  
    LTE_NUM_SLOTS_PER_FRAME      =    20
    LTE_NUM_SUBFRAMES_PER_FRAME  =    10
    LTE_NUM_FRAMES_IN_HYPERFRAME =  1024
    LTE_NUM_RE_PER_RB            =    12                                           
    LTE_SC_SPACING               = 15000                  # Hz     
    V2X_SYMBOLS_PER_SLOT         = 7                





# ---------------------------------------------------------
# Function: The test bench
# ---------------------------------------------------------

if __name__ == "__main__":
    # Testing the ELteNRB enum class
    A1 = ELteNRB.BwIs10MHz
    B1 = ELteNRB.CheckValidBandwidth(A1)  

    # Testing the EFrameStructure enum class
    A2 = EFrameStructure.IsUnknown
    B2 = EFrameStructure.IsFdd
    CheckA2 = EFrameStructure.CheckValidFrameStructure(A2)
    CheckB2 = EFrameStructure.CheckValidFrameStructure(B2)
    Hier = Params.LTE_NUM_RE_PER_RB
     
    # Testing ELteCP enum
    A3 = ELteCP.CpUnknown
    B3 = ELteCP.CpNormal
    CheckA3 = ELteCP.CheckValidCyclicPrefix(A3)
    CheckB3 = ELteCP.CheckValidCyclicPrefix(B3)

    # Testing ELteFftSizesCp enum
    FftParamsDict  = ELteFftSizesCp.FftSize1536.value
    Hier = 1