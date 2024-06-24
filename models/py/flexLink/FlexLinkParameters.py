# File:  FlexLinkParemeters.py
# Notes: This files provides basic parameters for the FlexLink Wireless Communication standard.

__title__     = "FlexLinkParemeters"
__author__    = "Andreas Schwarzinger"
__status__    = "preliminary"
__date__      = "April, 21, 2024"
__copyright__ = 'Andreas Schwarzinger'


import math
import numpy          as  np
from   enum           import unique, Enum
from   FlexLinkCoder  import CCrcProcessor


# ---------------------------------------------------------------
# > Declare and enumerate all possible resource elements types
# ---------------------------------------------------------------
@unique
class EReType(Enum):
    Unassigned           = -1    # Not assigned
    Emtpy                = 0     # Empty resource element = 0 + j0
    RefSignalPort0       = 1     # Demodulation reference signals port 0
    RefSignalPort1       = 2     # Demodulation reference signals port 1
    Control              = 3     # Control resource element
    Data                 = 4     # Generic data resource element
    SignalField          = 5     # Data resource element reserved for signal field
    PayloadAEvenCodeWord = 6     # I want to be able to see the placement of different 
    PayloadAOddCodeWord  = 7     # code words inside PayloadA
    PayloadBEvenCodeWord = 8     # I want to be able to see the placement of different 
    PayloadBOddCodeWord  = 9     # code words inside PayloadB
    
    
    # This function verifies that we currently support the requested subcarrier spacing
    @classmethod
    def CheckValidOption(cls, EnumInput) -> bool:
        assert isinstance(EnumInput, EReType), 'The EnumInput is of improper type.'





# ---------------------------------------------------------------
# > Declare and define the control information
# ---------------------------------------------------------------
class CControlInformation():
    """
    brief:   The class manages the control information embedded in the first pilot/reference symbol
    """
    AvailableReferenceSymbolPeriodicity      = [1, 3, 6, 12]  # Reference symbols can occur every 1, 3, 6, or 12 symbols
    AvailableReferenceSignalSpacing          = [3, 6, 12, 24] # Reference signal may occur every 3, 6, 12, or 24 subcarriers
    AvailableSignalFieldSymbols              = [1, 2, 4, 10]  # Number of OFDM symbols available for the signal field
    AvailableSignalFieldFormats              = [1, 2, 3, 4]   # Format 1, 2, 3, or 4
    AvailableNumBitsPerSymbols               = [1, 2]         # BPSK or QPSK
    AvailableTxAntennaPorts                  = [1, 2]         # Number of Tx Antenna ports for which Reference Signals exist
    AvailableNumDcSubcarriers                = [1, 13]        # No data information may be mapped into the DC subcarriers
                                                              # No information of any kind may be mapped at the center DC subcarrier
    NumberOfControlBits                       = 12


    # -----------------------------------------------------------
    # >> The constructor
    # -----------------------------------------------------------   
    def __init__(self
               , ReferenceSymbolPeriodicityIndex: int  = 1
               , ReferenceSignalSpacingIndex:     int  = 0
               , NumSignalFieldSymbolsIndex:      int  = 0
               , SignalFieldFormatIndex:          int  = 0
               , SignalFieldModulationFlag:       int  = 0
               , NumberTxAntennaPortFlag:         int  = 0
               , DcSubcarrierFlag:                int  = 0):

        # Basic error checking
        assert isinstance(ReferenceSymbolPeriodicityIndex, int) and ReferenceSymbolPeriodicityIndex >= 0 and ReferenceSymbolPeriodicityIndex < 4  
        assert isinstance(ReferenceSignalSpacingIndex,  int)    and ReferenceSignalSpacingIndex >= 0     and ReferenceSignalSpacingIndex < 4                 
        assert isinstance(NumSignalFieldSymbolsIndex,  int)     and NumSignalFieldSymbolsIndex >= 0      and NumSignalFieldSymbolsIndex < 4                              
        assert isinstance(SignalFieldFormatIndex,  int)         and SignalFieldFormatIndex >= 0          and SignalFieldFormatIndex < 4                    
        assert isinstance(SignalFieldModulationFlag, int)       and SignalFieldModulationFlag >= 0       and SignalFieldModulationFlag < 2                      
        assert isinstance(NumberTxAntennaPortFlag,  int)        and NumberTxAntennaPortFlag >= 0         and NumberTxAntennaPortFlag < 2                                
        assert isinstance(DcSubcarrierFlag,  int)               and DcSubcarrierFlag >= 0                and DcSubcarrierFlag < 2               
                
        # Save Input Arguments
        self.ReferenceSymbolPeriodicityIndex  = ReferenceSymbolPeriodicityIndex
        self.ReferenceSignalSpacingIndex      = ReferenceSignalSpacingIndex
        self.NumSignalFieldSymbolsIndex       = NumSignalFieldSymbolsIndex
        self.SignalFieldFormatIndex           = SignalFieldFormatIndex
        self.SignalFieldModulationFlag        = SignalFieldModulationFlag
        self.NumberTxAntennaPortFlag          = NumberTxAntennaPortFlag
        self.DcSubcarrierFlag                 = DcSubcarrierFlag
 
        # Determine the parity bit. It is even parity, the parity bit assumes a value that makes the number of 1 values even
        Sum = ReferenceSymbolPeriodicityIndex + ReferenceSignalSpacingIndex + NumSignalFieldSymbolsIndex +\
              SignalFieldFormatIndex + SignalFieldModulationFlag + NumberTxAntennaPortFlag + DcSubcarrierFlag
        
        self.ParityBit                        = int(Sum % 2)

        # Process the input arguments
        self.ReferenceSymbolPeriodicity     = CControlInformation.AvailableReferenceSymbolPeriodicity[ReferenceSymbolPeriodicityIndex] 
        self.ReferenceSignalSpacing         = CControlInformation.AvailableReferenceSignalSpacing[ReferenceSignalSpacingIndex]
        self.NumberSignalFieldSymbols       = CControlInformation.AvailableSignalFieldSymbols[NumSignalFieldSymbolsIndex]
        self.SignalFieldFormat              = CControlInformation.AvailableSignalFieldFormats[SignalFieldFormatIndex]
        self.NumBitsPerQamSymbols           = CControlInformation.AvailableNumBitsPerSymbols[SignalFieldModulationFlag]
        self.NumberTxAntennas               = CControlInformation.AvailableTxAntennaPorts[NumberTxAntennaPortFlag]
        self.NumberDcSubcarriers            = CControlInformation.AvailableNumDcSubcarriers[DcSubcarrierFlag]

        # Create the Control Bit Array
        self.ControlInformationArray        = np.zeros(CControlInformation.NumberOfControlBits, np.uint8)
        self.ControlInformationArray[0]     = math.floor(ReferenceSymbolPeriodicityIndex / 2) % 2         # MSB of ReferenceSymbolPeriodicityIndex 
        self.ControlInformationArray[1]     = ReferenceSymbolPeriodicityIndex % 2                         # LSB of ReferenceSymbolPeriodicityIndex
        self.ControlInformationArray[2]     = math.floor(ReferenceSignalSpacingIndex / 2) % 2             # MSB of ReferenceSignalSpacingIndex 
        self.ControlInformationArray[3]     = ReferenceSignalSpacingIndex % 2                             # LSB of ReferenceSignalSpacingIndex
        self.ControlInformationArray[4]     = math.floor(NumSignalFieldSymbolsIndex / 2) % 2              # MSB of NumSignalFieldSymbolsIndex 
        self.ControlInformationArray[5]     = NumSignalFieldSymbolsIndex % 2                              # LSB of NumSignalFieldSymbolsIndex
        self.ControlInformationArray[6]     = math.floor(SignalFieldFormatIndex / 2) % 2                  # MSB of SignalFieldFormatIndex 
        self.ControlInformationArray[7]     = SignalFieldFormatIndex % 2                                  # LSB of SignalFieldFormatIndex
        self.ControlInformationArray[8]     = SignalFieldModulationFlag
        self.ControlInformationArray[9]     = NumberTxAntennaPortFlag
        self.ControlInformationArray[10]    = DcSubcarrierFlag
        self.ControlInformationArray[11]    = self.ParityBit


        # Build the Return String
        self.ReturnString  = '------------- Summary of Control Information ----------------- \n'
        self.ReturnString += 'Reference symbols appear every:        (b' + "{:02b}".format(ReferenceSymbolPeriodicityIndex) + ')  -> ' + str(self.ReferenceSymbolPeriodicity) +  " Ofdm symbols\n"  
        self.ReturnString += 'Reference signal appear every:         (b' + "{:02b}".format(ReferenceSignalSpacingIndex)     + ')  -> ' + str(self.ReferenceSignalSpacing)     + " Subcarriers\n"
        self.ReturnString += 'Number of Signal Field Ofdm symbols:   (b' + "{:02b}".format(NumSignalFieldSymbolsIndex)      + ')  -> ' + str(self.NumberSignalFieldSymbols)   + "\n" 
        self.ReturnString += 'Using Signal Field format:             (b' + "{:02b}".format(SignalFieldFormatIndex)          + ')  -> ' + str(self.SignalFieldFormat)          + "\n"
        self.ReturnString += 'The Number of bits per QAM Symbol is:  (b' + "{:01b}".format(SignalFieldModulationFlag)       + ')   -> ' + str(self.NumBitsPerQamSymbols)       + ' (1 - BSPK / 2 - QPSK)\n'
        self.ReturnString += 'Number of Tx Antennas:                 (b' + "{:01b}".format(NumberTxAntennaPortFlag)         + ')   -> ' + str(self.NumberTxAntennas)           + '\n'
        self.ReturnString += 'Number of DC Subcarriers:              (b' + "{:01b}".format(DcSubcarrierFlag)                + ')   -> ' + str(self.NumberDcSubcarriers)        + '\n'
        self.ReturnString += 'Parity bit:                            (b' + "{:01b}".format(self.ParityBit) + ')\n'
        self.ReturnString += 'The Control Information Array:         b'

        for Index in range(0, len(self.ControlInformationArray)):
            self.ReturnString += str(self.ControlInformationArray[Index])

        self.ReturnString += '\n'


    # ------------------------------------------------------------
    # >> Overload the str() function
    # ------------------------------------------------------------
    def __str__(self):


        return self.ReturnString








# ---------------------------------------------------------------
# > Declare and define the signal field information
# ---------------------------------------------------------------
class CSignalField():
    """
    The class manages the information embedded in the signal field
    """
    AvailableFecOptions          = ['LDPC Coding', 'Polar Coding']
    AvailableCodeBlockSizesLDPC  = [648, 1296, 1944, 1944]
    AvailableCodeBlockSizesPolar = [256,  512, 1024, 2048]
    AvailableCodingRateLDPC      = [1/2,  2/3,  3/4,  5/6,  1/2,  1/2,  1/2,  1/2]
    AvailableCodingRatePolar     = [1/4,  3/8,  1/2,  5/8,  3/4,  7/8,  1/4,  1/4]
    AvailableRateMatchingFactors = [0, 0.5, 0.75, 1, 3, 7, 15, 31]    # No bits are repeated / Half the bits are repeated / ... / All bits are repeated 31 times
    AvailableBitsPerQamSymbol    = [1, 2, 4, 6]                       # BPSK, QPSK, 16QAM, 64QAM
    AvailableClientFlags         = ['Point-to-Point', 'Point-to-Multipoint']


    # -----------------------------------------------------------
    # >> The constructor
    # -----------------------------------------------------------   
    def __init__(self
               , FEC_Mode:              int = 0       # 0/1 - LDPC Coding / Polar Coding
               , CBS_A_Flag:            int = 0       # 0/1/2/3 -> 648/1296/1944/1944 (LDPC) or 256/512/1024/2048 (Polar Coding)
               , FEC_A_Flag:            int = 0       # 0/1/2/3/4/5/6/7 -> 1/2,2/3,3/4,5/6,1/2,1/2,1/2 (LDPC) or 1/4,3/8,1/2,5/8,3/4,7/8,1/4,1/4 (Polar)
               , NDB_A:                 int = 0       # 0 to 10**14 - 1 or 16383  The number of data blocks sent in payload A
               , RM_A_Flag:             int = 0       # 0/1/2/3/4/5/6/7 -> RateMatchingFactor = 0, 0.5, 0.75, 1, 3, 7, 15, 31  Number rate matched bits = CBS * (1 + RateMatchingFactor)
               , BPS_A_Flag:            int = 0       # 0/1/2/3 -> BPSK, QPSK, 16QAM, 64QAM
               , NumOfdmSymbols:        int = 0       # 0 to 10**14 - 1 or 16383  
               , TxReferenceClockCount: int = 0       # 0 to 10**14 - 1 or 16383  The number of data blocks sent in payload A
               , Client_Flag:           int = 0       # 0/1 -> Point-to-Point / Point-to-Multipoint
               ):

        # ----------------------------------------
        # Basic type checking
        assert isinstance(FEC_Mode, int)              and FEC_Mode >=0               and FEC_Mode < 2,         'The FEC_Mode argument is invalid'
        assert isinstance(CBS_A_Flag, int)            and CBS_A_Flag >=0             and CBS_A_Flag < 4,       'The CBS_A_Flag argument is invalid'
        assert isinstance(FEC_A_Flag, int)            and FEC_A_Flag >=0             and FEC_A_Flag < 8,       'The FEC_A_Flag argument is invalid'
        assert isinstance(NDB_A, int)                 and NDB_A >=0                  and NDB_A < 2**14,        'The NDB_A argument is invalid'
        assert isinstance(RM_A_Flag, int)             and RM_A_Flag >=0              and RM_A_Flag < 8,        'The RM_A_Flag argument is invalid'
        assert isinstance(BPS_A_Flag, int)            or  isinstance(BPS_A_Flag, np.ndarray),                  'The BPS_A_Flag argument is invalid'
        if isinstance(BPS_A_Flag, int):
            assert BPS_A_Flag >=0   and   BPS_A_Flag < 4, 'The BPS_A_Flag argument is invalid'
            self.SignalFieldFormat = 1
            self.SignalFieldArray = np.zeros(64, np.uint8)      # The bit stream representing the Signal Field format 1
        else:
            assert len(BPS_A_Flag) == 76 or len(BPS_A_Flag) == 70, 'The BPS_A_Flag argument must be of length 76 (LTE BW) or 70 (WLAN BW).'
            assert all([( (x >= 0 and x < 8) and isinstance(x, int)) for x in BPS_A_Flag.tolist()]), 'The BPS_A_Flag argument is invalid'
            self.SignalFieldFormat = 2
            self.SignalFieldArray = np.zeros(290, np.uint8)     # The bit stream representing the signal field format 2
        assert isinstance(NumOfdmSymbols, int)        and NumOfdmSymbols >=0        and NumOfdmSymbols < 2**14,        'The NumOfdmSymbols argument is invalid'
        assert isinstance(TxReferenceClockCount, int) and TxReferenceClockCount >=0 and TxReferenceClockCount < 2**14, 'The TxReferenceClockCount argument is invalid'
        assert isinstance(Client_Flag, int)           and Client_Flag >=0           and Client_Flag < 2,               'The Client_Flag argument is invalid'

        # Save input arguments
        self.FEC_Mode               = FEC_Mode
        self.CBS_A_Flag             = CBS_A_Flag
        self.FEC_A_Flag             = FEC_A_Flag
        self.NDB_A                  = NDB_A 
        self.RM_A_Flag              = RM_A_Flag 
        self.BPS_A_Flag             = BPS_A_Flag 
        self.NumOfdmSymbols         = NumOfdmSymbols 
        self.TxReferenceClockCount  = TxReferenceClockCount
        self.Client_Flag            = Client_Flag

        # ---------------------------------------
        # Process Input Arguments
        self.FEC      = CSignalField.AvailableFecOptions[FEC_Mode]
        if FEC_Mode == 0:
            self.CodeBlockSizeA    = CSignalField.AvailableCodeBlockSizesLDPC[CBS_A_Flag]
            self.CodingRate        = CSignalField.AvailableCodingRateLDPC[FEC_A_Flag]
        else:
            self.CodeBlockSizeA    = CSignalField.AvailableCodeBlockSizesPolar[CBS_A_Flag]
            self.CodingRate        = CSignalField.AvailableCodingRatePolar[FEC_A_Flag]
        
        self.NumberDataBlocksA     = NDB_A
        self.RateMatchingFactor    = CSignalField.AvailableRateMatchingFactors[RM_A_Flag]

        if isinstance(self.BPS_A_Flag, int):
            self.BitsPerQamSymbol      = self.AvailableBitsPerQamSymbol[BPS_A_Flag]
        else:
            self.BitsPerQamSymbol      = BPS_A_Flag
            for Index, BPS_Flag in enumerate(BPS_A_Flag):
                self.BitsPerQamSymbol[Index]   = self.AvailableBitsPerQamSymbol[BPS_A_Flag[Index]]

        self.NumberOfdmSymbols     = NumOfdmSymbols
        self.TxReferenceClockCount = TxReferenceClockCount
        self.ClientMode            = CSignalField.AvailableClientFlags[Client_Flag]


        # ----------------------------------------
        # Create the SignalFieldArray and the Return String
        self.ReturnString  = '------------- Summary of Signal Field ----------------- \n'
        self.SignalFieldArray[0]     = FEC_Mode
        self.ReturnString += '  0: ' + self.FEC + ' = b' + "{:01b}".format(FEC_Mode) + '\n' 

        self.SignalFieldArray[1]     = math.floor(CBS_A_Flag / 2) % 2          # MSB of CBS_A_Flag 
        self.SignalFieldArray[2]     = CBS_A_Flag % 2                          # LSB of CBS_A_Flag
        self.ReturnString += '  1: Code Block Size: ' + str(self.CodeBlockSizeA) + ' = b' + "{:02b}".format(CBS_A_Flag) + '\n' 

        self.SignalFieldArray[3]     = math.floor(FEC_A_Flag / 4) % 2          # MSB of FEC_A_Flag
        self.SignalFieldArray[4]     = math.floor(FEC_A_Flag / 2) % 2          #  
        self.SignalFieldArray[5]     = math.floor(FEC_A_Flag / 1) % 2          # LSB of FEC_A_Flag 
        self.ReturnString += '  2: Coding Rate: ' + str(self.CodingRate) + ' = b' + "{:03b}".format(FEC_A_Flag) + '\n' 

        NextIndex = 6
        for Index in range(0, 14):
            self.SignalFieldArray[NextIndex] = math.floor( NDB_A / (2**(13 - Index)) ) % 2  # LSB first ... MSB last
            NextIndex += 1
        
        self.ReturnString += '  3: Number of Data Blocks: ' + str(self.NumberDataBlocksA) + ' = b' + "{:014b}".format(self.NumberDataBlocksA) + '\n' 

        self.SignalFieldArray[NextIndex]    = math.floor(RM_A_Flag / 4) % 2           # MSB of RM_A_Flag
        NextIndex += 1
        self.SignalFieldArray[NextIndex]    = math.floor(RM_A_Flag / 2) % 2           #  
        NextIndex += 1
        self.SignalFieldArray[NextIndex]    = math.floor(RM_A_Flag / 1) % 2           # LSB of RM_A_Flag 
        NextIndex += 1
        self.ReturnString += '  4: Rate Matching Factor: ' + str(self.RateMatchingFactor) + ' = b' + "{:03b}".format(RM_A_Flag) + '\n' 

        # Handle Signal Field Format 1 and 2
        if   self.SignalFieldFormat == 1:
            self.SignalFieldArray[NextIndex]   = math.floor(BPS_A_Flag / 2) % 2         # MSB of BPS_A_Flag
            NextIndex += 1
            self.SignalFieldArray[NextIndex]   = math.floor(BPS_A_Flag / 1) % 2         # LSB of BPS_A_Flag
            NextIndex += 1
            self.ReturnString += '  5: Bits per QAM Value: ' + str(self.BitsPerQamSymbol) + ' = b' + "{:02b}".format(self.BitsPerQamSymbol) + '\n' 

        elif self.SignalFieldFormat == 2:
            for n in range(0, 76):
                if n >= len(BPS_A_Flag):    # Skip entries if WLAN BW and we only have 70 resource blocks rather than 76
                                            # The Signal field format 2 will always provide space for 76 resource blocks for now.
                                            # I did not want to create another signal field format to distinguish between
                                            # the two bandwidth option.
                    continue 
                self.SignalFieldArray[NextIndex]    = math.floor(BPS_A_Flag[n] / 4) % 2           # MSB of RM_A_Flag
                NextIndex += 1
                self.SignalFieldArray[NextIndex]    = math.floor(BPS_A_Flag[n] / 2) % 2           #  
                NextIndex += 1
                self.SignalFieldArray[NextIndex]    = math.floor(BPS_A_Flag[n] / 1) % 2           # LSB of RM_A_Flag
                NextIndex += 1 
                self.ReturnString += '  5: Bits per QAM Value RB[' + str(n) + ']' + str(self.BitsPerQamSymbol[n]) + ' = ' + "{:03b}".format(self.BitsPerQamSymbol[n]) + '\n' 


        for n in range(0, 14):
            self.SignalFieldArray[NextIndex] = math.floor( NumOfdmSymbols / (2**(13-n)) ) % 2          # LSB first ... MSB last
            NextIndex += 1
        self.ReturnString += '  6: Number of OFDM Symbols: ' + str(NumOfdmSymbols) + ' = b' + "{:014b}".format(NumOfdmSymbols) + '\n' 

        for n in range(0, 14):
            self.SignalFieldArray[NextIndex] = math.floor( TxReferenceClockCount / (2**(13 -n) ) ) % 2   # LSB first ... MSB last
            NextIndex += 1
        self.ReturnString += '  7: Tx Reference Clock Count: ' + str(TxReferenceClockCount) + ' = b' + "{:014b}".format(TxReferenceClockCount) + '\n' 

        self.SignalFieldArray[NextIndex]    = Client_Flag                           # LSB of SignalFieldFormatIndex
        NextIndex += 1
        self.ReturnString += '  8: Client Configuration: ' + self.ClientMode + ' = b' + "{:01b}".format(Client_Flag) + '\n' 

        # Compute the 10 bit CRC
        CrcOutput        = CCrcProcessor.ComputeCrc(CCrcProcessor.Generator10_GSM, self.SignalFieldArray.tolist())
        self.ReturnString += '  9: CRC = b' # + self.ClientMode + ' = ' + "{:01b}".format(Client_Flag) + '\n' 

        for n in range(0, 10):
            self.SignalFieldArray[NextIndex] = CrcOutput[n]
            self.ReturnString += "{:01b}".format(CrcOutput[n])
            NextIndex += 1

        self.ReturnString += '\n'

        # Error checking. The NextIndex must be either 64 or 290 at this point. Otherwise we did something wrong
        assert NextIndex == 64 or NextIndex == 290, 'The Signal field was not built correctly'

        # Build the bit string 
        self.BitString  = ''
        for Index in range(0, len(self.SignalFieldArray)):
            self.BitString += "{:01b}".format(self.SignalFieldArray[Index])

        self.ReturnString += '  10: b' + self.BitString



    def __str__(self):
        return self.ReturnString



# -------------------------------------------------------------
# Test bench
# -------------------------------------------------------------
if __name__ == '__main__':

    # --------------------------------
    # Excercise the CControlInformation class
    # --------------------------------
    ControlInfo = CControlInformation(ReferenceSymbolPeriodicityIndex= 1
                                    , ReferenceSignalSpacingIndex = 1
                                    , NumSignalFieldSymbolsIndex = 1
                                    , SignalFieldModulationFlag = 1
                                    , NumberTxAntennaPortFlag = 1
                                    , DcSubcarrierFlag = 1)

    print(ControlInfo)


    # ---------------------------------
    # Excercise the CSignalField class
    # ---------------------------------
    SignalField1 = CSignalField(FEC_Mode   = 0
                              , CBS_A_Flag = 1
                              , FEC_A_Flag = 5
                              , NDB_A = 100
                              , RM_A_Flag = 1
                              , BPS_A_Flag = 1
                              , NumOfdmSymbols = 20
                              , TxReferenceClockCount = 16383
                              , Client_Flag = 0)

    print(SignalField1)

    SignalField2 = CSignalField(FEC_Mode   = 0
                              , CBS_A_Flag = 1
                              , FEC_A_Flag = 5
                              , NDB_A = 100
                              , RM_A_Flag = 1
                              , BPS_A_Flag = np.random.randint(low=0, high=8, size=76, dtype=np.uint8)
                              , NumOfdmSymbols = 20
                              , TxReferenceClockCount = 16383
                              , Client_Flag = 1)
    
    print(SignalField2)