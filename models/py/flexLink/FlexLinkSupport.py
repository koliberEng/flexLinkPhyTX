# Filename: FlexLinkSupport.py
# Author:   Andreas Schwarzinger      Date: August 26, 2022
# The following script will indicate how to construct the first reference symbol

import math

def ConfigureReferenceSymbol1(iScIndex:  int = 0
                            , iBwIndex:  int = 2
                            , iNi:       int = 1
                            , iPi:       int = 2
                            , iCi:       int = 0
                            , iTi:       int = 0
                            , iDi:       int = 1):

    # 0. Error checking
    assert iScIndex  >= 0 and iScIndex  < 4, 'The subcarrier index is out of range.'
    assert iBwIndex  >= 0 and iBwIndex  < 4, 'The bandwidth index is out of range.'
    assert iNi  >= 0 and iNi  < 4,           'The index indicating the number of OFDM symbols in the signal field is invalid.'
    assert iPi  >= 0 and iPi  < 4,           'The index indicating the reference symbol periodicity if invalid.'
    assert iCi  >= 0 and iCi  < 4,           'The index indicating the constellation used in the symbol field is invalid.'
    assert iTi  >= 0 and iTi  < 4,           'The index indicating the number of user bits in the signal field is invalid.'
    assert iDi  >= 0 and iDi  < 4,           'The index indicating the number of subcarriers at DC to skip is invalid.'
     
    # -------------------------------------------------------------------------
    # Determine subcarrier and bandwidth characteristics
    # ------------------------------------------------------------------------- 
    # 1. Parameter Interpretation
    SubcarrierSpacingHz = 20e3 * 2 ** iScIndex        # Hertz
    IFFT_Length         = 1 / SubcarrierSpacingHz     # Seconds
    CP_Length           = 4e-6                        # Seconds
    OfdmSymbolLength    = IFFT_Length + CP_Length     # seconds
    OfdmSymbolPerSecond = 1/OfdmSymbolLength

    # 1a. Get number of bits for constellation
    match iCi:
        case 0: NumberQamBits = 1
        case 1: NumberQamBits = 2
        case 2: NumberQamBits = 4
        case 3: NumberQamBits = 6

    # 1b. Get the channel bandwidth and compute number of subcarriers
    match iBwIndex:
        case 0:  # 5Mhz
            TotalBw = 5e6
            if(SubcarrierSpacingHz == 20e3): NumberOfTriplesPerSide =  36 
            if(SubcarrierSpacingHz == 40e3): NumberOfTriplesPerSide =  18 
        case 1:  # 10MHz
            TotalBw = 10e6
            if(SubcarrierSpacingHz == 20e3): NumberOfTriplesPerSide =  73 
            if(SubcarrierSpacingHz == 40e3): NumberOfTriplesPerSide =  36 
        case 2:  # 20MHz
            TotalBw = 20e6
            if(SubcarrierSpacingHz == 20e3): NumberOfTriplesPerSide = 133
            if(SubcarrierSpacingHz == 40e3): NumberOfTriplesPerSide =  66 
        case 3:  # 40MHz
            TotalBw = 40e6
            if(SubcarrierSpacingHz == 20e3): NumberOfTriplesPerSide = 292
            if(SubcarrierSpacingHz == 40e3): NumberOfTriplesPerSide = 148


    match iDi:
        case 0:  # No DC subcarriers
            NumberOfDcSubcarriers = 0
        case 1:  # One DC subcarrier
            NumberOfDcSubcarriers = 1
        case 2:  # One DC subcarrier
            NumberOfDcSubcarriers = 3
        case 3:  # One DC subcarrier
            NumberOfDcSubcarriers = 5


    NumberOfDataSubcarriers = 2* (NumberOfTriplesPerSide*3 + 1)
    if(NumberOfDcSubcarriers == 0):   # There are no DC subcarriers
        NumberOfDataSubcarriers -= 1

    MaxThroughputBitsPerSecond = NumberOfDataSubcarriers * NumberQamBits * OfdmSymbolPerSecond
    
    # 1c. Determine the tone locations of all subcarriers
    TonePMin = math.ceil(NumberOfDcSubcarriers/2)       # The tone index for the least  positive data carrying subcarrier
    TonePMax = TonePMin + NumberOfTriplesPerSide*3      # The tone index for the most   positive data carrying subcarrier 
    ToneMMin = -TonePMin                                # The tone index for the least  negative data carrying subcarrier
    ToneMMax = -TonePMax                                # The tone index for the most   negative data carrying subcarrier


    print('NumberOfDataSubcarriersPerSide = ' + str(NumberOfTriplesPerSide*3 + 1))
    print('NumberOfDataSubcarriers        = ' + str(NumberOfDataSubcarriers))
    print('NumberOfTotalSubcarriers       = ' + str(NumberOfDataSubcarriers + NumberOfDcSubcarriers))
    print('NumberOfDcSubcarriers          = ' + str(NumberOfDcSubcarriers))
    print('[TonePMin TonePMax]            = [' + str(TonePMin) + ' ' + str(TonePMax) + ']') 
    print('[ToneMMin ToneMMax]            = [' + str(ToneMMin) + ' ' + str(ToneMMax) + ']') 
    print('Actual Single Sided BW (Hz)    = ' + str(TonePMax * SubcarrierSpacingHz) )
    print('Guard region backoff (%)       = ' + str(100 * (TotalBw/2 - TonePMax * SubcarrierSpacingHz)/(TotalBw/2)))
    print('Max Throughput (Bits/sec)      = ' + str(MaxThroughputBitsPerSecond))
    

    # -------------------------------------------------------------------------
    # Determine signal field characteristic
    # ------------------------------------------------------------------------- 
    # 2. Determine the number of signal field symbols
    NumberOfSignalFieldSymbols = iNi + 1

    # 3. Determine the reference symbol periodicity
    ReferenceSymbolPeriodicity = list([2, 4, 8, 16])[iPi]

    # 4. Determine the number of user bits
    NumberOfUserBits           = list([16, 128, 512, 2048])[iTi]

    



if __name__ == '__main__':
    iScIndex = 0   # 0/1/2/3 = 20e3 / 40e3 / 80e3 / 160e3 Hz
    iBwIndex = 2   # 0/1/2/3 =  5   /  10  /  20  / 40    MHz
    iNi      = 0   # 0/1/2/3 = 1/2/3/4 Ofdm symbols in the signal field
    iPi      = 0   # 0/1/2/3 = Reference symbols occur every 2/4/8/16 OFDM symbols
    iCi      = 3   # 0/1/2/3 = BPSK / QSPK / 16Qam / 64Qam
    iTi      = 0   # 0/1/2/3 = 16, 128, 512, 2048 user bits in signal field
    iDi      = 1   # 0/1/2/3 =  0   /   1  /  3   /  5    DC Subcarriers that remain unoccupied
    
    ConfigureReferenceSymbol1(iScIndex
                            , iBwIndex
                            , iNi
                            , iPi
                            , iCi
                            , 0
                            , iDi)

