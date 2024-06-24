# File:   LteModDemod.py
# Author: Andreas Schwarzinger                                                                 Date: August, 8, 2020
# Notes:  The following script implements OFDM / SC-FDMA modulation and demodulation tasks for the LTE specification.

__title__     = "flexLinkModDemod"
__author__    = "Andreas Schwarzinger, Leonard Dieguez"
__status__    = "preliminary"
__version__   = "0.0.0.1"
__date__      = "Sept, 15, 2022"
__copyright__ = 'Copyright 2022 by Rohde & Schwarz USA'


# Module Imports
import numpy as np
import math
# from   scipy.fftpack  import fft, ifft 
from   flexlinkCommParameters import *
from   matplotlib     import pyplot as plt
import copy

# -----------------------------------------------------------------------------------------
# The flex link OFDM Modulation Process
# -----------------------------------------------------------------------------------------
def flexlinkModulation(LinkType:            str
                    , resource_grid:        np.ndarray   
                    , NRB:                 flexlinkNRB
                    , CP:                  flexlinkCP
                    , fftParameters:       dict
                    , startSymbolIndex:    int
                    , numberTxOfdmSymbols: int
                     ) -> np.ndarray:
    """
    brief:  This function will transform portions of the flex link frequency domain resource grid into the time domain    
    param:  LinkType            -> A string that may be 'DL' or 'SL' or 'downlink' or 'sidelink'
                                   LinkType for FPV will not be sidelink uplink and downlink are symetrical and only use
                                   downlink for LinkType
    
    param:  resource_grid       -> A complex input LTE compliant resource grid featuring resource elements, input by the
                                   calling function contains data + pilots

    param:  NRB                 -> The number of resource blocks as type flexlinkNRB
    param:  CP                  -> A cyclic prefix (equal to guard interval, Tg) enum object of type flexlinkCP
    param:  fftParameters       -> Holds FFT based parameters like, FFT_Size, SampleRate, CP sample numbers
    param:  startSymbolIndex    -> The index of the first OFDM symbols we wish to transform with respect to the resource 
                                   grid start, normally zero
    param:  numberTxOfdmSymbols -> The number of OFDM symbols we wish to transform
    output: An flexlink compatible time domain waveform
    """
    
    # -----------------------------------------------------------------------------------------
    # > Type and Error checking
    # -----------------------------------------------------------------------------------------
    assert type(LinkType) == str,                    'The LinkType must be a string.'
    LinkType = LinkType.lower()
    if LinkType == 'sl': LinkType = 'sidelink'
    if LinkType == 'dl': LinkType = 'downlink'
    ProperLinkType = LinkType == 'downlink' or LinkType == 'sidelink'
    assert ProperLinkType,                           'The LinkType input string is invalid'
    ProperGridType = type(resource_grid == np.ndarray) and (type(resource_grid[0][0]) == np.complex64 or \
                                                           type(resource_grid[0][0]) == np.complex128())
    assert ProperGridType,                           'The resource grid type is unsupported'
    assert type(NRB) == flexlinkNRB,                     'The NRB input must be of type flexlinkNRB'
    assert flexlinkNRB.CheckValidBandwidth(NRB),         'The number of resource blocks is invalid'
    assert type(CP) == flexlinkCP,                       'The Cyclic Prefix input argument type is invalid'
    assert type(fftParameters) == dict,              'The fftParameters input argument type is invalid'
    assert type(startSymbolIndex) == int,            'The StartSymbolIndex input argument type is invalid'
    assert type(numberTxOfdmSymbols) == int,         'The NumberOfOutputSymbols input argument type is invalid'

    NUM_SYMBOLS_PER_SLOT                    = CP.GetNumSymbolsPerSlot()
    NumSubcarriers, NumAvailableOfdmSymbols = resource_grid.shape
    
    assert NumSubcarriers == NRB.GetNumSubcarriers(),  'NUM_SUBCARRIERS in the resource grid does not agree with the bandwidth definition'
    assert StartSymbolIndex < NumAvailableOfdmSymbols, 'The start symbol index is out of range'
    
    # Truncate the number of requested ofdm symbols if it exceeds the number of symbols available
    if NumberTxOfdmSymbols > NumAvailableOfdmSymbols - StartSymbolIndex:
        NumberTxOfdmSymbols = NumAvailableOfdmSymbols - StartSymbolIndex

    # --------------------------------------------------------------------------------------------
    # > Extract FFT based parameters
    # --------------------------------------------------------------------------------------------
    N_FFT         = fftParameters['fftSize']
    # SampleRate    = fftParameters['SampleRate']
    # CPSamples, number of adc samples, is based on time of CP at the sample rate
    # ex: 5.2us * 11 * 1.92e6 = ~99 samples, 
    # for LTE the number of samples have a common multiple of 9
    if(CP.name == 'CPNormal'):
        CPSamples = fftParameters['CPNormal']
    else:
        CPSamples = fftParameters['CPExtended']

    # -----------------------------------------------------------------------------------------
    # > Determine the size of output waveform array and instantiate it
    # -----------------------------------------------------------------------------------------
    # Iterate through all symbols to be transformed and count all the samples
    NumOutputSamples = 0
    for SymbolIndex in range(StartSymbolIndex, StartSymbolIndex + NumberTxOfdmSymbols): 
        SymbolInSlot      = np.mod(SymbolIndex, NUM_SYMBOLS_PER_SLOT)
        NumSampleInCP     = CPSamples[SymbolInSlot]
        NumOutputSamples += NumSampleInCP + N_FFT  # add the Ts = Tu + Tg (Tcp)

    # Instantiate the output waveform
    OutputWaveform = np.zeros([NumOutputSamples], dtype=np.complex64)

    # ----------------------------------------------------------------------------------------
    # > Prepare indexing ahead of IFFT operation
    # ----------------------------------------------------------------------------------------
    # 1. Determine the indices of the positive and negative frequency subcarriers in the input Resource Grid.
    HalfNumSubcarriers = int(np.floor(NumSubcarriers/2))
    PosFreqSubcarriers = range(HalfNumSubcarriers, NumSubcarriers)  # Subcarrier Indices of positive frequencies
    NegFreqSubcarriers = range(0, HalfNumSubcarriers)               # Subcarrier Indices of negative frequencies

    # 2. Determine the indices of the positive and negative frequency subcarriers in the IFFT input.
    PosFreqIfftIndices = range(1, HalfNumSubcarriers + 1)           # Ifft indices mapped to positive frequencies
    NegFreqIfftIndices = range(N_FFT - HalfNumSubcarriers, N_FFT)   # and negative frequencies


    # ---------------------------------------------------------------------------------------
    # > Transform the resource grid into the time domain
    # ---------------------------------------------------------------------------------------
    # The OutputSymbolStartIndex indicates where each new Ofdm Symbols starts in the output sequence
    OutputSymbolStartSampleIndex = 0
    for SymbolIndex in range(StartSymbolIndex, StartSymbolIndex + NumberTxOfdmSymbols):
        # Load and compute the IFFT
        IFFT_Input                     = np.zeros([N_FFT], dtype = np.complex64)
        IFFT_Input[PosFreqIfftIndices] = resource_grid[PosFreqSubcarriers, SymbolIndex] 
        IFFT_Input[NegFreqIfftIndices] = resource_grid[NegFreqSubcarriers, SymbolIndex]
        IFFT_Output                    = N_FFT * ifft(IFFT_Input, N_FFT)

        # Generate the cyclic prefix, which are the last few samples of the IFFT output
        SymbolInSlot         = np.mod(SymbolIndex, NUM_SYMBOLS_PER_SLOT)
        NumSamplesInCP       = CPSamples[SymbolInSlot]      
        CP                   = IFFT_Output[N_FFT - NumSamplesInCP:N_FFT]

        # Generate the Ofdm symbol in the time domain
        OfdmSymbol           = np.hstack([CP, IFFT_Output])

        # Add the 7.5KHz complex sinusoid to the modulation process for sidelink operation
        if(LinkType == 'sidelink'):
            Indices     = np.arange(-NumSamplesInCP, N_FFT, 1, dtype = np.float32)
            f           = np.float32(0.5 / N_FFT)
            Sinusoid    = np.exp(1j*2*3.1415926536*Indices*f) 
            OfdmSymbol *= Sinusoid

        # Insert the Ofdm Symbol into the Output Waveform Array
        OutputIndices                 = range(OutputSymbolStartSampleIndex,OutputSymbolStartSampleIndex + len(OfdmSymbol))
        OutputWaveform[OutputIndices] = OfdmSymbol
        OutputSymbolStartSampleIndex += len(OfdmSymbol)
        
    # -----------------------------------------------------------------------------------
    # > Verify the output waveform length and return it to the calling function.
    # -----------------------------------------------------------------------------------
    assert len(OutputWaveform) == NumOutputSamples, 'The output samples were not properly mapped into the complete Outputwaveform'
    return OutputWaveform




# -----------------------------------------------------------------------------------------
# The Lte OFDM Demodulation Process
# -----------------------------------------------------------------------------------------
def flexlinkDemodulation(LinkType:               str
                      , InputWaveform:          np.ndarray
                      , NRB:                    flexlinkNRB
                      , CP:                     flexlinkCP
                      , fftParameters:          dict
                      , SubframeStartTime:      float = 0
                      , StartSymbolIndex:       int   = 0
                      , NumberOfOutputSymbols:  int   = None
                      , TimeAdvanceInSec:       float = 0
                      , CompensateForAdvance:   bool  = True ) -> np.ndarray:   
    """
    brief:  This function will transform portions of the LTE downlink time domain waveform into the frequency domain   
    param:  LinkType              -> A string that may be 'DL' or 'SL' or 'downlink' or 'sidelink'
    param:  InputWaveform         -> The complex input waveform sequence
    param:  NRB                   -> The number of downlink/sidelink resource blocks
    param:  CP                    -> A cyclic prefix enum object
    param:  fftParameters         -> Holds FFT based parameters like, FFT_Size, SampleRate, CP sample numbers
    param:  SubframeStartTime     -> Known start time of the a particular subframe where we want to begin demodulation 
    param:  StartSymbolIndex      -> The index of the first OFDM symbols we wish to transform with respect to the subframe start time 
    param:  NumberOfOutputSymbols -> The number of OFDM symbols we wish to transform
    param:  TimeAdvanceInSec      -> Time that we pull the FFT start time instance into the cyclic prefix to avoid inter symbol interference.
    param:  CompensateForAdvance  -> Should we compensate for the advance or not? Usually we will want to do it.
    output: An Lte compliant resource grid
    note:   Given the SubframeStartTime, we believe that the IFFT portion of the first OFDM symbol in the subframe is only CP away.
    """
    
    # -----------------------------------------------------------------------------------------
    # > Type checking
    # -----------------------------------------------------------------------------------------
    assert type(LinkType) == str,                  'The LinkType must be a string.'
    LinkType = LinkType.lower()
    if LinkType == 'sl': LinkType = 'sidelink'
    if LinkType == 'dl': LinkType = 'downlink'
    ProperLinkType = LinkType == 'downlink' or LinkType == 'sidelink'
    assert ProperLinkType,                         'The link type is not supported.'
    assert type(InputWaveform) == np.ndarray,      'The InputWaveform must be of type np.ndarray'
    assert type(InputWaveform[0]) == np.complex64 or type(InputWaveform[0]) == np.complex128, \
                                                   'The entries InputWaveform must be of type np.complex64'
    assert type(NRB) == flexlinkNRB,                   'The NRB input must be of type flexlinkNRB'
    assert flexlinkNRB.CheckValidBandwidth(NRB),       'The number of resource blocks is invalid'
    assert type(CP) == flexlinkCP,                     'The Cyclic Prefix input argument type is invalid'
    assert type(fftParameters) == dict,            'The fftParameters input argument type is invalid'
    assert type(SubframeStartTime) == float,       'The StartTime input argument type is invalid'
    assert type(StartSymbolIndex) == int,          'The StartSymbolIndex input argument type is invalid'
    if NumberOfOutputSymbols != None:
        assert type(NumberOfOutputSymbols) == int, 'The NumberOfOutputSymbols input argument type is invalid'
    

    NUM_SYMBOLS_PER_SLOT = CP.GetNumSymbolsPerSlot()
    NUM_SUBCARRIERS      = NRB.GetNumSubcarriers()

    assert StartSymbolIndex < 2*NUM_SYMBOLS_PER_SLOT,  'The start symbol index must not exceed the max number of symbols in one subframe'
 
    # --------------------------------------------------------------------------------------------
    # > Extract FFT based parameters
    # --------------------------------------------------------------------------------------------
    N_FFT         = fftParameters['fftSize']
    SampleRate    = fftParameters['SampleRate']
    SamplePeriod  = 1/SampleRate
    if(CP.name == 'CPNormal'):
        CPSamples = fftParameters['CPNormal']
    else:
        CPSamples = fftParameters['CPExtended']

    assert TimeAdvanceInSec < CPSamples[0] and TimeAdvanceInSec >= 0, 'The Time advance is out of its normal range of 0 through CPSamples[0].' 

    # ----------------------------------------------------------------------------------------------
    # > Figure out how many symbols we need to transform
    # ----------------------------------------------------------------------------------------------
    NumInputSamples            = len(InputWaveform)
    TotalWaveformPeriod        = NumInputSamples * SamplePeriod
    AvailableWaveformPeriod    = TotalWaveformPeriod - SubframeStartTime
    NumFullSlotsInWaveform     = np.floor(AvailableWaveformPeriod/Params.LTE_SLOT_PERIOD)
    TimeRemainingInLastSlot    = AvailableWaveformPeriod - NumFullSlotsInWaveform * Params.LTE_SLOT_PERIOD

    # Determine how many symbols are present in the last slot
    NumSymbolsInLastSlot  = 0
    AccumulatedSymbolTime = 0
    for SymbolIndex in range(0, NUM_SYMBOLS_PER_SLOT):
        CurrentSymbolPeriod    = Params.LTE_IFFT_PERIOD + CPSamples[SymbolIndex] / SampleRate
        AccumulatedSymbolTime += CurrentSymbolPeriod
        if TimeRemainingInLastSlot >= AccumulatedSymbolTime:
            NumSymbolsInLastSlot += 1
        else:
            break    

    NumSymbolsInWaveform  = int(NumFullSlotsInWaveform * NUM_SYMBOLS_PER_SLOT + NumSymbolsInLastSlot)

    # If Needed, truncate the requested number of output symbols
    if NumberOfOutputSymbols == None or NumberOfOutputSymbols > NumSymbolsInWaveform:
        NumberOfOutputSymbols = NumSymbolsInWaveform
    

    # ------------------------------------------------------------------------------------------------
    # > Instantiate the output resource grid
    # ------------------------------------------------------------------------------------------------
    output_resource_grid = np.zeros([NRB.GetNumSubcarriers(), int(NumberOfOutputSymbols)], dtype=np.complex64)

    # ------------------------------------------------------------------------------------------------
    # > Run the Ofdm Demodulation
    # ------------------------------------------------------------------------------------------------
    CurrentSymbolStartTime = SubframeStartTime - TimeAdvanceInSec
    for OutputSymbolIndex in range(0, StartSymbolIndex + NumberOfOutputSymbols):
       CurrentSymbolPeriod = Params.LTE_IFFT_PERIOD + CPSamples[np.mod(OutputSymbolIndex, NUM_SYMBOLS_PER_SLOT)] * SamplePeriod

       if OutputSymbolIndex < StartSymbolIndex:  
           CurrentSymbolStartTime += CurrentSymbolPeriod
           continue                                           # Iterate until the first OFDM symbol to be transformed
        
       # At this point we have reached the start time of the first symbol that we wish to transform. We now determine
       # the sample range of the IFFT portion of the OFDM symbol. It is the sample associated with this range that we
       # want to transform.
       # -> Find the first sample in the current OFDM symbols
       FirstSampleIndex       = math.floor(CurrentSymbolStartTime*SampleRate)   # Casting to int causes a floor operation
       
       # Increment to the first sample of the IFFT portion of the current OFDM symbol
       SymbolInSlot           = np.mod(OutputSymbolIndex, NUM_SYMBOLS_PER_SLOT)
       FirstIfftSampleIndex   = FirstSampleIndex + CPSamples[SymbolInSlot]  # The index of the first and
       CurrentIfftStartTime   = CurrentSymbolStartTime + CPSamples[SymbolInSlot] * SamplePeriod
       LastIfftSampleIndex    = FirstIfftSampleIndex + N_FFT - 1  # last sample of the IFFT portion of the OFDM symbol

       # Extract the IFFT portion of the current output symbol from the input time domain waveform
       ExtractedSamples       = copy.deepcopy(InputWaveform[FirstIfftSampleIndex:LastIfftSampleIndex+1])

       # ------------------------------------------------------------------------------------------------------------
       # The current symbol start time does not necessarily fall exactly on an available sample. It may well be in 
       # between available samples. In that case, the time associated with FirstSampleIndex is slightly earlier than
       # CurrentIfftStartTime. If we take the FFT at an earlier time, then it equivalently looks like we are taking
       # the FFT at the original time, but the waveform was delayed. This apparent delay is increased by the TimeAdvance
       # that we want to insert as the waveform looks even more delayed.
       # We need to compensate the FFT output for the apparent delay due to fractional sample time difference and may 
       # want to do the same for the apparent delay caused by the the additional time shift via 'CompensateForAdvance'.
       # Remember the time shift property of the FT -> FT(x(t-TimeShift)) --> X(f) * exp(-j2piTimeShift*f)
       # TimeShift > 0 represents a delay, whereas TimeShift < 0 represent a time advance.
       # In order to make up for this apparent delay, we simply apply the time shiting property of the FT to the output
       # of the FFT using a TimeShift value that is the opposite (or minus) of the apparent delay.
       TimeShift = 0 
       if CompensateForAdvance == True:
            TimeShift += CurrentIfftStartTime - FirstIfftSampleIndex * SamplePeriod  # Due to fractional sample offset
            TimeShift += TimeAdvanceInSec                                      # Due to the desired time advance
       CompensatingTimeShift = -TimeShift

       # Remove the 7.5KHz complex sinusoid in the demodulation process for sidelink operation
       if(LinkType == 'sidelink'):
            Indices              = np.arange(0, N_FFT, 1, dtype = np.float32) + CompensatingTimeShift / SamplePeriod
            f                    = np.float32(-0.5 / N_FFT)
            Sinusoid             = np.exp(1j*2*3.1415926536*Indices*f) 
            ExtractedSamples    *= Sinusoid 

       FFT_Output                = fft(ExtractedSamples, N_FFT) / N_FFT


       # Apply the time shift to the positive frequencies
       HalfTheSubcarriers        = int(NUM_SUBCARRIERS/2) 
       PosFrequencies            = np.arange(1, HalfTheSubcarriers + 1) * Params.LTE_SC_SPACING
       PosIndexRange             = range(1, HalfTheSubcarriers + 1)  # Remember, the DC carrier is empty
       FFT_Output[PosIndexRange] = FFT_Output[PosIndexRange] * np.exp(-1j*2*np.pi*CompensatingTimeShift*PosFrequencies)
       
       # Apply the time shift to the negative frequencies
       NegFrequencies            = np.arange(-HalfTheSubcarriers, 0) * Params.LTE_SC_SPACING
       NegIndexRange             = range(N_FFT - HalfTheSubcarriers, N_FFT)
       FFT_Output[NegIndexRange] = FFT_Output[NegIndexRange] * np.exp(-1j*2*np.pi*CompensatingTimeShift*NegFrequencies)

       # It is now time to map the FFT_Output into to the resource grid
       PosFrequencyIndices       = range(HalfTheSubcarriers, NUM_SUBCARRIERS)
       NegFrequencyIndices       = range(0, HalfTheSubcarriers)
       output_resource_grid[PosFrequencyIndices, OutputSymbolIndex - StartSymbolIndex] = FFT_Output[PosIndexRange]
       output_resource_grid[NegFrequencyIndices, OutputSymbolIndex - StartSymbolIndex] = FFT_Output[NegIndexRange]

       CurrentSymbolStartTime += CurrentSymbolPeriod

    # Return the output
    return output_resource_grid


# --------------------------------------------------------------
# > GeneratePreambleA()
# --------------------------------------------------------------
def GeneratePreambleA(SampleRate: float = 20.48e6) -> np.ndarray:
    """
    This function generates the PreambleA Waveform
    """
    CosineFrequencyA = 4*160e3
    CosineFrequencyB = 12*160e3
    Ts = 1/SampleRate
    NumSamples = math.floor(220e-6 / Ts)
    Time  = np.arange(0, NumSamples*Ts, Ts, dtype = np.float64)
    Tone1 = np.exp( 1j*2*np.pi*CosineFrequencyA*Time, dtype = np.complex64)
    Tone2 = np.exp( 1j*2*np.pi*CosineFrequencyB*Time, dtype = np.complex64)
    Tone3 = np.exp(-1j*2*np.pi*CosineFrequencyA*Time, dtype = np.complex64)
    Tone4 = np.exp(-1j*2*np.pi*CosineFrequencyB*Time, dtype = np.complex64)
    PreambleA = (1/4) * Tone1 + Tone2 + Tone3 + Tone4
    return PreambleA, Time



#  ----------------------------------------------------------------------------
#  >  GeneratePreambleA()
#  ----------------------------------------------------------------------------
def  ProcessPreambleA(RxPreambleA:       np.ndarray,
                      SampleRate:        float =  20.48e6,
                      bHighCinr:         bool  =  False,
                      bShowPlots:        bool  =  False)  ->  float :
    """
    This  function  estimates  the  frequency  offset  in  the  PreambleA  Waveform
    """
    #  ------------------------------------------------------------------------
    #  Error  checking
    #  ------------------------------------------------------------------------
    FFT_Size    =  4096

    assert  np.issubdtype(RxPreambleA.dtype,  np.complexfloating),  'Error.'
    assert  len(RxPreambleA.shape)  ==  1,                          'Error.'
    assert  len(RxPreambleA)  >=  FFT_Size,                         'Error.'
    assert  isinstance(bHighCinr,  bool),                           'Error.'
    assert  isinstance(bShowPlots,  bool),                          'Error.'

    #  ------------------------------------------------------------------------
    #  Overlay  the  Hanning  Window  to  induce  more  FFT  leakage
    #  ------------------------------------------------------------------------
    N              =  FFT_Size
    n              =  np.arange(0,  N,  1,  dtype  =  np.int32)
    Hanning        =  0.5  -  0.5  *  np.cos(2*np.pi  *  (n  +  1)  /  (N  +  1))
    RxWaveform     =  RxPreambleA[0:N].copy()  *  Hanning

    #  ------------------------------------------------------------------------
    #  Take  the  FFT  and  rearrange  its  output  such  that  negative  frequency  bins  are  first
    #  ------------------------------------------------------------------------
    ZeroIndex       =  int(FFT_Size/2)
    FFT_Output      =  np.fft.fft(RxWaveform[0:FFT_Size])
    FFT_Rearranged  =  np.hstack([FFT_Output[ZeroIndex  :  ],  FFT_Output[:ZeroIndex]])  # np.fft.fftshift ?

    #  ------------------------------------------------------------------------
    #  Find  all  peak  bin  indices
    #  ------------------------------------------------------------------------
    MaxIndex        =  np.argmax(abs(FFT_Rearranged))
    OffsetIndex      =  MaxIndex  -  ZeroIndex
    #  PeakIndexDeltas  is  the  distance  in  bins  of  all  peaks  relative  to  the  most
    #  negative  peak  index
    PeakIndexDeltas  =  np.array([0,  256,  512,  768])

    #  Now  that  we  have  the  peak,  we  need  to  figure  out  the  indices  of  the  other  peaks
    if    OffsetIndex  <  -320:  #  Then  the  maximum  peak  is  the  one  belonging  to  -1920MHz
            PeakIndices  =  MaxIndex  +  PeakIndexDeltas
    elif  OffsetIndex  >=  -320  and  OffsetIndex  <  0:    #  MaxPeak  at  -640MHz
            PeakIndices  =  MaxIndex  +  PeakIndexDeltas  -  PeakIndexDeltas[1]
    elif  OffsetIndex  <=  320  and  OffsetIndex  >=  0:    #  MaxPeak  at  +640MHz
            PeakIndices  =  MaxIndex  +  PeakIndexDeltas  -  PeakIndexDeltas[2]
    elif  OffsetIndex  >    320:                        #  MaxPeak  at  +1920MHz
            PeakIndices  =  MaxIndex  +  PeakIndexDeltas  -  PeakIndexDeltas[3]

    #  ------------------------------------------------------------------------
    #  Use  maximum  ratio  combining  to  compute  the  frequency  offset
    #  ------------------------------------------------------------------------
    MRC_Scaling  =  FFT_Rearranged[PeakIndices]  *  np.conj(FFT_Rearranged[PeakIndices])

    Sum0  =  np.sum(FFT_Rearranged[PeakIndices  -  3]  *  MRC_Scaling)
    Sum1  =  np.sum(FFT_Rearranged[PeakIndices  -  2]  *  MRC_Scaling)
    Sum2  =  np.sum(FFT_Rearranged[PeakIndices  -  1]  *  MRC_Scaling)
    Sum3  =  np.sum(FFT_Rearranged[PeakIndices  +  0]  *  MRC_Scaling)
    Sum4  =  np.sum(FFT_Rearranged[PeakIndices  +  1]  *  MRC_Scaling)
    Sum5  =  np.sum(FFT_Rearranged[PeakIndices  +  2]  *  MRC_Scaling)
    Sum6  =  np.sum(FFT_Rearranged[PeakIndices  +  3]  *  MRC_Scaling)

    #  Note  that  the  sample  rate  of  20.48MHz  /  4096  results  in  tone  spacing  of  5KHz.  If
    #  the  frequency  offset  is  beyond  5  KHz,  then  we  must  adjust  the  recreating
    #  frequencies  below.  If  the  frequency  offset  is  less  than  2.5KHz  away,  then  the
    #  peaks  will  be  at  [1664,  1920,  2176,  2432]
    SubcarrierOffset  =  PeakIndices[0]  -  1664

    n  =  np.array([(2048  -  256),  (2048  +  256)])
    N  =  FFT_Size
    if  bHighCinr  ==  False:
        #  Coarse  reconstruction  of  waveform  uses  fewer  FFT  results
        Tone  = np.exp(  1j*2*np.pi*n*(SubcarrierOffset-2)/N)    *  Sum1  +  \
                np.exp(  1j*2*np.pi*n*(SubcarrierOffset-1)/N)    *  Sum2  +  \
                np.exp(  1j*2*np.pi*n*(SubcarrierOffset  )/N)    *  Sum3  +  \
                np.exp(  1j*2*np.pi*n*(SubcarrierOffset+1)/N)    *  Sum4  +  \
                np.exp(  1j*2*np.pi*n*(SubcarrierOffset+2)/N)    *  Sum5
    else:
        #  Higher  resolution  reconstruction  of  waveform  uses  more  FFT  results
        Tone  = np.exp(  1j*2*np.pi*n*(SubcarrierOffset-3)/N)    *  Sum0  +  \
                np.exp(  1j*2*np.pi*n*(SubcarrierOffset-2)/N)    *  Sum1  +  \
                np.exp(  1j*2*np.pi*n*(SubcarrierOffset-1)/N)    *  Sum2  +  \
                np.exp(  1j*2*np.pi*n*(SubcarrierOffset  )/N)    *  Sum3  +  \
                np.exp(  1j*2*np.pi*n*(SubcarrierOffset+1)/N)    *  Sum4  +  \
                np.exp(  1j*2*np.pi*n*(SubcarrierOffset+2)/N)    *  Sum5  +  \
                np.exp(  1j*2*np.pi*n*(SubcarrierOffset+3)/N)    *  Sum6

    Rotation1    =  Tone[1]  *  np.conj(Tone[0])

    AngleRad    =  np.angle(Rotation1)
    AngleCycles  =  AngleRad  /  (2*np.pi)
    FreqOffset  =  AngleCycles  *  SampleRate  /  512

    if  bShowPlots:
        print('Frequency  Offset  =  '  +  str(FreqOffset)  +  '  Hz')
        print('MaxIndex  =    '  +  str(MaxIndex))
        print('OffsetIndex  =  '  +  str(OffsetIndex))
        print(PeakIndices)

        plt.figure(1)
        plt.stem(np.arange(0,  len(FFT_Rearranged)),  np.abs(FFT_Rearranged))
        plt.grid(True)
        plt.show()

    return  FreqOffset


# --------------------------------------------------------------
# > generatePreambleB()
# --------------------------------------------------------------
def generatePreambleB(SampleRate: float = 20.48e6) -> np.ndarray:
    """
    This function generates the PreambleB Waveform
    """
    Nzc = 331
    # lwd added
    cf = Nzc%2
    q = 0
    scPositive = int(np.ceil(Nzc/2))
    scNegative = int(np.floor(Nzc/2))
    u1 = 34
    n = np.arange(0,Nzc) # cannot use range with complex numbers

    # Definition of the Zadoff-Chu Sequence
    # zc = np.exp(-1j*np.pi*u1*n* (n+1)/Nzc)
    zc = np.exp(-1j*np.pi*u1*n*(n+cf+2.0*q)/Nzc)

    # The Fourier Transform of the Zadoff-Chu Sequence
    preambleB_FFT = (1/np.sqrt(Nzc))*np.fft.fft(zc)
    # ------------------------------------
    # Mapping into IFFT buffer and upsampling
    # ------------------------------------
    # Loading the PreambleB_FFT sequence into the IFFT buffer (N=1024) for rendering into the time domain
    IFFT_Buffer1024                     = np.zeros(1024, dtype='complex')
    IFFT_Buffer1024[0:scPositive]       = preambleB_FFT[0:scPositive]          
    IFFT_Buffer1024[-scNegative:]       = preambleB_FFT[Nzc-scNegative:Nzc]
    # IFFT_Buffer1024[0]                  = 0
    preambleB                           = np.sqrt(1024)*np.fft.ifft(IFFT_Buffer1024)

    # IFFT_Buffer1024[0:Nzc]  = zc
    # IFFT_Buffer1024 = np.fft.fft(IFFT_Buffer1024)
    # preambleB = np.sqrt(1024)*np.fft.ifft(IFFT_Buffer1024)


    zc_corr = np.correlate(zc,zc)
    plt.figure()
    plt.plot(abs(zc_corr)/len(zc))
    plt.grid('on')
    plt.show(block=False)



    plt.figure()
    plt.plot(np.real(zc), np.imag(zc), 'r.')
    plt.grid('on')
    plt.show(block=False)

    plt.figure()
    plt.plot(np.real(zc))
    plt.plot(np.imag(zc))
    plt.grid('on')
    plt.show(block=False)

    plt.figure()
    plt.plot(20*np.log10(np.abs(zc)))
    plt.grid('on')
    plt.show(block=False)

    plt.figure()
    plt.plot(20*np.log10(np.abs(preambleB_FFT)))
    plt.grid('on')
    plt.show(block=False)


    
    plt.figure()
    plt.plot(np.real(preambleB), np.imag(preambleB), 'r.')
    plt.grid('on')
    plt.show(block=False)

    plt.figure()
    plt.plot(np.real(preambleB))
    plt.plot(np.imag(preambleB))
    plt.grid('on')
    plt.show(block=False)


    return preambleB



# --------------------------------------------------------------------------------------------------
# The test bench
# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    preamble_a, time_vec = GeneratePreambleA()
    ProcessPreambleA(preamble_a,
                    SampleRate =  20.48e6,
                    bHighCinr  =  False,
                    bShowPlots  =  True) 

    preamble_b = generatePreambleB()


    Test = 'OFDM'       #  only using 'OFDM' (not using 'SC_FDMA')

    if(Test == 'OFDM'):
        input_resource_grid       = np.ones([1200, 2], dtype=np.complex64)
        NRB                     = flexlinkNRB.BwIs20MHz
        CP                      = flexlinkCP.CPNormal
        fftParams               = flexlinkfftSizesCP.fftSize1024.value
        NumSamples              = int(np.floor(50.1e-3 * fftParams['SampleRate']))
        StartSymbolIndex        = 0
        NumberOfdmSymbols       = 2

        OutputWaveform = flexlinkModulation('sidelink'
                                         , input_resource_grid   
                                         , NRB
                                         , CP
                                         , fftParams
                                         , StartSymbolIndex
                                         , NumberOfdmSymbols
                                          ) 

        #fig = plt.figure(1)
        #plt.plot(OutputWaveform.real, 'r', OutputWaveform.imag, 'b')
        #plt.title('Output Waveform')
        #plt.grid(True)

        if(0):
            # This example simply allows us to play with the SubframeStartTime, when the first sample 
            #in the waveform isn't the start of the first subframe.
            SampleDelay             = 1
            OutputWaveformShift     = np.roll(OutputWaveform, -SampleDelay) # Use negative SampleDelay to simulate advance
            SubframeStartTime       = -1/21.12e6 
            StartSymbolIndex        = 0
            NumberOfOutputSymbols   = 2
            TimeAdvanceInSec        = 0 
            CompensateForAdvance    = False
        else:
            # This example simply shows that the TimeAdvance and CompensateForAdvance features work
            SampleDelay             = 1
            OutputWaveformShift     = np.roll(OutputWaveform, -SampleDelay) # Use negative SampleDelay to simulate advance
            SubframeStartTime       = -1/21.12e6
            StartSymbolIndex        = 0
            NumberOfOutputSymbols   = 2
            TimeAdvanceInSec        = 2/21.12e6   # Some Time advance that we desire
            CompensateForAdvance    = True        # Compensate for the advance
        
        output_resource_grid = flexlinkDemodulation('sidelink'
                                               , OutputWaveformShift
                                               , NRB
                                               , CP
                                               , fftParams
                                               , float(SubframeStartTime)
                                               , StartSymbolIndex
                                               , NumberOfOutputSymbols
                                               , TimeAdvanceInSec
                                               , CompensateForAdvance
                                                )

        fig = plt.figure(2)
        plt.plot(output_resource_grid[:,0].real, 'r', output_resource_grid.imag[:,0], 'b')
        plt.title('Output Grid')
        plt.grid(True)
        plt.show()

        # -------------------------------------------
        # Check the input and output resource grids
        # -------------------------------------------
        Difference = input_resource_grid[:,0] - output_resource_grid[:,0]
        MaxError   = np.max(np.abs(Difference))
        print('MaxError = ' + str(MaxError))
 
    print('end')
        
        