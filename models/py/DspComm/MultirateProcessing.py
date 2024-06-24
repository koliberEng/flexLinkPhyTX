# File:        MultirateProcessing.py
# Content:     1. Halfband Filter Design and Implementation
# Interpoator: 2. See Notes one for a description
# Notes1:     This script enables interpolation of input sequences.
#             -> The bulk of the interpolation shall be done using an
#                all-pass FIR filter structure.
#             -> At the beginning and at the end of the waveform, the
#                algorithm will use Lagrange interpolation, as the 
#                all pass filter structure needs to see an almost 
#                equal number of sample to the right and left of 
#                the interpolation time.
#             -> The larger the number of taps the larger the bandwidth
#                over which we can accurately interpolate.
#                 8 taps allow interpolation to +/- 0.38 * SamplingRate
#                16 taps allow interpolation to +/- 0.42 * SamplingRate
#                32 taps allow interpolation to +/- 0.45 * SamplingRate
# Note2:      This script implements a 33 tap halfband filter that is
#             eventually destined for AVX2 implementation in C++


__title__     = "MultirateProcessing"
__author__    = "Andreas Schwarzinger"
__status__    = "preliminary"
__version__   = "0.2.0"
__date__      = "March, 1st, 2023"
__copyright__ = 'Andreas Schwarzinger'

# ----------------------------------------- #
# Import statements                         #
# ----------------------------------------- #
import numpy             as     np
import math
import matplotlib.pyplot as     plt
# import Resampler








# -------------------------------------------------------------------------------------------------- #
#                                                                                                    #
# > CHalfbandfilter                                                                                  #
#                                                                                                    #
# -------------------------------------------------------------------------------------------------- #
class CHalfbandFilter():
    '''
    This class will compute and run a decimation by 4 Halfband filter
    '''
    def __init__( self
                , NumberOfTaps:    int = 31
                , bHanningOverlay: bool = True):
        '''
        brief: Initialize the CHalfbandFilter
        param: NumberOfTaps    - The number of FIR taps
        param: bHanningOverlay - Multiply the impulser response by a Hanning window 
        '''
        # Error checking
        assert isinstance(NumberOfTaps, int)
        assert (NumberOfTaps == 7  or NumberOfTaps == 15) or \
               (NumberOfTaps == 31 or NumberOfTaps == 63)
        assert isinstance(bHanningOverlay, bool)
        

        # ---------------------------------------------
        # Compute Impulse and Frequency Response
        # ---------------------------------------------
        self.N = NumberOfTaps
        self.n = np.arange(0, self.N, 1, np.int8)
        Hann   = 0.5*np.ones(self.N, np.float32) - 0.5*np.cos(2*np.pi*(self.n+1)/(self.N+1)) 

        # The impulse response
        Arg  = self.n/2 - (self.N-1)/4
        self.h    = np.sinc(Arg)
        # Add the Hanning Overlay if desired
        if bHanningOverlay == True:
            self.h *= Hann**.4
        
        # Normalize to sum of 1.0
        self.h /= np.sum(self.h)
        
        # Compute the Frequency Response using multiple DTFT operations
        self.Frequencies = np.arange(-0.5, 0.5, 0.01, np.float32)
        self.FResponse   = np.zeros(len(self.Frequencies), np.complex64)
        self.FResponsedB = np.zeros(len(self.Frequencies), np.float32)
        for Index, f in enumerate(self.Frequencies):
            AnalysisTone            = np.exp(-1j*2*np.pi*self.n*f)
            self.FResponse[Index]   = np.sum(AnalysisTone * self.h)
            self.FResponsedB[Index] = 20*np.log10(np.absolute(self.FResponse[Index]))



    # ------------------------------------------ #
    # > Instance method: PlotHalfbandResponse()  #
    # ------------------------------------------ #
    def PlotHalfbandResponses(self):
        '''
        brief: This method will plot the impulse and frequency response of the filter
        '''
        plt.figure()
        plt.subplot(2,1,1)
        markerline, stemlines, baseline = plt.stem(self.h, markerfmt = 's')
        baseline.set_color('#aaaaaa')              # Grey
        stemlines.set_color('#cccccc')             # Light grey
        markerline.set_color('#666666')            # Dark Grey
        markerline.set_markerfacecolor('#000000')  # Black
        plt.grid(color = '#eeeeee')                # Very light Grey
        plt.title('Filter Impulse Response')
        plt.xlabel('Discrete Time n')
        plt.tight_layout()
        plt.subplot(2,1,2)
        plt.plot(self.Frequencies, self.FResponsedB,  color = 'grey')
        plt.grid(color = '#eeeeee')                # Very light Grey
        plt.title('Filter Frequency Response')
        plt.xlabel('Frequency Hz')
        plt.ylabel('dB')
        plt.xticks([-0.5, -0.25, 0, 0.25, 0.50])  
        plt.tight_layout()
        plt.show()
         


    # ------------------------------------------ #
    # > Instance method: RunHalfbandFilter()     #
    # ------------------------------------------ #
    def RunHalfbandFilter(self
                        , InputSequence: np.ndarray) -> np.ndarray:
        '''
        brief:  This function is highly efficient implementation of the halfband
                filtering process
        param:  InputSequence  - Self explanatory
        return: OutputSequence - Self explanatory
        notes:  This filter is meant to be implemented using AVX2 intrinsic 
                in C++ on a x86 processor 
        '''
        # Error checking
        assert isinstance(InputSequence, np.ndarray)
        assert np.issubdtype(InputSequence.dtype, np.inexact)
        
        OutputSequence = np.zeros(math.floor(len(InputSequence)/2), InputSequence.dtype)
        
        # 1. Savings A
        # As we would down sample anyway by a factor of two after filtering the signal
        # at the default sample rate, it is in fact only necessary to execute the 
        # computations for every 2nd sample. This way, filtering and down sampling are
        # done in one step and we eliminate 50% of calculations.
        n = np.arange(math.floor(self.N/2), len(InputSequence) - math.ceil(self.N/2), 2, np.int32) 

        # 2. Savings B and C
        # -> Ensure that you don't process taps whose coefficients are zero
        # -> Ensure that you add samples on opposite sides of the FIR structure
        #    before executing the complex multiplication.
        if self.N == 7:
            Before_n    = [  -3, -1]
            After_n     = [   3,  1]
            Indices     = range(0, 3, 2)
            h_Truncated = self.h[Indices]
        elif self.N == 15:
            Before_n    = [ -7, -5, -3, -1]
            After_n     = [  7,  5,  3,  1]
            Indices     = range(0, 7, 2)
            h_Truncated = self.h[Indices] 
        elif self.N == 31:
            Before_n    = [-15, -13, -11, -9, -7, -5, -3, -1]
            After_n     = [ 15,  13,  11,  9,  7,  5,  3,  1]
            Indices     = range(0, 15, 2)
            h_Truncated = self.h[Indices] 
        elif self.N == 63:
            Before_n    = [-31, -29, -27, -25, -23, -21, -19, -17, -15, -13, -11, -9, -7, -5, -3, -1]
            After_n     = [ 31,  29,  27,  25,  23,  21,  19,  17,  15,  13,  11,  9,  7,  5,  3,  1]
            Indices     = range(0, 31, 2)
            h_Truncated = self.h[Indices] 
        else:
            assert False, 'Invalid number of filter taps'

        for DecimatedIndex, Index in enumerate(n):
            # The addition step
            Temp   = InputSequence[Before_n + Index] + InputSequence[After_n + Index]
            # The truncated multiplication step (only 8 multiplications for real and 8 for imag)
            OutputSequence[DecimatedIndex]  = np.sum(Temp * h_Truncated)
            OutputSequence[DecimatedIndex] += self.h[math.floor(self.N/2)] * InputSequence[Index]

        return OutputSequence
    




















# -------------------------------------------------------------------------------------------------- #
#                                                                                                    #
# > CInterpolator                                                                                    #
#                                                                                                    #
# -------------------------------------------------------------------------------------------------- #
class CInterpolator():
    '''
    This class provides interpolation tasks
    '''
    # ------------------------------- #
    # > InstanceMethod(): Constructor #
    # ------------------------------- #
    def __init__(self
               , NumberOfFirTaps:      int = 16):
        '''
        brief: Sets some basic parameters and precomputes the coefficient matrix
        param: NumberOfFirTaps - The number of taps in the final FIR filter
        param: Resolution      - The number of positions between two sample
                                 times for which we will compute all-pass coefficients
        '''
        # --------------------------------- #
        # Type and Error checking           #
        # --------------------------------- #
        assert isinstance(NumberOfFirTaps, int)
        assert NumberOfFirTaps == 8 or NumberOfFirTaps == 16 or NumberOfFirTaps == 32
       
        # Transfer input arguments to instance attributes
        self.NumberOfFirTaps = NumberOfFirTaps
        
        self.Resolution      = 16   # The all-pass filters will be generated to produce
                                    # values at times [-8/16, -7/16, ... , 8/16]
                                    # Further resolution is achieved via linear interpolation

        # --------------------------------- #
        # Compute the FIR coefficients      #
        # --------------------------------- #
        # Note, that we will compute the impulse responses of the interpolation FIR.
        # This impulse response includes the FIR coefficients and the tap times.
        # The tap times will always be centered around zero for 0 delay.
        # Thus for a 8 tap filter: Tap times = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
        # The interpolation times shall always be between -0.5 and 0.5 such that
        # an equal number of source samples is available on both sides.
        # We will establish 'Resolution' coefficient sets that will interpolate values
        # at sample times -8/16, -7/16, ..., 8/16 (i.e. Resolution = 16). For any given 
        # time, we will use all-pass interpolation to get the values at the nearest times, 
        # for which we have coefficients. The remaining interpolation is linear.
        self.TapTimesIndices  = np.arange(0, self.Resolution + 1, 1, np.uint16)
        self.TapTimes         = self.TapTimesIndices * (1/self.Resolution)    
        self.TapTimes        -= 0.5       #  [-8/16, -7/16, ... , 8/16]
        self.NumberFirs       = len(self.TapTimes)

        # Compute the coefficients
        self.CoefficientMatrix8 = self.ComputeAllPassTaps(8)
        if NumberOfFirTaps > 8:
            self.CoefficientMatrix16 = self.ComputeAllPassTaps(16)
        if NumberOfFirTaps > 16:
            self.CoefficientMatrix32 = self.ComputeAllPassTaps(32)






    # ---------------------------------------- #
    # > Static Method(): ComputeCoefficients() #
    # ---------------------------------------- #
    @staticmethod
    def ComputeCoefficients(NumberOfTaps: int
                          , LocalDelay:   float) -> np.ndarray:
        '''
        brief: This function will compute the FIR filter coefficients of an all-pass filter
               with a certain delay. It exists as a generic function that computes the 
               impulse response of an all-pass filter with arbitrary time delay.
        param: NumberOfTaps  -  The number of taps of the FIR filter (a.k.a. N)
        param: LocalDelay    -  This method will design filter impulse response with the following
                                sample times: t = -(N-1)/2:1:(N-1)/2 
                                i.e.: N = 3   t = -1, 0, 1
                                i.e.: N = 4   T = -1.5, -0.5, 0.5, 1.5
                                The local delay will render the signal at position t = -LocalDelay
        '''
        # -------------------------------------------
        # Error and type checking
        assert isinstance(NumberOfTaps, int)
        assert isinstance(LocalDelay, float) or isinstance(LocalDelay, int)
        assert LocalDelay >= -0.5 and LocalDelay <= 0.5

        # -------------------------------------------
        # Basic setup
        N         = NumberOfTaps
        if N%2 == 0:
            # The normalized frequencies (i.e.: N = 8 -> [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])
            m         = np.arange(-math.floor(N/2), 0.99999 * math.floor(N/2), 1, np.float32)
            m        += 0.5
            # The normalized time
            n         = m.copy()
        else:
            # The normalized frequencies (i.e.: N = 8 -> [-3, -2, -1, 0, 1, 2, 3])
            m         = np.arange(-math.floor((N-1)/2), 1.01 * math.floor((N-1)/2), 1, np.float32)
            # The normalized time
            n         = m.copy()

        Mag       = np.ones(N, np.complex64)

        # Compute the Hanning overlay
        k         = np.arange(0, N, 1, np.int32)
        Hanning   = np.ones(N, np.float32) - np.cos(2*np.pi*(k+1)/(N+1))

        # See Digital Signal Processing in Modern Communication Systems (Edition 3) 3.3.3
        x     = -LocalDelay
        Phase = -2*np.pi*m*x/N             # Time shifting property of Fourier Transform
        
        # The frequency response of the all pass filter
        H     = Mag * np.cos(Phase) + 1j* Mag * np.sin(Phase) 
                                           
        # Compute inverse IDFT to get the impulse response of the all-pass filter 
        h     = np.zeros(N, np.complex64)
        for Index, Time in enumerate(n):
            h[Index] = np.sum(H * np.exp(1j*2*np.pi*m*Time/N)).real

        # The hanning overlay really helps a lot to reduce ripple
        Coefficients = (h * Hanning) / np.sum(h*Hanning)

        return Coefficients
    




    # ------------------------------------- #
    # > InstanceMethod(): RunInterpolator() #
    # ------------------------------------- #
    def RunInterpolator(self
                      , InputSequence:      np.ndarray
                      , SourceSampleTimes:  np.ndarray    
                ):
        '''
        brief: This function will interpolate a complex, float or integer waveform
        param: InputSequence      - A numpy array of complex, float or integer type
        param: SourceSampleTimes  - The times in source samples where to interpolate
        '''
        # --------------------------------- #
        # Type and Error checking           #
        # --------------------------------- #
        assert isinstance(InputSequence, np.ndarray)
        assert np.issubdtype(InputSequence.dtype, np.inexact) or \
               np.issubdtype(InputSequence.dtype, np.integer)
        assert isinstance(SourceSampleTimes, np.ndarray)
        assert np.issubdtype(SourceSampleTimes.dtype, np.floating) or \
               np.issubdtype(SourceSampleTimes.dtype, np.integer)
        assert len(InputSequence) >= 4, 'The input waveform length must be >= 4 samples.'
        
        self.InputSequence     = InputSequence
        self.SourceSampleTimes = SourceSampleTimes 
        OutputSequence         = np.zeros(len(SourceSampleTimes), InputSequence.dtype)

        # ---------------------------------- #
        # Set up                             #
        # ---------------------------------- #
        Length = len(InputSequence)
        for Index, SampleTime in enumerate(SourceSampleTimes): 
            # Use the Lagrange interpolator
            if SampleTime < 3  or SampleTime >= Length - 4:
                OutputSequence[Index] = self.LagrangeInterpolator(SampleTime)
                continue

            # Determine the size of the all pass filters we will be using for now
            # Use all-pass filters of length 8 
            if SampleTime >= 3 and SampleTime <= Length - 4 and Length > 8:
                NumberOfTaps      = 8
                CoefficientMatrix = self.CoefficientMatrix8
            # Use all-pass filters of length 16 
            if SampleTime >= 7 and SampleTime <= Length - 8 and self.NumberOfFirTaps > 8  \
                               and Length > math.ceil(SampleTime) + 8:
                NumberOfTaps      = 16
                CoefficientMatrix = self.CoefficientMatrix16
            # Use all-pass filters of length 32
            if SampleTime >= 15  and SampleTime <= Length - 16 and self.NumberOfFirTaps > 16 \
                                 and Length > math.ceil(SampleTime) + 16:
                NumberOfTaps      = 32
                CoefficientMatrix = self.CoefficientMatrix32


            LowerSampleIndex    = math.floor(SampleTime)
            StartSample         = LowerSampleIndex - int(NumberOfTaps/2) + 1
            UpperSample         = StartSample + NumberOfTaps
            ExtractedSamples    = InputSequence[StartSample:UpperSample]
            NewSampleTime       = SampleTime - LowerSampleIndex - 0.5
            

            assert NewSampleTime >= -0.5
            assert NewSampleTime <= 0.5

            # Find the lower and upper TapTimeIndex
            TapTimeNormalized = (NewSampleTime + 0.5) * self.Resolution
            UpperTapTimeIndex = math.ceil(TapTimeNormalized)
            LowerTapTimeIndex = math.floor(TapTimeNormalized)

            # Compute the weights for the linear interpolation
            UpperWeight       = (TapTimeNormalized - LowerTapTimeIndex)
            LowerWeight       = 1 - UpperWeight

            LowerCoefficients = CoefficientMatrix[LowerTapTimeIndex]
            UpperCoefficients = CoefficientMatrix[UpperTapTimeIndex]

            if np.issubdtype(ExtractedSamples.dtype, np.complexfloating):
                UpperValueReal = np.sum(UpperCoefficients * ExtractedSamples.real)
                UpperValueImag = np.sum(UpperCoefficients * ExtractedSamples.imag)
                LowerValueReal = np.sum(LowerCoefficients * ExtractedSamples.real)
                LowerValueImag = np.sum(LowerCoefficients * ExtractedSamples.imag)

                OutputSequence[Index] = UpperValueReal * UpperWeight + LowerValueReal * LowerWeight + \
                                1j *   (UpperValueImag * UpperWeight + LowerValueImag * LowerWeight)

            else:
                UpperValue = np.sum(UpperCoefficients * ExtractedSamples)
                LowerValue = np.sum(LowerCoefficients * ExtractedSamples)
                OutputSequence[Index] = UpperValue * UpperWeight + LowerValue * LowerWeight

        return OutputSequence






    # --------------------------------------------------------- #
    # > Instance Method: ComputeAllPassTaps()                   #
    # --------------------------------------------------------- #
    # See Digital Signal Processing in Modern Communication Systems (Edition 3) 3.3.3
    def ComputeAllPassTaps(self
                         , FilterLength: int):
        # Error and type checking
        assert isinstance(FilterLength, int)
        assert FilterLength == 8 or FilterLength == 16 or FilterLength == 32, 'Filter length is invalid'
        
        # Basic setup
        N         = FilterLength
        # The normalized frequencies (i.e.: N = 8 -> [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])
        m         = np.arange(-math.floor(N/2), 0.99999 * math.floor(N/2), 1, np.float32)
        m        += 0.5
        # The normalized time
        n         = m.copy() 

        Mag       = np.ones(N, np.complex64)

        # Compute the Hanning overlay
        k         = np.arange(0, N, 1, np.int32)
        Hanning   = np.ones(N, np.float32) - np.cos(2*np.pi*(k+1)/(N+1))

        # Establish the coefficient Matrix
        # NumRows = The number of coefficient sets we want (self.Resolution + 1)
        NumRows           = self.NumberFirs 
        NumColumns        = FilterLength
        CoefficientMatrix = np.zeros([NumRows, NumColumns], np.float32)

        # See Digital Signal Processing in Modern Communication Systems (Edition 3) 3.3.3
        for RowIndex in range (0, NumRows):    # Compute each FIR tap sequence
            x     = self.TapTimes[RowIndex]    # Usually, [-0.5, -0.5 + 1/16, ... , 0.5]
            Phase = -2*np.pi*m*x/N             # Time shifting property of Fourier Transform
            # The frequency response of the all pass filter
            H     = Mag * np.cos(Phase) + 1j* Mag * np.sin(Phase) 
                                           
            # Compute inverse IDFT to get the impulse response of the all-pass filter 
            h     = np.zeros(N, np.complex64)
            for Index, Time in enumerate(n):
                h[Index] = np.sum(H * np.exp(1j*2*np.pi*m*Time/N))

            # The hanning overlay really helps a lot to reduce ripple
            h_windowed = (h * Hanning) / np.sum(h*Hanning)

            # The set of FIR coefficients is the windowed impulse response
            CoefficientMatrix[RowIndex, :] = h_windowed.real

        return CoefficientMatrix









    # --------------------------------------------------------- #
    # > Instance Method: LagrangeInterpolator()                 #
    # --------------------------------------------------------- #
    def LagrangeInterpolator( self
                            , SampleTime: float):
        '''
        brief: Run Lagrange implementation of polynomial interpolation
        '''
        MinimumNumberOfTaps = 4   # Use no less than cubic interpolation

        # ---------------------------------------------------------------------
        # Lambda to reduce typing
        def Lagrange( SampleTime:   float
                    , SampleTimes:  float
                    , SampleValues: np.ndarray):
            # Error checking
            assert isinstance(SampleTime,   float)  
            assert isinstance(SampleTimes,  np.ndarray)
            assert isinstance(SampleValues, np.ndarray)
            assert len(SampleTimes) == len(SampleValues)
            NumberTaps = len(SampleValues)

            B = np.ones(NumberTaps, np.float32)    
            for i in range(0, NumberTaps):   # Compute each FIR tap value
                for m in range(0, NumberTaps):
                    if m == i: 
                        continue
                    x     = SampleTime
                    xm    = SampleTimes[m]
                    xi    = SampleTimes[i]
                    Temp  = (x - xm) / (xi - xm)
                    B[i] *= Temp 

            if np.issubdtype(SampleValues.dtype, np.complexfloating):
                return np.sum(B * SampleValues.real + 1j * B * SampleValues.imag)
            else:
                return np.sum(B * SampleValues)
        # ------------------------------------------------------------------

        # -------------------------------------------------
        # We are at the start of the sequence
        # -------------------------------------------------
        if SampleTime < 3:
            LowerSourceSampleTime = math.floor(SampleTime)
            # Allowing for extrapolation
            if LowerSourceSampleTime < 0:
                LowerSourceSampleTime = 0
            NumberOfTapsToUse     = 2*(LowerSourceSampleTime + 1)
            # We don't want to use too few taps for the Lagrange Interpolator
            if NumberOfTapsToUse < MinimumNumberOfTaps:
                NumberOfTapsToUse = MinimumNumberOfTaps
            TimeOffset            = (NumberOfTapsToUse - 1) / 2
            ImpulseResponseTimes  = np.arange(0, NumberOfTapsToUse, 1, np.float32) - TimeOffset
            NewSampleTime         = SampleTime - TimeOffset
            NewSampleValues       = self.InputSequence[0:NumberOfTapsToUse]

            NewSample = Lagrange( NewSampleTime
                                , ImpulseResponseTimes
                                , NewSampleValues)                         
        # -------------------------------------------------    
        # We are at the end of the sequence
        # -------------------------------------------------
        else:
            UpperSourceSampleTime = math.ceil(SampleTime)
            # Allowing for extrapolation
            if UpperSourceSampleTime >= len(self.InputSequence):
                UpperSourceSampleTime = len(self.InputSequence) - 1
            NumberOfTapsToUse     = 2*(len(self.InputSequence) - UpperSourceSampleTime)
             # We don't want to use too few taps for the Lagrange Interpolator
            if NumberOfTapsToUse < MinimumNumberOfTaps:
                NumberOfTapsToUse = MinimumNumberOfTaps
            TimeOffset            = (NumberOfTapsToUse - 1) / 2
            ImpulseResponseTimes  = np.arange(0, NumberOfTapsToUse, 1, np.float32) - TimeOffset
            NewSampleTime         = NumberOfTapsToUse - 1 - (len(self.InputSequence) - 1 - SampleTime) - TimeOffset
            NewSampleValues       = self.InputSequence[-NumberOfTapsToUse:]

            NewSample = Lagrange( NewSampleTime
                                , ImpulseResponseTimes
                                , NewSampleValues)  
            
        return NewSample







    # ------------------------------------------------------- #
    # > Static Method: ComputeFrequencyResponse() of filter   #
    # ------------------------------------------------------- #
    # This is a convenient function to have in this class as it will show me
    # the magnitude and phase response of the filter. It allows me to verify
    # the frequency range over which the interpolation is accurate
    @staticmethod
    def ComputeFrequencyResponse(FirCoefficients: np.ndarray
                               , SamplingRate: float = 1
                               , bPlot: bool = False):
        # ------------------------ #
        # Check type and errors    #
        # ------------------------ #
        assert isinstance(FirCoefficients, np.ndarray)
        assert np.issubdtype(FirCoefficients.dtype, np.inexact) # float or complex
        assert isinstance(SamplingRate, float) or isinstance(SamplingRate, int)
        assert SamplingRate > 0
        assert isinstance(bPlot, bool)

        # Recast the FIR coefficients to complex numbers
        h              = FirCoefficients.astype(np.complex64)

        # The frequencies at which we want to run the DTFT
        Frequencies    = np.arange(-0.5, 0.5, 0.005, np.float32)
        NumFrequencies = len(Frequencies)
        DTFT_Output    = np.zeros(NumFrequencies, np.complex64)

        # The time at which we assume the coefficients to be valid
        N     = len(h)
        if N%2 == 0:
            Start = -math.floor((N - 1)/2) - 0.5
            Stop  = Start + N
        else:
            Start = -math.floor((N - 1)/2) 
            Stop  = Start + N
        n     = np.arange(Start, Stop * 0.9999, 1, np.float32)

        # Run the DTFT for each frequency
        for Index, Frequency in enumerate(Frequencies):
            AnalysisTone       = np.exp(1j*2*np.pi*n*Frequency)
            DTFT_Output[Index] = np.sum(h * AnalysisTone) 

        # -------------------------- #
        # To plot or not to plot     #
        # -------------------------- #
        if bPlot == True:
            plt.figure(1)
            plt.subplot(3, 1, 1)
            plt.plot(n, h.real, 'r:')
            plt.plot(n, h.imag, 'b:')
            plt.stem(n, h.real, linefmt = 'black')
            plt.grid(True)
            plt.tight_layout()
            plt.xlabel('Time in Seconds')
            plt.title('Impulse Response of FIR Filter')
            plt.subplot(3, 1, 2)
            plt.plot(Frequencies*SamplingRate, DTFT_Output.real, 'r:')
            plt.plot(Frequencies*SamplingRate, DTFT_Output.imag, 'b:')
            plt.plot(Frequencies*SamplingRate, np.absolute(DTFT_Output), 'k')
            plt.tight_layout()
            plt.grid(True)
            plt.xlabel('Frequency in Hz')
            plt.legend(['Real', 'Imag', 'Mag'])
            plt.title('Magnitude and IQ Response of the FIR Filter')
            plt.subplot(3, 1, 3)
            plt.plot(Frequencies*SamplingRate, np.angle(DTFT_Output), 'k')
            plt.tight_layout()
            plt.grid(True)
            plt.title('Phase Response of the FIR Filter')
            plt.xlabel('Frequency in Hz')
            plt.ylabel('Radians')
            plt.show()

        return (DTFT_Output, Frequencies)
    














# ---------------------------------------------------
# Test bench
# ---------------------------------------------------
if __name__ == '__main__':
   
    Test = 0          # 0 - Full Interpolator test
                      # 1 - Static method CInterpolator.ComputeCoefficients test (Resampling)
                      # 2 - Halfband filter impulse and frquency response test
                      # 3 - Halfband filter execution test

    # ----------------------------------------------------------   
    if Test == 0:   # Full Interpolator test
    # ----------------------------------------------------------
        Interpolator = CInterpolator(NumberOfFirTaps = 32)

        # Iterate through the frequency responses of the designed filters
        if False:
            for Index in range(0, len(Interpolator.TapTimes)):
                Coefficients = Interpolator.CoefficientMatrix8[Index]
                CInterpolator.ComputeFrequencyResponse(Coefficients, 1, True)

        n                 = np.arange(0, 40, 1, np.int32)
        nHighRes          = np.arange(0, 40, 0.1, np.float32)
        SampleRate        = 1
        SineFrequency     = 0.38
        InputSequence     = np.sin(2*np.pi*SineFrequency*n/SampleRate)
        Reference         = np.sin(2*np.pi*SineFrequency*nHighRes/SampleRate)
        SourceSampleTimes = np.arange(0.25, len(InputSequence) -1, .5, np.float32)
    
        Output            = Interpolator.RunInterpolator(InputSequence, SourceSampleTimes) 

        plt.figure(1)
        plt.plot(n,         InputSequence.real,  c = 'grey', marker = 's', linestyle = '')
        plt.plot(nHighRes,  Reference.real, c = 'tab:grey')
        plt.plot(SourceSampleTimes, Output.real, 'ko')
        plt.legend(['Original Samples', 'Original Waveform', 'Interpolated Samples'])
        plt.title('Interpolation')
        plt.xlabel('Discrete time n')
        plt.grid(True)
        plt.show()

        # plt.figure(2)
        # plt.plot(n,         InputSequence.imag, 'gd')
        # plt.plot(nHighRes,  Reference.imag, 'g:')
        # plt.plot(SourceSampleTimes, Output.imag, 'ko')
        # plt.grid(True)
        # plt.show()


    # ----------------------------------------------------------
    if Test == 1:     # Static method CInterpolator.ComputeCoefficients test
    # ----------------------------------------------------------
        NumberOfTaps = 16
        LocalDelay   = 0.1
        Coefficients = CInterpolator.ComputeCoefficients(NumberOfTaps 
                                                       , LocalDelay)
        
        CInterpolator.ComputeFrequencyResponse(Coefficients, 1, True)

        # I require the following coefficient sets to implement the resampler in the P25Scanner.
        # The resampler is an interpolator that uses different coefficient sets at different times.
        # We will resample from 15.36MHz to 12.8MHz, which when divides by 1024 yield 12.5Khz
        # The source sample step is 15.36/12.8 = 1.2. We thus need the following delays
        # 0.5, 0.3, 0.1, -0.1, -0.3 and then we wrap. Theoretically, we can do without the -0.5
        # as it simply isolates a single source sample. We will compute the coefficients here anyways.
        NumberOfTaps   = 16 # We must use 16 taps as the filter fits well with AVX intrinsics
        #CoefficientsP5 =  CInterpolator.ComputeCoefficients(NumberOfTaps,  0.5).real
        #print(np.flip(CoefficientsP5))
        CoefficientsP3 =  CInterpolator.ComputeCoefficients(NumberOfTaps,  0.3).real
        print(np.flip(CoefficientsP3))
        CoefficientsP1 =  CInterpolator.ComputeCoefficients(NumberOfTaps,  0.1).real
        print(np.flip(CoefficientsP1))
        CoefficientsM1 =  CInterpolator.ComputeCoefficients(NumberOfTaps, -0.1).real
        print(np.flip(CoefficientsM1))
        CoefficientsM3 =  CInterpolator.ComputeCoefficients(NumberOfTaps, -0.3).real
        print(np.flip(CoefficientsM3))
    
        # ------------------------------------------
        # Here we will use these coefficients to implement a resampler based on the Coefficients above
        # ------------------------------------------
        NumberOfSamples = 100
        n     = np.arange(0, NumberOfSamples, 1, np.int32)
        Input = np.cos(2*np.pi*n*0.1) + 1j * np.sin(2*np.pi*n*0.1)
        Input = Input.astype(np.complex64)

        Output = np.zeros(math.ceil( (5/6)*(NumberOfSamples-16)), np.complex64)
        OutputTime = []

        OutputIndex = 0
        for InputIndex in range(0, NumberOfSamples - 16):
            Remainder = InputIndex % 6

            match(Remainder):
                case 0:
                    Output[OutputIndex] = Input[InputIndex + 7]
                case 1:
                    Samples             = Input[InputIndex:InputIndex + 16]
                    Out                 = np.sum(CoefficientsP3 * Samples)
                    Output[OutputIndex] = Out
                case 2:
                    Samples             = Input[InputIndex:InputIndex + 16]
                    Out                 = np.sum(CoefficientsP1 * Samples)
                    Output[OutputIndex] = Out
                case 3:
                    Samples             = Input[InputIndex:InputIndex + 16]
                    Out                 = np.sum(CoefficientsM1 * Samples)
                    Output[OutputIndex] = Out
                case 4:
                    Samples             = Input[InputIndex:InputIndex + 16]
                    Out                 = np.sum(CoefficientsM3 * Samples)
                    Output[OutputIndex] = Out
                case 5:
                    continue

            OutputTime.append(7 + OutputIndex * 1.2) 
            OutputIndex += 1
        
        plt.figure(1)
        plt.subplot(2,1,1)
        plt.plot(n, Input.real, 'r', n, Input.imag, 'b')
        plt.plot(OutputTime, Output.real, 'ro', OutputTime, Output.imag, 'bo')
        plt.title('Simulation of Resampler in Python')
        plt.grid(True)
   

        # ------------------------------------------
        # Here we will use the extension module to run the resampler
        # ------------------------------------------
   #     Out1 = Resampler.wReserveSequenceSize(NumberOfSamples)
   #     Out2 = Resampler.BurstInputWaveform(Input)
   #     Out3 = Resampler.wResample()
   #     NumOutputSamples = Resampler.wGetNumOutputSamples()
   #     ResamplerOutput = np.zeros(NumOutputSamples, np.complex64)
   #     Out4 = Resampler.BurstOutputWaveform(ResamplerOutput)
      
   #     plt.subplot(2,1,2)
   #     plt.plot(n, Input.real, 'r', n, Input.imag, 'b')
   #     plt.plot(OutputTime, ResamplerOutput.real, 'ro', OutputTime, ResamplerOutput.imag, 'bo')
   #     plt.grid(True)
   #     plt.title('Simulation of Resampler in C++')
   #     plt.show()

   #     print(np.absolute(Output - ResamplerOutput))
        

    # ----------------------------------------------------------
    if Test == 2:      # Halfband filter impulse and frequency response test
    # ----------------------------------------------------------
        # In this test we control the design of the halfband filter and verify
        # its impulse and frequency response
        NumberOfTaps = 31
        HanningOverlay = True
        Halfband = CHalfbandFilter(NumberOfTaps, HanningOverlay)
        Halfband.PlotHalfbandResponses()
        print(Halfband.h)




    # ----------------------------------------------------------
    if Test == 3:      #  Halfband filter execution test
    # ----------------------------------------------------------
        # In this test, we will verify the RunHalfbandFilter() function of the
        # CHalfbandFilter class
        NumberOfTaps = 31
        HanningOverlay = True
        Halfband = CHalfbandFilter(NumberOfTaps, HanningOverlay)

        # Generate an appropriate input signal
        n              = np.arange(0, 1000, 1, np.int32)
        Signal1        = np.exp(1j*2*np.pi*n*0.01)
        Signal2        = np.exp(1j*2*np.pi*n*0.28)
        InputSequence  = Signal1 + Signal2 
        OutputSequence = Halfband.RunHalfbandFilter(InputSequence)

        # Non optimized execution
        OutputHighFs   = np.convolve(InputSequence, Halfband.h)

        plt.figure(1)
        plt.subplot(3,1,1)
        plt.plot(n, InputSequence.real, 'r',  n, InputSequence.imag, 'b')
        plt.title('Input to halfband filter')
        plt.grid(True)
        plt.tight_layout()
        plt.subplot(3,1,2)
        plt.plot(OutputHighFs.real, 'r', OutputHighFs.imag, 'b')
        plt.title('Ideal Output of halfband filter')
        plt.grid(True)
        plt.tight_layout()
        plt.subplot(3,1,3)
        plt.plot(OutputSequence.real, 'r', OutputSequence.imag, 'b')
        plt.title('Output of halfband filter')
        plt.grid(True)
        plt.tight_layout()
        plt.show()