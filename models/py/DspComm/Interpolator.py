# File:        Interpolator.py
# Notes:       This python module is a template for a similar implementation to be done in C++.
#              It is a general, high speed interpolator that will use a mixture of an all-pass 
#              16 tap filter process (linear combination) and linear interpolation. 
#              Steps: 
#              1. 481 Coefficient sets are computed at t0 -> 0, 1/32, 2/32, ... , 14 + 31/32, 15
#              2. Assume that the input sequence features N samples.
#              3. For a desired sample time 0 <= t <= N - 1, a set of 16 samples are addressed
#                 from within the input sequence, such that we (preferably) have 8 samples before
#                 and 8 samples after t.
#              4. We will interpolate two new values from the addressed 16 sample sequence. The 
#                 two values of t0 will be the closest to possible to t (one smaller, one larger)  
#                 Preferable 7 <= t0 < 8. 
#              5. Once we have the two interpolated values, we will use linear interpolation to 
#                 find the exact value at t.
#              Notes: Using 16 taps is a good compromise between computational load and 
#                     interpolation accuracy. At 16 taps a sinewave with a normalized frequency of
#                     0.4 can be accurately estimated. The performance decreases quickly after a
#                     a normalized frequency of 0.42.
#              Notes: The input sequence length must be at least 2, otherwise interpolation is meaningless.
#                     The interpolator will exterpolate by truncating the given value t, to 0 and N - 1.

__title__     = "Interpolator"
__author__    = "Andreas Schwarzinger"
__status__    = "preliminary"
__version__   = "0.2.0"
__date__      = "May 5th, 2023"
__copyright__ = 'Andreas Schwarzinger'

# ----------------------------------------- #
# Import statements                         #
# ----------------------------------------- #
import numpy             as     np
import math
import matplotlib.pyplot as     plt


# -------------------------------------------------------------------------------------------------- #
#                                                                                                    #
# > CInterpolator                                                                                    #
#                                                                                                    #
# -------------------------------------------------------------------------------------------------- #
class CInterpolator():
    '''
    This class provides interpolation services
    '''
    ResolutionPerSamplePeriod   = 32  # The number of interpolation points with one sample period
    MaxN                        = 16  # Number of taps in linear combiner
    assert MaxN == 16 or MaxN == 32

    
    def __init__(self):
        # Let's construct the CoefficientContainer
        self.NumTaps                 = CInterpolator.MaxN
        self.NumTapsOver2            = math.floor(self.NumTaps/2)
        self.NumTapsOver2Minus1      = self.NumTapsOver2 - 1

        NumberOfInterpolationPoints  = (self.NumTaps - 1) * CInterpolator.ResolutionPerSamplePeriod + 1
        self.NumberOf256BitVariables = NumberOfInterpolationPoints * 2
        self.CoefficientContainer    = np.zeros([self.NumberOf256BitVariables, self.NumTapsOver2], np.float32)

        for Index in range(0, NumberOfInterpolationPoints):
            dSampleTime = Index / CInterpolator.ResolutionPerSamplePeriod - (self.NumTaps - 1)/2
            C           = CInterpolator.ComputeAllPassCoefficients(self.NumTaps, dSampleTime)
            Entry0      = 2*Index
            Entry1      = Entry0 + 1
            self.CoefficientContainer[Entry0, :] = C[0                 : self.NumTapsOver2]
            self.CoefficientContainer[Entry1, :] = C[self.NumTapsOver2 : self.NumTaps]
        



       


    # -------------------------------------------------- #
    # > Instance method: InterpolateFloat32()            #
    # -------------------------------------------------- #
    def InterpolateFloat32(self
                         , InputVector:       np.ndarray
                         , TargetSampleTimes: np.ndarray 
                         , StartSample:       float = 0
                         , SampleStep:        float = 0
                         , NumOutputSamples:  int   = 0):
        '''
        brief: This function will interpolate a value from a floating point numpy array
        param: InputVector       - A np.ndarray of type np.float32 holding the source sequence values
        param: TargetSampleTimes - A np.ndarray of type np.float32 holding the sample times at which to intepolate
        param: StartSample       - If TargetSampleTimes.size() = 0, then this is the start sample time
        param: SampleStep        - If TargetSampleTimes.size() = 0, then this is the sample step
        param: NumOutputSamples  - If TargetSampleTimes.size() = 0, then this is the number of samples to interpolate
        notes: If TargetSampleTimes.size() == 0, then interpolate at StartSample + n * SampleStep 
               where n = 0, 1, 2, 3 ... NumOutputSamples - 1
        notes: Some of the code is written to easily debug the C++ implementation that was also written.
        '''
        # ----------------------------------- # 
        # > Error checking                    #
        # ----------------------------------- #
        assert isinstance(InputVector, np.ndarray)
        assert isinstance(TargetSampleTimes, np.ndarray)
        assert isinstance(StartSample, float) or isinstance(StartSample, int)
        assert isinstance(SampleStep, float)  or isinstance(SampleStep, int)
        assert isinstance(NumOutputSamples, int)
        assert NumOutputSamples != 0 or len(TargetSampleTimes) != 0

        # ------------------------------------ #
        # > Begin Interpolating Process        #
        # ------------------------------------ #
        NumberInputSamples  = len(InputVector)
        NumberOutputSamples = len(TargetSampleTimes)
        if len(TargetSampleTimes) == 0:
            NumberOutputSamples = NumOutputSamples

        Output      = np.zeros(NumberOutputSamples, np.float32)
        OutputTimes = np.zeros(NumberOutputSamples, np.float64)

        for Index in range(0, NumberOutputSamples):
            if len(TargetSampleTimes) != 0:
                dTargetSampleTime = TargetSampleTimes[Index]
            else:
                dTargetSampleTime = StartSample + Index * SampleStep

            OutputTimes[Index] = dTargetSampleTime

            # ------------------------------------- #
            # > Handling Extrapolation              #
            # ------------------------------------- # 
            if dTargetSampleTime < 0                     : dTargetSampleTime = 0
            if dTargetSampleTime > NumberInputSamples - 1: dTargetSampleTime = NumberInputSamples - 1

            # -------------------------------------------------- #
            # > Fetch the 16 samples from the input sequence     #
            # -------------------------------------------------- #
            dwFloorSampleTime = math.floor(dTargetSampleTime)
            dwStartSampleTime = 0
            if dwFloorSampleTime >= self.NumTapsOver2Minus1:
                dwStartSampleTime = dwFloorSampleTime - self.NumTapsOver2Minus1    
            if (dwFloorSampleTime >= NumberInputSamples - self.NumTapsOver2):   
                dwStartSampleTime = NumberInputSamples - self.NumTaps  

            First8SourceSamples  = InputVector[dwStartSampleTime + 0                 : dwStartSampleTime + self.NumTapsOver2]
            Second8SourceSamples = InputVector[dwStartSampleTime + self.NumTapsOver2 : dwStartSampleTime + self.NumTaps ]

            # ------------------------------------------------------------------------------ #
            # > Fetch the 2 sets of 16 coefficients for the two intermediate interpolations  #
            # ------------------------------------------------------------------------------ #
            dwBaseSampleAddress = dwFloorSampleTime - dwStartSampleTime
            dRemainder          = dTargetSampleTime - dwFloorSampleTime
            dwSubSampleAddress  = math.floor(dRemainder * CInterpolator.ResolutionPerSamplePeriod)
            dwIndex             = 2 * int(dwBaseSampleAddress * CInterpolator.ResolutionPerSamplePeriod + dwSubSampleAddress)

            Coeff1A             = self.CoefficientContainer[dwIndex + 0, :]
            Coeff1B             = self.CoefficientContainer[dwIndex + 1, :]
            if dwIndex + 2 >= self.NumberOf256BitVariables:   # This can happen
                Coeff2A             = self.CoefficientContainer[dwIndex + 0, :]
                Coeff2B             = self.CoefficientContainer[dwIndex + 1, :]                
            else:
                Coeff2A             = self.CoefficientContainer[dwIndex + 2, :]
                Coeff2B             = self.CoefficientContainer[dwIndex + 3, :]

            # ---------------------------------------------------------- #
            # > This multiplications                                     #
            # ---------------------------------------------------------- #
            Product1A           = Coeff1A * First8SourceSamples
            Product1B           = Coeff1B * Second8SourceSamples
            Result1             = np.sum(Product1A) + np.sum(Product1B)

            Product2A           = Coeff2A * First8SourceSamples
            Product2B           = Coeff2B * Second8SourceSamples
            Result2             = np.sum(Product2A) + np.sum(Product2B)
    
            # ---------------------------------------------------------- #
            # > The linear interpolation                                 #
            # ---------------------------------------------------------- #
            dStep          = 1.0 / CInterpolator.ResolutionPerSamplePeriod
            dStepRemainder = np.fmod(dRemainder, dStep)
            fWeightfSum2   = dStepRemainder * CInterpolator.ResolutionPerSamplePeriod
            fWeightfSum1   = 1.0 - fWeightfSum2

            Output[Index] = fWeightfSum1 * Result1 + fWeightfSum2 * Result2

        return (Output, OutputTimes)







    # --------------------------------------------------------- #
    # > Instance Method: ComputeLagrangeCoefficients()          #
    # --------------------------------------------------------- #
    # Notes: We ended up not using this function in the module. It is here only 
    #        for reference in case its content can be used in other modules.
    @staticmethod
    def ComputeLagrangeCoefficients(NumTaps: int 
                                  , SampleTime: float) -> np.ndarray:
        '''
        brief: Use Lagrange method to compute interpolator coefficients
        param: NumTaps    - The number of coefficients to compute
        param: SampleTime - The fractional sample time for which to compute the coefficients
        notes: Sample times are assumed to be 0, 1, 2, ... NumTaps - 1
        '''
        # -----------------------------------
        # 0. Error and type checking
        # -----------------------------------
        assert isinstance(NumTaps, int)
        assert isinstance(SampleTime, int) or isinstance(SampleTime, float)
        assert SampleTime >= 0 and SampleTime <= NumTaps - 1

        # ------------------------------------
        # 1. Compute the coefficients
        # ------------------------------------
        Coefficients = np.ones(NumTaps, np.float32)    
        for i in range(0, NumTaps):   # Compute each FIR tap value
            for m in range(0, NumTaps):
                if m == i: 
                    continue
                x     = SampleTime
                Temp  = (x - m) / (i - m)
                Coefficients[i] *= Temp 
                T = 0

        return Coefficients







    # --------------------------------------------------------- #
    # > Instance Method: ComputeAllPassCoefficients()           #
    # --------------------------------------------------------- #
    @staticmethod
    def ComputeAllPassCoefficients(NumTaps: int 
                                 , SampleTime: float) -> np.ndarray:
        '''
        brief: Use the all pass FIR method to compute interpolator coefficients
        param: NumTaps    - The number of coefficients to compute
        param: SampleTime - The fractional sample time for which to compute the coefficients
        notes: Sample times are assumed to be 0, 1, 2, ... NumTaps - 1
        '''
        # -----------------------------------
        # 0. Error and type checking
        # -----------------------------------
        assert isinstance(NumTaps, int)
        assert isinstance(SampleTime, int) or isinstance(SampleTime, float)
        Limit = (NumTaps - 1)/2 
        assert SampleTime >= -Limit  and SampleTime <= Limit

        # ----------------------------------
        # 1. Basic setup
        # ----------------------------------
        N         = NumTaps
        # The normalized frequencies (i.e.: N = 8 -> [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])
        if N%2 == 0:
            m         = np.arange(-math.floor(N/2), 0.99999 * math.floor(N/2), 1, np.float32)
            m        += 0.5
        else:
            m         = np.arange(-(N-1)/2, 1.0001 * (N - 1)/2, 1, np.float32)
        # The normalized time
        n         = m.copy() 
        Mag       = np.ones(N, np.complex64)

        # Compute the Hanning overlay
        k         = np.arange(0, N, 1, np.int32)
        Hanning   = np.ones(N, np.float32) - np.cos(2*np.pi*(k+1)/(N+1))

        # -----------------------------------
        # 2. Compute the coefficients
        # -----------------------------------
        Coefficients = np.ones(NumTaps, np.float32)
        Phase        = -2*np.pi*m*SampleTime/N
        H            = Mag * np.cos(Phase) + 1j * Mag * np.sin(Phase)

        # Compute inverse IDFT to get the impulse response of the all-pass filter 
        h     = np.zeros(N, np.complex64)
        for Index, Time in enumerate(n):
            h[Index] = np.sum(H * np.exp(1j*2*np.pi*m*Time/N))

        Coefficients = (h * Hanning)/ np.sum(h*Hanning)

        return Coefficients.real









# ------------------------------------------------
# > Test bench
# ------------------------------------------------
if __name__ == '__main__':

    Test = 3    # 0 - Just construct the CInterpolator object to see the constructor build the coefficient tables
                # 1 - Testing ComputeLagrangeCoefficients() and ComputeAllPassCoefficients()
                # 2 - Testing InterpolateFloat32() using TargetSampleTimes
                # 3 - Testing InterpolateFloat32() using StartSample, SampleStep, and NumOutputSamples (len(TargetSampleTimes) = 0)
                  
    assert Test < 4

    # -------------------------------------------------------------------------
    # Test 0 - Just construct the CInterpolator object to see the constructor build the coefficient tables
    # -------------------------------------------------------------------------
    if Test == 0:
        Interpolator  = CInterpolator()
        print(Interpolator.CoefficientContainer[100])
        print(Interpolator.CoefficientContainer[300])
        print(Interpolator.CoefficientContainer[500])
        print(Interpolator.CoefficientContainer[700])
        print(Interpolator.CoefficientContainer[900])


    # -------------------------------------------------------------------------
    # Test 1 primarily tests the proper functing of ComputeLagrangeCoefficients() and ComputeAllPassCoefficients()
    # -------------------------------------------------------------------------
    if Test == 1:
        # -> Testing the ComputeLagrangeCoefficients() and ComputeAllPassCoefficients()
        # To evaluate the performance change the following:
        # > NormFrequency (> -0.5 and < 0.5)
        # > NumSamples (from 2 to 16)
        # The larger the normalized frequency gets to the limits the harder the sinusoid is to interpolate
        # The larger the number of samples (thus the number of taps used), the better the interpolation.
        # -> The Lagrange method is somewhat inferior in certain situation than the All Pass method.
        
        NumSamples    = 16
        NormFrequency = 0.40

        # Create the source waveform    
        Time          = np.arange(0, NumSamples, 1, np.int32)
        SourceValues  = np.sin(2*np.pi*NormFrequency * Time) 
        SourceArray   = SourceValues[0:NumSamples]
        SourceTimes   = np.arange(0, len(SourceArray), 1, np.int32)
        TargetTimes   = np.arange(0, NumSamples - .999, 0.1, np.float64)
        Output1       = np.zeros(len(TargetTimes), np.float64)
        Output2       = np.zeros(len(TargetTimes), np.float64)
        IdealOutput   = np.sin(2*np.pi* NormFrequency * TargetTimes)
        
        # Determine the interpolator coefficients for both methods and compute the linear combination
        # that yields the interpolated values. The figure shows the performance differences.
        for Index in range(0, len(TargetTimes)):
            Time1          = TargetTimes[Index]
            Time2          = Time1 - (NumSamples - 1)/2
            B1             = CInterpolator.ComputeLagrangeCoefficients(NumSamples, Time1)
            B2             = CInterpolator.ComputeAllPassCoefficients(NumSamples, Time2)
            Output1[Index] = np.sum(SourceArray * B1)
            Output2[Index] = np.sum(SourceArray * B2)

        plt.figure(1)
        plt.subplot(2,1,1)
        plt.plot(SourceTimes, SourceArray, 'ko')
        plt.plot(TargetTimes, IdealOutput, 'k:')
        #plt.plot(TargetTimes, Output1, 'r')
        plt.plot(TargetTimes, Output2, 'b')
        plt.title('Lagrange Interpolated Values')
        plt.xlabel('Discrete / Fractional Time')
        #plt.legend(['Source Values', 'Perfect', 'Lagrange', 'AllPass'])
        plt.legend(['Source Values', 'Perfect', 'AllPass'])
        plt.grid(True)
        plt.tight_layout()
        plt.subplot(2,1,2)
        plt.plot(TargetTimes, IdealOutput - Output2, 'k')
        plt.title('Error between the Ideal and AllPass Values')
        plt.xlabel('Discrete / Fractional Time')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    


    # -----------------------------------------------------------------------------------
    # Testing InterpolateFloat32() using TargetSampleTimes
    # -----------------------------------------------------------------------------------
    if Test == 2:
        # To evaluate the performance of the interpolator:
        # > NormFrequency (> -0.5 and < 0.5)
        # > NumSamples (from 2 to 16)
        # The larger the normalized frequency gets to the limits the harder the sinusoid is to interpolate
        # The larger the number of samples (thus the number of taps used), the better the interpolation.
        
        # ----------------------------------------
        # Test initialization
        # ----------------------------------------
        NumSamples    = 16  
        NormFrequency = 0.4
        assert NumSamples <= 16, "The NumSamples must be less than MaxN"
        assert NormFrequency > -0.5 and NormFrequency < 0.5, "The NormFrequency is invalid"

        # ----------------------------------------
        # Run test
        # ----------------------------------------
        Interpolator  = CInterpolator()
        SourceTimes   = np.arange(0, 15.1, 1, np.int32)
        SourceValues  = np.sin(2*np.pi*NormFrequency * SourceTimes)

        # The desired time 
        T             = (7 + 1/32 + 1/128)
        TargetTimes   = T * np.ones(1, np.float32) #np.arange(0, 15, 0.1, np.float32)
        PerfectTargetValues = np.sin(2*np.pi*NormFrequency * TargetTimes)

        # Let's do the interpolation without the secondary linear intepolation
        PerfectInt = np.zeros(len(TargetTimes), np.float32)
       # for Index, Time2 in enumerate(range(0, 150, 1)):
        Time2 = T
        B2            = Interpolator.ComputeAllPassCoefficients(NumSamples, Time2 - 7.5)
        PerfectInt[0] = np.sum(SourceValues * B2)

        (Output, OutputTimes)  = Interpolator.InterpolateFloat32(SourceValues
                                                               , np.zeros(0, np.float32)
                                                               , StartSample = 0
                                                               , SampleStep  = 0.1
                                                               , NumOutputSamples = int(31/0.1))

        plt.figure(1)
        plt.subplot(2,1,1)
        plt.plot(SourceTimes,  SourceValues, 'ko')
        plt.plot(OutputTimes,  Output, 'b')
        plt.title('Interpolation Test')
        plt.xlabel('Discrete Time')
        plt.grid(True)
        plt.tight_layout()
        plt.subplot(2,1,2)
        plt.plot(OutputTimes, Output - PerfectTargetValues, 'k')
        #plt.plot(OutputTimes, PerfectInt - PerfectTargetValues, 'b')
        plt.title('Interpolation Error')
        plt.xlabel('Discrete Time')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        Start = 0.5032914
       # Ideal = 0.48190898 (linear int = 0.4817789 matches my linear interpolation 0.25 from first point)
      #  Ideal = 0.4604307  (linear int = 0.46026634 matches my linear interpolation 0.5 between points)
        
        End = 0.41724128

        # Hurra, it seem to be working fine now.


    # -----------------------------------------------------------------------------------
    # Testing InterpolateFloat32() using StartSample, SampleStep, and NumOutputSamples (len(TargetSampleTimes) = 0)
    # -----------------------------------------------------------------------------------
    if Test == 3:
        # To evaluate the performance of the interpolator:
        # > NormFrequency (> -0.5 and < 0.5)
        # > NumSamples (from 2 to CInterpolator.MaxN  )
        # The larger the normalized frequency gets to the limits the harder the sinusoid is to interpolate
        # The larger the number of samples (thus the number of taps used), the better the interpolation.
        
        # ----------------------------------------
        # Test initialization
        # ----------------------------------------
        NumSamples    = 40
        NormFrequency = 0.4
        assert NormFrequency > -0.5 and NormFrequency < 0.5, "The NormFrequency is invalid"

        # ----------------------------------------
        # Run test
        # ----------------------------------------
        Interpolator        = CInterpolator()
        SourceTimes         = np.arange(0, NumSamples - 1 + 0.1, 1, np.int32)
        SourceValues        = np.cos(2*np.pi*NormFrequency * SourceTimes)

        # The desired time 
        TStartSample        = 0
        TEndSample          = NumSamples - 1
        TSampleStep         = 0.1
        TNumSamples         = int(TEndSample/0.1)    # Potentially forcing extrapolation
        TargetTimes         = np.arange(TStartSample, TEndSample, TSampleStep, np.float32) 
        
        # The perfect interpolated values
        PerfectTargetValues = np.sin(2*np.pi*NormFrequency * TargetTimes)

        (Output, OutputTimes)  = Interpolator.InterpolateFloat32(SourceValues
                                                               , np.zeros(0, np.float32)
                                                               , StartSample = 0
                                                               , SampleStep  = 0.1
                                                               , NumOutputSamples = len(TargetTimes))
        Error = []
        for Index in range(0, len(TargetTimes)):
            Error.append(PerfectTargetValues[Index] - Output[Index])
            assert abs(TargetTimes[Index] - OutputTimes[Index]) < 0.001

        plt.figure(1)
        plt.subplot(2,1,1)
        plt.plot(SourceTimes,  SourceValues, 'ko')
        plt.plot(TargetTimes,  PerfectTargetValues, 'k:')
        plt.plot(OutputTimes,  Output, 'b')
        plt.title('Interpolation Test')
        plt.xlabel('Discrete Time')
        plt.legend(['Source Values', 'Perfect Interpolation', 'Out Interpolation'] )
        plt.grid(True)
        plt.tight_layout()
        plt.subplot(2,1,2)
        plt.plot(OutputTimes, Error, 'k')
        plt.title('Interpolation Error')
        plt.xlabel('Discrete Time')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        Start = 0.5032914
       # Ideal = 0.48190898 (linear int = 0.4817789 matches my linear interpolation 0.25 from first point)
      #  Ideal = 0.4604307  (linear int = 0.46026634 matches my linear interpolation 0.5 between points)
        
        End = 0.41724128

        # Hurra, it seem to be working fine now.