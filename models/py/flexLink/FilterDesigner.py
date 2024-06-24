# File:       FilterDesigner.py
# Notes:      This script supports FIR filter design tasks

__title__     = "FilterDesigner"
__author__    = "Andreas Schwarzinger"
__status__    = "preliminary"
__date__      = "Oct, 9th, 2022"
__copyright__ = 'Andreas Schwarzinger'

# Import statements
import numpy             as     np
import math
import matplotlib.pyplot as     plt

# ---------------------------------------------------
# > CFilterDesigner class
# ---------------------------------------------------
class CFilterDesigner():
    """
    This class will manage FIR filter design and analysis tasks
    """




    # -------------------------------------
    # > ShowFrequencyResponse()
    # -------------------------------------
    @staticmethod
    def ShowFrequencyResponse(ImpulseResponse
                            , SampleRate:        float = 1.0
                            , OversamplingRatio: int   = 4
                            , bShowInDb:         bool  = True
                            , bShowIqVersion:    bool  = False):   # By default we show the Mag/Phase response,
                                                                   # not the I/Q version of the response
        # ------------------------
        # Error checking
        bRightSampleRateType = isinstance(SampleRate, int) or isinstance(SampleRate, float)
        assert bRightSampleRateType,               'The SampleRate input argument has invalid type.'
        assert isinstance(bShowInDb, bool),        'The bShowInDb input argument must be a boolean type.'
        assert isinstance(bShowIqVersion, bool),   'The bShowIqVersion input argument must be a boolean type.'
        bProperType = isinstance(ImpulseResponse, list) or isinstance(ImpulseResponse, np.ndarray)
        assert bProperType,                        'The ImpulseResponse is of invalid type.'
        assert isinstance(OversamplingRatio, int), 'The OversamplingRatio must be an integer.'
        assert OversamplingRatio <= 64,            'The OversamplingRatio is unreasonably large. Should be <= 64.'

        # -------------------------
        # Convert into an np.array if necessary
        if isinstance(ImpulseResponse, list):
            ImpulseResponseNew = np.array(ImpulseResponse, dtype = np.complex64)
        else:
            ImpulseResponseNew = ImpulseResponse.astype(dtype = np.complex64)

        # Ensure that the impulse response is a simple array and not a matrix with a single row
        if len(ImpulseResponseNew.shape) == 2:
            ImpulseResponseNew = ImpulseResponse.flatten()

        # -------------------------
        # Find the frequency step associated with the oversampling ratio
        N                 = len(ImpulseResponse)
        n                 = np.arange(0, N)
        FreqStep          = 1 / (N * OversamplingRatio)

        # ------------------------
        # Start running the DTFT for each frequency to check
        FrequenciesToCheck  = np.arange(-0.5, 0.5, FreqStep)
        FrequencyResponse   = np.zeros(len(FrequenciesToCheck), dtype = np.complex64)
        MagnitudeResponse   = np.zeros(len(FrequenciesToCheck), dtype = np.float32)
        PhaseResponse       = np.zeros(len(FrequenciesToCheck), dtype = np.float32)
        for Index, F in enumerate(FrequenciesToCheck):
            AnalysisWaveform          = np.exp(-1j*2*np.pi*n*F, dtype = np.complex64) 
            FrequencyResponse[Index]  = (1/N) * np.sum(ImpulseResponseNew * AnalysisWaveform)
            MagnitudeResponse[Index]  = np.abs(FrequencyResponse[Index])
            PhaseResponse[Index]      = np.angle(FrequencyResponse[Index])
            
            if bShowInDb == True:
                MagnitudeResponse[Index] = 20*np.log10(MagnitudeResponse[Index])
            
        # ------------------------
        # Plot the frequency response
        plt.figure(20)
        if bShowIqVersion == False:
            plt.subplot(3,1,1)
            plt.plot(np.arange(0, len(ImpulseResponse)), ImpulseResponse.real, 'r-o')
            plt.plot(np.arange(0, len(ImpulseResponse)), ImpulseResponse.imag, 'b-o')
            plt.title('Impulse Response')
            plt.xlabel('n')
            plt.grid(True)
            plt.tight_layout()
            
            plt.subplot(3,1,2)
            plt.plot(FrequenciesToCheck * SampleRate, MagnitudeResponse)
            plt.title('Magnitude Response')
            plt.xlabel('Normalized Frequency in Hz')
            if(bShowInDb):
                plt.ylabel('dB')
            plt.grid(True)
            plt.tight_layout()
        
            plt.subplot(3,1,3)
            plt.plot(FrequenciesToCheck * SampleRate, PhaseResponse)
            plt.title('Phase Response Response')
            plt.xlabel('Normalized Frequency in Hz')
            plt.ylabel('radians')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            plt.subplot(3,1,1)
            plt.plot(np.arange(0, len(ImpulseResponse)), ImpulseResponse.real, 'r')
            plt.plot(np.arange(0, len(ImpulseResponse)), ImpulseResponse.imag, 'b')
            plt.title('Impulse Response')
            plt.grid(True)
            plt.tight_layout()

            plt.subplot(3,1,2)
            plt.plot(FrequenciesToCheck * SampleRate, FrequencyResponse.real)
            plt.title('Real Portion of Frequency Response')
            plt.xlabel('Normalized Frequency in Hz')
            plt.grid(True)
            plt.tight_layout()
        
            plt.subplot(3,1,3)
            plt.plot(FrequenciesToCheck * SampleRate, FrequencyResponse.imag)
            plt.title('Imaginary Portion of Frequency Response')
            plt.xlabel('Normalized Frequency in Hz')
            plt.ylabel('radians')
            plt.grid(True)
            plt.tight_layout()
            plt.show()








    # -------------------------------------
    # > The Frequency Sampling Method
    # -------------------------------------
    @staticmethod
    def FrequencySampling(LinearGainList:        list
                        , NormalizedFreqList:    list
                        , N:                     int
                        , bShowPlots:            bool = False) -> list:

        '''
        brief: This function computes the complex FIR filter response for the attenuation and frequency values provided.
        param: LinearGainList      Input ->   A list of linear gains at desired a frequency positions described in NormalizedFreqList
        param: NormalizedFreqList  Input ->   The list of normalized frequencies (-0.5 to 0.5) where the LinearGainList entries are defined. 
        param: N                   Input ->   The number of filter tabs
        notes: The Entries in the LinearGainList and the NormalizedFreqList are pairs that belong togethe.
        '''

        # ----------------------
        # Error checking
        assert isinstance(LinearGainList, list),               'The LinearGainList must be a list.'
        assert isinstance(NormalizedFreqList, list),           'NormalizedFreqList must be a list.'
        assert isinstance(N, int),                             'NumberTabs must be an int.'
        assert len(LinearGainList) == len(NormalizedFreqList), 'The frequency and attenuation lists must be of the same size.'

        bSensibleNumberFilterTabs = N > 1 and isinstance(N, int) and N <= 10000
        assert bSensibleNumberFilterTabs,                           'The NumberOfFilterTabs argument is invalid.'

        bProperFrequencies = all([((isinstance(x, int) or isinstance(x, float) or np.issubdtype(x, np.floating)) and x <= 0.51 and x >= -0.51 ) for x in NormalizedFreqList])
        assert bProperFrequencies,                             'Error in NormalizedFreqlist.'
        
        # -----------------------
        # Sort NormalizeFreqList from most negative to most positive frequencies
        SortedFrequencies     = np.sort(NormalizedFreqList)
        SortedIndices         = np.argsort(NormalizedFreqList)
        SortedGainList        = [LinearGainList[x] for x in SortedIndices]

        # -----------------------
        # Find linearly interpolated Gain values
        NormalizedFrequencies = []
        for F in range(0, N):
            F_Norm = F/N
            if F_Norm >= 0.5:
                F_Norm -= 1
            NormalizedFrequencies.append(F_Norm)

        NormalizedFrequencies = sorted(NormalizedFrequencies)   # Sort from the lowest to the highest frequency
        GainList              = np.interp(NormalizedFrequencies, SortedFrequencies, SortedGainList)

        # -----------------------
        # Program the IFFT buffer
        GainListNegative       = GainList[0 : math.floor(N/2)]
        GainListPositive       = GainList[math.floor(N/2): N]
        IFFT_Buffer            = np.hstack([GainListPositive, GainListNegative])

        ImpulseResponse        = N * np.fft.ifft(IFFT_Buffer)   # The ifft() function internally divides by N 
        ImpulseResponseOrdered = np.hstack([ImpulseResponse[math.ceil(N/2):N], ImpulseResponse[0 : math.ceil(N/2)]]) 

        # -----------------------
        # Overlay the impulse response with a Hanning window
        n                    = np.arange(0, N, 1, dtype = np.int32)
        Hanning              = (0.5 - 0.5 * np.cos(2*np.pi * (n + 1) / (N + 1)))**1
        ImpulseResponseFinal = ImpulseResponseOrdered * Hanning * N / np.sum(Hanning)

        if bShowPlots == True:
            plt.figure(1)
            plt.subplot(3, 1, 1)
            plt.stem(NormalizedFrequencies, GainList)
            plt.title('The desired frequency response')
            plt.xlabel('Normalid Frequency in Hz')
            plt.tight_layout()
            plt.grid(True)

            plt.subplot(3, 1, 2)
            plt.plot(range(0,N), ImpulseResponseFinal.real, 'r', range(0,N), ImpulseResponseFinal.imag, 'b')
            plt.title('The final impulse response')
            plt.xlabel('Tab Index')
            plt.tight_layout()
            plt.grid(True)

            FrequencyResponsedB = []
            FrequenciesToCheck  = np.arange(-0.5, 0.5, 0.01)
            for F in FrequenciesToCheck:
                AnalysisWaveform = np.exp(-1j*2*np.pi*n*F, dtype = np.complex64) 
                FreqResponse     = (1/N) * np.sum(ImpulseResponseFinal * AnalysisWaveform)
                FrequencyResponsedB.append(20*np.log10(np.abs(FreqResponse)))

            plt.subplot(3, 1, 3)
            plt.plot(FrequenciesToCheck, FrequencyResponsedB)
            plt.title('The actual frequency response in dB')
            plt.xlabel('Normalid Frequency in Hz')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return ImpulseResponseFinal











    # -------------------------------------
    # > The Least squares method of FIR design
    # -------------------------------------
    @staticmethod
    def LeastSquares(FreqResponseList:      list   # or np.ndarray - The desired complex frequency response       
                   , NormalizedFreqList:    list   # or np.ndarray - The associated normalized frequencies
                   , N:                     int    #               - The number of tabs (degrees of freedom)
                   , WeightingList:         list = [1]) -> list:
        '''
        This function implements the least squares method of FIR filter design
        '''

        NumConstraints        = len(FreqResponseList)
        # NumDegreesOfFreedom = N
        
        # --------------------------------------
        # Type checking
        # --------------------------------------
        # Everything must be converted to an np.ndarray type
        if isinstance(FreqResponseList, list):
            FreqResponseArray = np.array(FreqResponseList, dtype = np.complex64) # Change the type to np.complex64
        else:
            FreqResponseArray = FreqResponseList.astype(np.complex64)   # Recast to complex64 from whatever it was

        if isinstance(NormalizedFreqList, list):
            NormalizedFreqArray = np.array(NormalizedFreqList, dtype = np.float32) # Change the type to np.complex64
        else:
            NormalizedFreqArray = NormalizedFreqList.astype(np.float32)   # Recast to complex64 from whatever it was

        if isinstance(WeightingList, list):
            WeightingArray= np.array(WeightingList, dtype = np.float32)  
        else:
            WeightingArray = WeightingList.astype(np.float32)    

        # Ensure that the lengths of the input arguments are the same. If Weighting array is two small, then
        # overwrite it to be unity.
        assert NumConstraints == len(NormalizedFreqArray), 'The lengths of the two vectors must be equal'
        if len(WeightingArray) < NumConstraints:
            WeightingArray= np.ones(NumConstraints, dtype = np.complex64)

        assert isinstance(N, int),          'N must be an integer'
        assert N <= NumConstraints, \
            'The number of filter taps (degrees of freedom) must be <= the number of constraints.'

        # Ensure that the normalized frequencies make sense
        for NormalizedFreq in NormalizedFreqArray:
            assert NormalizedFreq >= -0.5 and NormalizedFreq <= 0.5, 'The normalized frequency must be between +/- 0.5'

        

        # ---------------------------------------------
        # Define the time, t, at which to compute the taps (e.g. N = 4 -> t = [-1.5, -0.5, 0.5, 1.5])
        #                                                  (e.g. N = 5 -> t = [-2.0, -1.0, 0, 1.0, 2.0])
        # ---------------------------------------------
        if N%2 == 1:     # N is odd
            Start_t = -(N-1)/2
            Stop_t  = -Start_t
        else:            # N is even
            Start_t = -(N/2 - 0.5)
            Stop_t  = -Start_t

        t = np.arange(Start_t, Stop_t + 0.001, 1, np.float32)

        # -------------------------------------------------
        # Compute the least squares solution 
        # -------------------------------------------------
        H = FreqResponseArray.reshape(NumConstraints, 1)
        F = np.zeros([NumConstraints, N], H.dtype)
        for IndexA in range(0, NumConstraints):
            for IndexB in range(0, N):
                F[IndexA, IndexB] = np.exp(1j*2*np.pi*t[IndexB]*NormalizedFreqList[IndexA])

        # h = inv(F' * W * F) * F'* W * H   (Easy in MatLab, but not in python)
        W = np.diag(WeightingArray)
        Temp1 = (F.conj().transpose().dot(W)).dot(F)
        Temp2 = (F.conj().transpose().dot(W)).dot(H)
        h = np.linalg.inv( Temp1 ).dot(Temp2)

        return h






# ------------------------------------------------------
# Test bench
# ------------------------------------------------------
if __name__ == '__main__':
    FilterDesigner = CFilterDesigner()

    Test = 3    # 1/2/3 - FrequencySampling / LeastSquares / Equiripple

    if Test == 1:
        # Test the frequency sampling method
        GainListLinear = [.0001, .0001,     1,    1,  .0001, .0001 ]
        NormalizedFreq = [-0.4,   -0.2, -0.19, 0.19,    0.2,    0.5]
        N              = 51
        ImpulseResponse = FilterDesigner.FrequencySampling(GainListLinear, NormalizedFreq, N,  False)

        FilterDesigner.ShowFrequencyResponse(ImpulseResponse   = ImpulseResponse
                                           , SampleRate        = 20.0e6
                                           , OversamplingRatio = 32
                                           , bShowInDb         = False)

    if Test == 2:
        # Test the least squares method
        FreqResponseList   = [ .1, .1, .1, .1, .5, 1, 1, 1, 1, 1, 1, 1, .5, .1, .1, .1, .1]
        NormalizedFreqList = (1/17)*np.arange(-8, 9, 1, dtype = np.int16) 
        W                  = np.ones(len(FreqResponseList), dtype = np.float32)
        N                  = 11
        ImpulseResponse = FilterDesigner.LeastSquares(FreqResponseList       
                                                    , NormalizedFreqList
                                                    , N
                                                    , W)
        
        FilterDesigner.ShowFrequencyResponse(ImpulseResponse   = ImpulseResponse
                                           , SampleRate        = 40.0e6
                                           , OversamplingRatio = 32
                                           , bShowInDb         = False
                                           , bShowIqVersion    = False)

    if Test == 3:
        # I use this test to illustrate frequency analysis in my book.
        N           = 8
        RatioMOverN = 8
        M           = N * RatioMOverN
        n  = np.arange(0, N, 1, np.int32)
        nn = np.arange(0, M, 1, np.int32)
        k  = nn - math.floor(M/2)
        m = 1
        OversamplingRatio = 8
        SampleRate        = 1

        # -------------------------------------------
        ImpulseResponseN = np.exp(1j*2*np.pi*n*m/N)
        ImpulseResponseM = np.exp(1j*2*np.pi*nn*m/N)
        ImpulseResponseY = np.exp(1j*2*np.pi*nn*(m+1)/N) 

        # -------------------------
        # Find the frequency step associated with the oversampling ratio
        FreqStepN            = 1 / (N * OversamplingRatio)
        FreqStepM            = 1 / (M * OversamplingRatio)
        Hanning              = 2 * (0.5 - 0.5 * np.cos(2*np.pi * (n + 1) / (N + 1))) 
        ImpulseResponseHann1 = ImpulseResponseN * Hanning * N / np.sum(Hanning)

        Hanning2             = 2 * (0.5 - 0.5 * np.cos(2*np.pi * (nn + 1) / (M + 1))) 
        Tones                = 0.5 * Hanning2 * (\
                               0.91416*np.exp(-1j*2*np.pi*k*(4)/M) + \
                               1.0*np.exp(-1j*2*np.pi*k*(3)/M) + \
                               1.0*np.exp(-1j*2*np.pi*k*(2)/M) + \
                               1.0*np.exp(-1j*2*np.pi*k*(1)/M) + \
                               1.0*np.exp(1j*2*np.pi*k*(0)/M) + \
                               1.0*np.exp(1j*2*np.pi*k*( 1)/M) + \
                               1.0*np.exp(1j*2*np.pi*k*( 2)/M) + \
                               1.0*np.exp(1j*2*np.pi*k*( 3)/M) + \
                               0.91416*np.exp(1j*2*np.pi*k*( 4)/M))

        ImpulseResponseHann5 = np.exp(1j*2*np.pi*nn*1/N) * Hanning2 
        ImpulseResponseHann6 = np.exp(1j*2*np.pi*nn*1/N) * Tones 

        GainListLinear = [0, 0.0, .52]
        for i in range(0, RatioMOverN -1):
            GainListLinear.append(1)
        GainListLinear.append(0.52)
        GainListLinear.append(0)
        GainListLinear.append(0)
        
        
        NormalizedFreq = [-0.5, -int(RatioMOverN/2 + 1)/M,  -int(RatioMOverN/2)/M]
        for i in range(-int(RatioMOverN/2 - 1), int(RatioMOverN/2), 1):
            NormalizedFreq.append(i/M)
        NormalizedFreq.append(int(RatioMOverN/2)/M)
        NormalizedFreq.append(int(RatioMOverN/2 + 1)/M)
        NormalizedFreq.append(0.5)

        ImpulseResponseHannT = FilterDesigner.FrequencySampling(GainListLinear, NormalizedFreq, M,  False)
        ImpulseResponseHann2 = 0.5 * ImpulseResponseHannT * ImpulseResponseM  
        ImpulseResponseHann3 = 0.5 * ImpulseResponseHannT * ImpulseResponseY 

        # ------------------------
        # Start running the DTFT for each frequency to check
        FrequenciesToCheckN = np.arange(-0.5, 0.5, FreqStepN)
        FrequenciesToCheckM = np.arange(-0.5, 0.5, FreqStepM)
        FrequencyResponse0  = np.zeros(len(FrequenciesToCheckN), dtype = np.complex64)
        FrequencyResponse1  = np.zeros(len(FrequenciesToCheckN), dtype = np.complex64)
        MagnitudeResponse0  = np.zeros(len(FrequenciesToCheckN), dtype = np.float32)
        MagnitudeResponse1  = np.zeros(len(FrequenciesToCheckN), dtype = np.float32)
        MagnitudeResponse2  = np.zeros(len(FrequenciesToCheckM), dtype = np.float32)
        MagnitudeResponse3  = np.zeros(len(FrequenciesToCheckM), dtype = np.float32)
        MagnitudeResponse4  = np.zeros(len(FrequenciesToCheckM), dtype = np.float32)
        MagnitudeResponse5  = np.zeros(len(FrequenciesToCheckM), dtype = np.float32)
        MagnitudeResponse6  = np.zeros(len(FrequenciesToCheckM), dtype = np.float32)
        PhaseResponse       = np.zeros(len(FrequenciesToCheckN), dtype = np.float32)
        for Index, F in enumerate(FrequenciesToCheckN):
            AnalysisWaveform          = np.exp(-1j*2*np.pi*n*F, dtype = np.complex64) 
            FrequencyResponse0[Index] = (1/N) * np.sum(ImpulseResponseN * AnalysisWaveform)
            MagnitudeResponse0[Index] = np.abs(FrequencyResponse0[Index])
            FrequencyResponse1[Index] = (1/N) * np.sum(ImpulseResponseHann1 * AnalysisWaveform)
            MagnitudeResponse1[Index] = np.abs(FrequencyResponse1[Index])

        for Index, F in enumerate(FrequenciesToCheckM):
            AnalysisWaveform          = np.exp(-1j*2*np.pi*nn*F, dtype = np.complex64) 
            MagnitudeResponse2[Index] = np.abs((1/M) * np.sum(ImpulseResponseM * AnalysisWaveform))
            MagnitudeResponse3[Index] = np.abs((1/M) * np.sum(ImpulseResponseHann2 * AnalysisWaveform))
            MagnitudeResponse4[Index] = np.abs((1/M) * np.sum(ImpulseResponseHann3 * AnalysisWaveform))
            MagnitudeResponse5[Index] = np.abs((1/M) * np.sum(ImpulseResponseHann5 * AnalysisWaveform))
            MagnitudeResponse6[Index] = np.abs((1/M) * np.sum(ImpulseResponseHann6 * AnalysisWaveform))

        
        plt.figure(1)
        plt.subplot(3,1,1)
        plt.plot(FrequenciesToCheckN * SampleRate, MagnitudeResponse0, 'k')
        plt.title('Magnitude Response for Rectangular Window')
        plt.xlabel('Normalized Frequency in Hz')
        plt.xticks(np.arange(-0.5, 0.501, 1/N))
        plt.tight_layout()
        plt.grid(True)
        plt.subplot(3,1,2)
        plt.plot(FrequenciesToCheckN * SampleRate, MagnitudeResponse1, 'k')
        plt.title('Magnitude Response for Hanning Window')
        plt.xlabel('Normalized Frequency in Hz')
        plt.xticks(np.arange(-0.5, 0.501, 1/N))
        plt.tight_layout()
        plt.grid(True)        
        plt.subplot(3,1,3)
        #plt.plot(FrequenciesToCheckM * SampleRate, MagnitudeResponse2)
        plt.plot(FrequenciesToCheckM * SampleRate, MagnitudeResponse3, 'k')
        plt.plot(FrequenciesToCheckM * SampleRate, MagnitudeResponse4, 'k:')
        plt.title('Magnitude Response for Proper Window')
        plt.xlabel('Normalized Frequency in Hz')
        plt.xticks(np.arange(-0.5, 0.501, 1/N))
        plt.tight_layout()
        plt.grid(True)   


        plt.figure(2)
        plt.plot(FrequenciesToCheckN * SampleRate, MagnitudeResponse0, 'k')
        plt.plot(FrequenciesToCheckN * SampleRate, MagnitudeResponse1, 'k:')
        plt.plot([-.5, 0.061, 0.06125, 0.1875, 0.190, 0.5 ], [0, 0, 1, 1, 0, 0], 'g')
        plt.title('Magnitude Response for the Rectangular, Hanning and Desired Windows')
        plt.xlabel('Normalized Frequency in Hz')
        plt.xticks(np.arange(-0.5, 0.501, 1/N))
        plt.tight_layout()
        plt.legend(['Rectangular Window', 'Hanning Window', 'Desired Window'])
        plt.grid(True)

        plt.figure(3)
        plt.plot(FrequenciesToCheckM * SampleRate, MagnitudeResponse5, 'k')
        plt.plot(FrequenciesToCheckM * SampleRate, MagnitudeResponse6, 'k:')
        plt.title('Magnitude Response for Hanning and Desired Window')
        plt.xlabel('Normalized Frequency in Hz')
        plt.xticks(np.arange(-0.5, 0.501, 1/N))
        plt.tight_layout()
        plt.legend(['Hanning Window', 'Desired Window'])
        plt.grid(True)


        plt.figure(4)
        plt.subplot(212)
        plt.stem(nn, Tones * Hanning2/2, 'k')
        plt.tight_layout()
        plt.xlabel('Discrete Time n')
        plt.grid(True)
        plt.title('DesiredWindow[n]')
        plt.subplot(221)
        plt.plot(k/M, Tones, 'k')
        plt.title('Tones(t)')
        plt.xlabel('Normalized Time t in Seconds')
        plt.tight_layout()
        plt.grid(True)
        plt.subplot(222)
        plt.plot(k/M, Hanning2/2, 'k')
        plt.title('Hanning(t)')
        plt.tight_layout()
        plt.xlabel('Normalized Time t in Seconds')
        plt.grid(True)
        plt.show()