# File:       PreambleB.py
# Notes:      This file is used to generate the preambleB, which is used for timing
#             synchronization in of the waveform at the receiver.
# Notes:      Matches MatLab code in FlexLink document

__title__     = "PreambleB"
__author__    = "Andreas Schwarzinger"
__status__    = "released"
__date__      = "Sept, 16rd, 2022"
__copyright__ = 'Andreas Schwarzinger'


import numpy as np
import math

# -----------------------------------------------------
# > GeneratePreambleB()
# -----------------------------------------------------
def GeneratePreambleB() -> np.ndarray:

    # PreambleB Generation
    Nzc        = 331 
    ScPositive = math.ceil(Nzc/2)
    ScNegative = math.floor(Nzc/2)
    u1         = 34
    n          = np.arange(0, Nzc, 1, np.int16)
 
    # Definition of the Zadoff-Chu Sequence
    zc         = np.exp(-1j*np.pi*u1*n*(n+1)/Nzc)

    # The Fourier Transform of the Zadoff-Chu Sequence
    PreambleB_FFT = (1/np.sqrt(Nzc))*np.fft.fft(zc); 

    # ------------------------------------
    # Mapping into N = 1024 IFFT buffer and render into time domain
    # ------------------------------------
    IFFT_Buffer1024                = np.zeros(1024, np.complex64);  
    IFFT_Buffer1024[0:ScPositive]  = PreambleB_FFT[0:ScPositive]          
    IFFT_Buffer1024[-ScNegative:]  = PreambleB_FFT[-ScNegative:]  
    IFFT_Buffer1024[0]             = 0
    PreambleB                      = np.sqrt(1024)*np.fft.ifft(IFFT_Buffer1024)

    return PreambleB


# ---------------------------------------------------------------
# > Test bench
# ---------------------------------------------------------------
if __name__ == '__main__':
    GeneratePreambleB()