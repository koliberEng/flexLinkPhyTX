# --------------------------------------------------------------------------------
# File:       LTE_CyclicRedundancyCheck.py
# References: This file supports the book 'Digital Signal Processing in Modern Communication Systems (Edition 2)
#             -> Chapter 5 Section 6.1   
#             -> TS36.212 Section 5.1.1 (LTE 3GPP Specifications)

__title__     = "LTE_CyclicRedundancyCheck.py"
__author__    = "Andreas Schwarzinger"
__status__    = "released"
__date__      = "Feb, 5, 2022"
__version__   = "1.0.0.0"
__copyright__ = "2022 - Andreas Schwarzinger (MIT License)"

import numpy   as np
import logging                # If you are serious about writing Python professionally, start using loggers.
from   typing  import Tuple   # This module aids in type documentation of functions. 

# ---------------------------------------------------------------------------------------------------------------
# Brief       -> The cyclic redundancy check (CRC) is an error-detection code commonly used in digital networks, 
#                storage and communication systems. Blocks of data entering these systems get a short check value 
#                attached, based on the remainder of a polynial division. The Lte specification provides 4 polynomials
#                that can be selected. If none of them are selected, a custom polynomial may be used. After the CRC 
#                is computed and appended to the message, a forward error correction algorithm such as Binary Convoluation
#                Coding (BCC), Turbo Coding, LDPC Coding or Polar Coding is performed on the total bit message.
#                At the receiver, the CRC and message bit block are extracted and the CRC algorithm is run once again
#                given the message bit block. This new CRC is compared to the received CRC sequence. If they do not match
#                the message contains errors and must be discarded.
#
# Inputs      -> Message Bits     (as np.array)
#             -> logger           (as logging.logger) - loggers are very handy (use them)
#             -> Mode String      (as str)        - 'ltecrc24a', 'ltecrc24b', 'ltecrc16', ltecrc8', otherwise
#             -> InputPolynomial  (as np.array)   - use this input polynomial
#                          
# Output      -> The CRC Bit sequence
def CrcGenerator(MessageBits:       np.ndarray      # I like documenting the input type as I come from the C++/C world
               , logger:            logging.Logger  # This, however, is not required in Python
               , ModeString:        str = 'polynomial'
               , InputPolynomial:   np.ndarray = np.zeros([0,0])
               ) -> Tuple[np.ndarray, bool]:
    
    # Define Output
    CRC = np.zeros([1,1])

    # ------------------------------------------------------------------------------------------------------------------------------
    # > Type checking (Boring but when you are supposed to produce professional code, there is no such thing as enough error checking)
    # ------------------------------------------------------------------------------------------------------------------------------
    if (ModeString == None): ModeString = 'polynomial'

    Success = True
    if(type(MessageBits) != np.ndarray):
        Success = False
        logger.log(logging.ERROR, "Input argument 1 must be of type 'np.ndarray'")

    if(type(ModeString) != str):
        Success = False
        logger.log(logging.ERROR, "Input argument 3 must be of type 'str'")
    
    if(type(InputPolynomial) != np.ndarray):
        Success = False
        logger.log(logging.ERROR, "Input argument 4 must be of type 'np.ndarray'")

    if (Success == False):
        return CRC, False

    # -----------------------------------------------------------------
    # > Select the Polynomial
    # -----------------------------------------------------------------
    if  (ModeString.lower() == 'ltecrc24a'):
        Polynomial = np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1], dtype=np.int)
    elif(ModeString.lower() == 'ltecrc24b'):
        Polynomial = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1], dtype=np.int)
    elif(ModeString.lower() == 'ltecrc16'):
        Polynomial = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype=np.int)
    elif(ModeString.lower() == 'ltecrc8'):
        Polynomial = np.array([1, 1, 0, 0, 1, 1, 0, 1, 1], dtype=np.int)
    elif(len(InputPolynomial) == 0):
        logger.log(logging.ERROR, 'The custom polynomial was required but it is empty.')
        return CRC, False
    else:
        Polynomial = InputPolynomial    

    if (Polynomial[0] != 1):
        logger.log(logging.ERROR, "The CRC polynomial must feature '1' in the first position")
        return CRC, False

    # ---------------------------------------------------------------
    # > Compute the CRC
    # ---------------------------------------------------------------
    TempMessage = np.hstack([MessageBits, np.zeros(len(Polynomial)-1, dtype=np.int)])
    for Index in range(0, len(MessageBits)):
        if(TempMessage[Index] != 0):
            Range              = range(Index, Index + len(Polynomial))
            TempMessage[Range] = np.mod(TempMessage[Range] + Polynomial, 2)
        if (TempMessage.sum() == 0): break

    CRC = TempMessage[-(len(Polynomial) - 1):]

    return CRC, Success




# ------------------------------------------------------------------------------
# Test bench
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Create  a logger object that the function above will use.
    FileName  = 'Debugger.log'
    logger    = logging.getLogger('Test_Logger')
    logger.setLevel(logging.DEBUG)                      # This level must be <= the level of the f_handler
    f_handler = logging.FileHandler(FileName, mode='w') # Logs to file
    f_handler.setLevel(logging.DEBUG)                   # This level must be >= the level of the base logger.
    format    = logging.Formatter('[%(asctime)s.%(msecs)03d] %(levelname)s: %(message)s                - (%(filename)s:%(lineno)s)', datefmt = '%d-%b-%Y  %H:%M:%S')
    f_handler.setFormatter(format)
    logger.addHandler(f_handler)
     
    # Generate a random bit stream for all our test 
    RandomGenerator   = np.random.default_rng(seed = 12345)   # Make sure you have an up to date version of numpy
    NumberMessageBits = 100                                   # If not type >pip install numpy --upgrade --user
    MessageBits       = RandomGenerator.integers(low=0, high=2, size=NumberMessageBits) 

    # Test 1 -> Insert our own polynomial
    Polynomial      = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1], dtype=np.int)
    Output, Success = CrcGenerator(MessageBits, logger, '', Polynomial)
    assert Success == True, 'Test 1 failed. Check log file Debugger.log.'

    # Test 2 -> Insert our own polynomial
    Output, Success = CrcGenerator(MessageBits, logger, 'ltecrc24a')
    assert Success == True, 'Test 2 failed. Check log file Debugger.log.'

    # Test 3 -> Insert our own polynomial
    Output, Success = CrcGenerator(MessageBits, logger, 'ltecrc24b')
    assert Success == True, 'Test 3 failed. Check log file Debugger.log.'

    # Test 4 -> Insert our own polynomial
    Output, Success = CrcGenerator(MessageBits, logger, 'ltecrc16')
    assert Success == True, 'Test 4 failed. Check log file Debugger.log.'

    # Test 4 -> Insert our own polynomial
    Output, Success = CrcGenerator(MessageBits, logger, 'ltecrc8')
    assert Success == True, 'Test 5 failed. Check log file Debugger.log.'

    if(Success == True):
        print('All test have passed.')