# --------------------------------------------------------------------------------
# File:       BinaryConvolutionEncode.py
# References: This file supports the book 'Digital Signal Processing in Modern Communication Systems (Edition 2)
#             -> Chapter 5 Section 6.3   

__title__     = "BinaryConvolutionEncoder.py"
__author__    = "Andreas Schwarzinger" 
__status__    = "released"
__date__      = "Feb, 6, 2022"
__version__   = "1.0.0.0"
__copyright__ = "2022 - Andreas Schwarzinger (MIT License)"

import numpy   as np
import logging                # If you are serious about writing Python professionally, start using loggers.

# --------------------------------------------------------------------------------
# The Binary Convolution Encoder
# Brief -> A class that implements a binary convoluational encoder. 
#       -> Puncturing is not currently supported
#       -> Similarly to MatLab, you supply the constraint length and a list of octal values.
#       -> The length of that list determines the Rate = 1 / len(List)
#       -> The output of the Encoder can be formatted in different way. Check the ModeString below.
#       -> The encoder can be run with different initial states to support Tail-Biting Binary Convolutional Coding

class CBinaryConvolutionalEncoder():

    # ----------------------------------------------------------------------------------
    # The constructor: i.e. >> MyBCC = BinaryConvolutionEncoder(Logger, 7, np.array([133, 171, 165], dtype=np.int, 'soft-inverting')
    def __init__(self
               , ConstraintLength:    int
               , PolynomialsOct:      np.ndarray
               , ModeString:          str = 'hard'):  # 'hard'              - Produces 0s and 1s just as we would expect
                                                      # 'soft-noninverting' - Changes the output bits from 0/1 to -1.0/+1.0
                                                      # 'soft-inverting'    - Changes the output bits from 0/1 to +1.0/-1.0)
        """
            params: ConstraintLength   Input    An integer indicating the constraint length of the BCC code.
            params: PolynomialsOct     Input    (i.e. np.array([133, 171, 165], dtype-np.int))
            params: ModeString         Input    Either 'hard', 'soft-inverting' or 'soft-noninverting'
        """

        # Type checking
        assert type(ConstraintLength) == int,            "Input argument 2 must be of type 'int'"
        assert type(PolynomialsOct)   == np.ndarray,     "Input argument 3 must be of type 'np.ndarray'" 
        assert type(ModeString)       == str,            "Input argument 4 must be a string."
        ProperModeString = ModeString  == 'hard' or ModeString == 'soft-inverting' or ModeString == 'soft-noninverting'
        assert ProperModeString,       "Input argument 4 must be either 'hard', 'soft-inverting' or 'soft-noninverting'"
      
        # Save the input arguments as member variables of this class
        self.ConstraintLength = ConstraintLength
        self.Rate             = 1/len(PolynomialsOct)
        self.PolynomialsOct   = PolynomialsOct
        self.PolynomialsBin   = self.PolynomialsToBinary(self.PolynomialsOct)
        self.ModeString       = ModeString
        
    # ----------------------------------------------------------------------------------
    # Brief     -> What is displayed via the str() and print() statements
    def __str__(self):
        ReturnString = "Convolutional Encoder  >>Constraint Length = " + str(self.ConstraintLength)  + \
                                           "     Polynomials (oct) = " + str(self.PolynomialsOct)    + \
                                           "     Rate = "              + '{:5.3f}'.format(self.Rate) + \
                                           "     Output formatting = " + self.ModeString
        return ReturnString

    # ----------------------------------------------------------------------------------
    # Brief     -> The octal representation of the polynomials is not terribly useful for us. 
    #              Here we translate it into binary arrays.
    def PolynomialsToBinary(self
                          , PolynomialsOct:   np.ndarray
                           ) -> np.ndarray:

        # Type checking
        assert type(PolynomialsOct)   == np.ndarray,     "Input argument 3 must be of type 'np.ndarray'" 
    
        # Declare the output
        PolynomialsAsBinaryVectors = np.zeros([len(PolynomialsOct), self.ConstraintLength], dtype = np.int)

        for Index, OctNumber in enumerate(PolynomialsOct, start = 0):
            DecimalNumber = int(str(OctNumber), 8)                   # Converts a base 8 number to base 10
            BinaryVector  = (( DecimalNumber & np.flip((2**np.arange(self.ConstraintLength)),0) ) != 0).astype(int)
            PolynomialsAsBinaryVectors[Index, :] = BinaryVector

        return PolynomialsAsBinaryVectors
  

    # -----------------------------------------------------------------------------------
    # Brief   -> Encode the binary message
    def Encode(self
             , MessageBits:    np.ndarray   # must be bits
             , InitialState:   np.ndarray = np.zeros([0], dtype = np.int8)  # must be bits
              ) -> np.ndarray:

        # Type checking  
        assert type(MessageBits)  == np.ndarray,     "Input argument 1 must be of type 'np.ndarray'" 
        assert type(InitialState) == np.ndarray,     "Input argument 2 must be of type 'np.ndarray'" 

        # Define the output
        OutpuBitsPerInputBit = self.PolynomialsOct.size
        OutputBits           = np.zeros([OutpuBitsPerInputBit * MessageBits.size], dtype = np.int8)

        # If the initial state is not supplied, then assume that the initial state is the zero state
        if len(InitialState) == 0: InitialState = np.zeros(self.ConstraintLength - 1, dtype = np.int8)   

        # Verify the initial state
        assert InitialState.size == self.ConstraintLength - 1, "The initial state vector length is invalid."
        for Bit in InitialState:
            assert Bit == 0 or Bit == 1, "The input value must be a bit."
           
        # Define and initialize the shift register
        ShiftReg = np.hstack([np.array([0], dtype = np.int8), InitialState])

        # Start the encoding process
        OutputBitIndex = 0
        for MessageBitIndex, MessageBit in enumerate(MessageBits, start = 0):
            assert MessageBit == 0 or MessageBit == 1, "The input value must be a bit." 
            
            # Insert the input bit into the first position of the shift register
            ShiftReg[0]    = MessageBit

            # The XOR for each polynomial
            for PolynomialVector in self.PolynomialsBin:
                OutputBit                  = np.mod( (ShiftReg * PolynomialVector).sum(), 2)

                # If the mode string is hard, then we will continue with 0/1 bit values
                if self.ModeString != 'hard': 
                    if OutputBit == 0:                        OutputBit  = -1  # A zero must be remapped to a -1 (soft non-inverting)
                    if self.ModeString == 'soft-inverting':   OutputBit *= -1  # The bit must be reversed

                OutputBits[OutputBitIndex] = OutputBit
                OutputBitIndex += 1

            # Clock the shift register
            ShiftReg[1:] = ShiftReg[0:-1] 

        return OutputBits






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
     
     # --------------------------------------------------------------------------
    # Testing the convolutional encoder()
    # --------------------------------------------------------------------------
    # Test 1: The test vectors used to verify the encoder's performance are from TB_Conbolutional_Encoder.m
    ConstraintLength = 7
    PolynomialsOct   = np.array([133, 171, 165], dtype=np.int)
    BCC              = CBinaryConvolutionalEncoder(ConstraintLength, PolynomialsOct)

    # Test 1: Initial state = 0
    MessageBits1      = np.array([1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype = np.int8)
    OutputBits1       = BCC.Encode(MessageBits1)

    CorrectOutput1    = [1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1]

    Test1_Passed      = np.abs(OutputBits1  - CorrectOutput1).sum() == 0
    if Test1_Passed:
        logger.log(logging.INFO,  "The BinaryConvolutionalEncoder test 1 has passed.")
    else:
        logger.log(logging.ERROR, "The BinaryConvolutionalEncoder test 1 has failed.")



    # Test 2: Initial state = [1, 1, 1, 1, 1, 1]
    MessageBits2      = np.array([1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype = np.int8)
    InitialState      = np.array([1, 1, 1, 1, 1, 1], dtype = np.int8)
    OutputBits2       = BCC.Encode(MessageBits2, InitialState)
    CorrectOutput2    = [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1]

    Test2_Passed      = np.abs(OutputBits2 - CorrectOutput2).sum() == 0
    if Test2_Passed:
        logger.log(logging.INFO,  "The BinaryConvolutionalEncoder test 2 has passed.")
    else:
        logger.log(logging.ERROR, "The BinaryConvolutionalEncoder test 2 has failed.")


    
    # Test 3: Initial state = [0, 0, 0]
    ConstraintLength  = 4
    PolynomialsOct    = np.array([13, 15, 17], dtype=np.int)
    BCC              = CBinaryConvolutionalEncoder(ConstraintLength, PolynomialsOct)

    #MessageBits3      = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0], dtype = np.int8)
    #MessageBits3      = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype = np.int8)
    #MessageBits3      = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0], dtype = np.int8)
    MessageBits3      = np.array([1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0], dtype = np.int8)
    InitialState      = np.array([0, 0, 0], dtype = np.int8)
    OutputBits3       = BCC.Encode(MessageBits3, InitialState)
    #CorrectOutput3    = [1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1]
    #CorrectOutput3    = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1] 
    #CorrectOutput3    = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1]    
    CorrectOutput3     = [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1]

    Test3_Passed      = np.abs(OutputBits3 - CorrectOutput3).sum() == 0
    if Test3_Passed:
        logger.log(logging.INFO,  "The BinaryConvolutionalEncoder test 3 has passed.")
    else:
        logger.log(logging.ERROR, "The BinaryConvolutionalEncoder test 3 has failed.")