# File:       LteColdCodeGenerator.py
# Website:    www.signal-processing.net
# References: This file supports the book 'Digital Signal Processing in Modern Communication Systems (Edition 2)

__title__     = "LteColdCodeGenerator.py"
__author__    = "Andreas Schwarzinger"
__status__    = "released"
__date__      = "Feb, 10, 2022"
__version__   = "1.0.0.0"
__copyright__ = "2022 - Andreas Schwarzinger (MIT License)"

import numpy as np      

# --------------------------------------------------------------------------------
# The GoldCodeGenerator as described in 3GPP TS36.211 Section 7.2. 
# Brief ->      This function provides the Gold Code scrambling sequence used in many parts of LTE.
#               For CRS (Cell Specific Reference Signals) generation, a new scrambling sequence is 
#               provided to generate CRS for every new symbol.
#
# References -> TS36.211 Section 6.10.1.1 - Explains the QPSK mapping of the scrambled bits.
#                                           Explains how to initialized the Cold Code generator
#               TS36.211 Section 7.2      - Explains the structure of the Gold Code Generator
#
# Inputs ->     c_init_decimal            - The cold code generator initializer
#               NumberOfOutputBits        - Self Explanatory
#               shift                     - Number of idle executions before getting output sequence from the generator
#
# Output ->     The Gold Code Scrambling sequence

def GoldCodeGenerator(c_init_decimal:      int
                    , NumberOfOutputBits:  int
                    , shift:               int = 1600):   # 1600 is set by the LTE specification for CRS Generation

    # Type checking
    assert type(c_init_decimal)     == int, "Input argument 1 must be of type 'int'"
    assert type(NumberOfOutputBits) == int, "Input argument 2 must be of type 'int'"
    assert type(shift)              == int, "Input argument 3 must be of type 'int'"

    # Some initial constants
    FirstInitialCondition = np.hstack([0, np.zeros(29), 1])

    # Converting the c_init_decimal input argument into a bit vector of length 31 
    SecondInitialCondition = [int(i) for i in "{0:031b}".format(c_init_decimal)]   # Convert c_init to list of bit
    SecondInitialCondition = np.asarray(SecondInitialCondition)                    # Convert to np.array

    # Provide memory for the output vector
    Output                 = np.zeros(NumberOfOutputBits)

    # Setup GoldCodeGenerator for action
    TotalLength            = shift + NumberOfOutputBits
    Reg1                   = np.hstack([np.zeros(TotalLength), FirstInitialCondition])
    Reg2                   = np.hstack([np.zeros(TotalLength), SecondInitialCondition])
    EndLocation            = TotalLength - 1   # Indexing the last location in Reg1 and Reg2

    # The initial setup where we clock the gold code generator Shift times
    # The following method is a little harder to picture but it is faster than implementing the shift register as
    # it meant to operate in hardware
    for Iteration in range(0, shift):    # Iteration assumes value from 0 to Shift - 1
        Mod1 = np.mod(Reg1[EndLocation + 28] + Reg1[EndLocation + 31], 2)
        Mod2 = np.mod(Reg2[EndLocation + 28] + Reg2[EndLocation + 29] + 
                      Reg2[EndLocation + 30] + Reg2[EndLocation + 31], 2)
        Reg1[EndLocation] = Mod1
        Reg2[EndLocation] = Mod2
        EndLocation      -= 1

    # Now that we are finished with the first shift iterations, it is time to save the scrambling sequence
    for Iteration in range(0, NumberOfOutputBits):    # Iteration assumes value from 0 to Shift - 1
        Output[Iteration] = np.mod(Reg1[EndLocation + 31] + Reg2[EndLocation + 31], 2)
        Mod1 = np.mod(Reg1[EndLocation + 28] + Reg1[EndLocation + 31], 2)
        Mod2 = np.mod(Reg2[EndLocation + 28] + Reg2[EndLocation + 29] + 
                      Reg2[EndLocation + 30] + Reg2[EndLocation + 31], 2)
        Reg1[EndLocation] = Mod1
        Reg2[EndLocation] = Mod2
        EndLocation      -= 1

    return Output




# ------------------------------------------------------------------------------
# Test bench
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # Testing the Gold Code Generator
    # --------------------------------------------------------------------------
    # Test 1:
    c_init  = 1982345
    NumBits = 20
    ScramblingSequence        = GoldCodeGenerator(c_init, NumBits)
    ScramblingSequence_MatLab = np.array([1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0]) # From MyGoldCodeGenerator.m
    Test1_Failed              = np.abs(ScramblingSequence - ScramblingSequence_MatLab).sum() > 0
     

    # ----------------------------------------
    # Test 2:
    c_init  = 6661
    NumBits = 20
    ScramblingSequence        = GoldCodeGenerator(c_init, NumBits)
    ScramblingSequence_MatLab = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0 ]) # From MyGoldCodeGenerator.m
    Test2_Failed              = np.abs(ScramblingSequence - ScramblingSequence_MatLab).sum() > 0

    if Test1_Failed or  Test1_Failed:
        print('The Gold Code Generator test has failed.')
    else:
        print('The Gold Code Generator test has passed.')