# File:       FlexLinkReceiver.py
# Notes:      This script unifies the receiver chain of the FlexLink modem.

# ---------------------------------------
# --------- Helper Functions ------------
#           1.  Use the PlotResourceGrid() function to visually inspect the resource grid.


__title__     = "FlexLinkReceiver"
__author__    = "Andreas Schwarzinger"
__status__    = "preliminary"
__date__      = "April, 27th, 2024"
__copyright__ = 'Andreas Schwarzinger'

# --------------------------------------------------------
# Import Statements
# --------------------------------------------------------
from   FlexLinkParameters import *
from   FlexLinkCoder      import CCrcProcessor, CLdpcProcessor, CPolarProcessor, InterleaverFlexLink, CLinearFeedbackShiftRegister
from   QamMapping         import CQamMappingIEEE
import Preamble
import numpy              as np
import matplotlib.pyplot  as plt



# --------------------------------------------------------
# > Class: CFlexLinkReceiver
# --------------------------------------------------------
class CFlexLinkReceiver():
    '''
    brief: A class that unifies the FlexLink transmitter portion of the modem
    '''