# Filename: FreqOffsetAcquisition.py
# The following script supports the the FlexLink standard. It discusses methods of frequency offset detection
# at the start of a packet

__title__     = "FreqOffsetAcquisition"
__author__    = "Andreas Schwarzinger"
__status__    = "preliminary"
__date__      = "Sept, 4rd, 2022"
__copyright__ = 'Andreas Schwarzinger'


# Import modules
import numpy as np
import math

def GeneratePreampleOne