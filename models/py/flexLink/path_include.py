
# depending on location of the other libraries used adjust this file. 

import os
import sys  

OriginalWorkingDirectory = os.getcwd()   # Get the current directory
# DirectoryOfThisFile      = os.path.dirname(__file__)   # FileName = os.path.basename
DirectoryOfThisFile      = os.path.dirname(os.path.abspath(__file__))   # FileName = os.path.basename

if DirectoryOfThisFile != '':
    os.chdir(DirectoryOfThisFile)        # Restore the current directory

# There are modules that we will want to access and they sit two directories above DirectoryOfThisFile
sys.path.append(DirectoryOfThisFile + "\\..\\DspComm")
sys.path.append(DirectoryOfThisFile + "\\..\\KoliberEng")
# sys.path.append(os.path.dirname(DirectoryOfThisFile + "\\..\\DspComm"))
# sys.path.append(os.path.dirname(DirectoryOfThisFile + "\\..\\KoliberEng"))


# import os, sys
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(CURRENT_DIR))