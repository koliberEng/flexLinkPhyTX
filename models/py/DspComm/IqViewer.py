# File:  IqViewer.py
# Notes: This script will unpack an IQ file and attempt to generate a PCOLOR plot
#        indicating the power in each Resource element or resource block

import sys
sys.path.append("..\dspcomm")  # Go up a directory and down into the DspComm directory to find Channel and LteModDemod

# Module imports
import time              as Time
import numpy             as np             # Used to read in the binary waveform file at high speed
from   CommParameters    import *          # The naming is appropriate for this import method
import LteModDemod       as modem          # Used to OFDM demodulate the waveform
import tkinter           as tk             # for file dialog
from   tkinter           import filedialog # for file dialog

def ProcessV2xIqSample(bShowExternally: bool = False
                     , FileName:        str  = 'Unknown') -> np.ndarray:

    if FileName == 'Unknown' or FileName == None:
        # ------------------------------------------
        # 1. Open a file
        root     = tk.Tk()    
        root.withdraw()  # Get rid of the root window that appears after tk.Tk()
        FileName = filedialog.askopenfilename(initialdir="C:\Work\TsmxSw_LteM\VMSDev\PhylisModules\PhylisModuleTester",
                                            title = "Select an IQ file",
                                            filetypes =(("bin files", "*.bin"),
                                                        ("all files",  "*.*"))
                                            )

    assert isinstance(FileName, str), 'The provided file name must be a string.' 

    # ------------------------------------------------------------
    # 2. Read the binary file content  
    f              = open(FileName, 'rb')
    ModuleCode     = np.fromfile(f, dtype = '<u2', count = 1)[0]   # Fetch word:   Must be 271, otherwise we don't continue
    NumIqSamples   = np.fromfile(f, dtype = '<u4', count = 1)[0]   # Fetch dword:  The number of IQ samples in the file
    TimeToSubframe = np.fromfile(f, dtype = '<d',  count = 1)[0]   # Fetch double: Time to the subframe where we start to demodulate
    NSLRB          = np.fromfile(f, dtype = '<u1', count = 1)[0]   # Fetch uchar:  The number of sidelink resource blocks 100 or 50
    SampleRate     = np.fromfile(f, dtype = '<f4', count = 1)[0]   # Fetch float: The sample rate
    Floats         = np.fromfile(f, dtype = '<f4', count = NumIqSamples*2) # Same as above
    f.close()

    assert ModuleCode == 271,           'The first parameter must be the module code, which equals 271'
    assert NSLRB == 50 or NSLRB == 100, 'The number of sidelink resource blocks must be 50 or 100'
    assert SampleRate == 21120000 or SampleRate == 1024 * 15000, ' The sample rate ' + str(SampleRate) + \
                                                                ' should be either 21.12MHz or 15.36MHz.'

    # 3. Process header information in binary file
    CaptureTime    = NumIqSamples / SampleRate   
    NumSubframes   = int(CaptureTime / Params.LTE_SUBFRAME_PERIOD)      
    NumOfdmSymbols = NumSubframes    * 2 * Params.V2X_SYMBOLS_PER_SLOT           

    if NSLRB == 100:
        NRB_V2X  = ELteNRB.BwIs20MHz
        FftParam = ELteFftSizesCp.FftSize1408.value 
    else:
        NRB_V2X  = ELteNRB.BwIs10MHz      
        FftParam = ELteFftSizesCp.FftSize1024.value                                    

    ReturnString = "ModuleCode:       " + str(ModuleCode) + "\n"         + \
                   "Number Samples:   " + str(NumIqSamples) + "\n"       + \
                   "Time to Subframe: " + "{:8.7f}".format(TimeToSubframe) + " sec\n" + \
                   "NSLRB:            " + str(NSLRB) + "\n" + \
                   "Sample Rate (Hz): " + str(int(SampleRate)) + "\n" +   \
                   "Capture time:     " + str(CaptureTime) + "\n" + \
                   "Num Ofdm Symbols: " + str(NumOfdmSymbols)                                                   

    # -------------------------------------------
    # 4. Extract the IQ data from what we read in
    I            = Floats[0:NumIqSamples*2:2]
    Q            = Floats[1:NumIqSamples*2:2]
    IQ           = np.complex64(I + 1j*Q)


    # -------------------------------------------
    # 5. It's time to run the Lte Demodulator for the entire IQ waveform
    StartTime = Time.perf_counter()  

    ResourceGrid = modem.LteOfdmDemodulation(LinkType              = 'sidelink'
                                           , InputWaveform         = IQ
                                           , NRB                   = NRB_V2X
                                           , Cp                    = ELteCP.CpNormal
                                           , FftParameters         = FftParam
                                           , SubframeStartTime     = float(TimeToSubframe)
                                           , StartSymbolIndex      = 0
                                           , NumberOfOutputSymbols = NumSubframes
                                           , TimeAdvanceInSec      = 1e-6
                                           , CompensateForAdvance  = True )

    StopTime = Time.perf_counter()
    print('Elapsed Time: ' + str(StopTime - StartTime) + str(' sec'))

    return (ReturnString, ResourceGrid, I, Q)
'''
# ------------------------------------------
# Plot the data if needed
SamplePeriod = 1/SampleRate
CaptureTime  = NumIqSamples * SamplePeriod
time         = np.arange(0, CaptureTime, SamplePeriod)

bNeedPlotting = True
if(bNeedPlotting):
    plt.figure(1, figsize=(9, 6))               # manipulate figure size (size in inches)
    plt.plot(I, 'r', Q, 'b')                    # plot the IQ data versus sample index
    #plt.plot(time, I, 'r', time, Q, 'b')       # plot the IQ data versus time
    plt.grid(True)
    plt.show()



# ------------------------------------------
# Generate a Pcolor plot of the received resource grid
Z = np.array(np.abs(ResourceGrid), dtype = np.float32)

fig = plt.figure(2)            
plt.pcolor(Z,  cmap="Blues",  )
plt.title('LTE Signal Power Over Time')
plt.ylabel('Subcarriers')
plt.xlabel('Symbol Index')
plt.grid(color='#999999') 



fs = SampleRate

plt.figure(3, figsize=(9, 6))        
f, Pxx_den = signal.periodogram(IQ[240000:250000], fs)
plt.semilogy(f, Pxx_den)
#plt.ylim([1e-7, 1e2])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.grid(True)
plt.show()

halt  = 1
'''