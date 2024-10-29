"""
use to plot comms simulations
"""
"""
MIT License

Copyright (c) 2020 Koliber Engineering, koliber.eng@gmail.com
Copyright (c) 2020 Koliber Engineering, koliber.eng@gmail.com
Copyright (c) 2022 Koliber Engineering, koliber.eng@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import numpy as np
import matplotlib.pyplot as plt


class Visualization:
    def __init__(self):
        self.fig_num = 0

    def plot_data(self, data, names, start=0, points=False):
        """ 
        function plots n rows of data based on list length
            
        data : list of data to plot
        names : list of names of data
        title : plot name and file name. 
        """
        end = len(data[0])
        
        if len(data) != len(names):
            print('list of data and names are not equal length')
            return
        
        if len(data) == 1:
            self.fig_num = self.fig_num + 1 
            # fig = plt.figure()
            plt.figure(self.fig_num)
            plt.plot(data[0][start:end])
            if points:
                plt.plot(data[0][start:end], 'r.')
            plt.grid('on')
            plt.title(names[0]+'\n'+'data plot')
            plt.xlabel('x-values')
            plt.ylabel('(y) ' + names[0])
            plt.savefig('data-plot'+names[0])
            plt.show(block=False)
            # plt.show()
        
        else:
            self.fig_num = self.fig_num + 1
            # fig, ax = plt.subplots(len(data), 1)  # N by 1 subplots
            fig, ax = plt.subplots(len(data), 1, num=self.fig_num, clear=True)
            # title = ' '.join(names)
            # plt.title(title)
            for i in range(len(data)):
                ax[i].plot(data[i])
                if points:
                    ax[i].plot(data[i], 'r.')
                ax[i].grid('on')
                # ax[i].legend()
                ax[i].set_ylabel(names[i])
            plt.show(block=False)
            # plt.show()
        return

    def plot_data_xy_complex(self, x, y ):
        #view the data in time and frequency domain
        #calculate the frequency domain for viewing purposes
        #N_FFT = float(len(y))
        # N_FFT = len(y)
        # f = np.arange(0,Fs,Fs/N_FFT)
        # w = np.hanning(len(y))
        # #y_f = np.fft.fft(np.multiply(y,w)) # apply a window to the input data
        # y_f = np.fft.fft(y*w) # apply a window to the input data
        # y_f = np.fft.fftshift(y_f)
        # y_freqs = np.fft.fftshift(np.fft.fftfreq(N_FFT, d=1/Fs))
        # #y_f = 10*np.log10(np.abs(y_f[0:int(N_FFT/2)]  /N_FFT))
        # y_f = 10*np.log10(np.abs(y_f /N_FFT))
    # fignum +=1
    # plt.figure(fignum)
    # plt.clf()
    # ##plot(f/1000, abs(sig_fft), 'r.')
    # plt.plot(hh, 'r.')
    # plt.grid(True)
    # plt.show()
        #plt.figure()
        self.fig_num = self.fig_num + 1 
        plt.figure(self.fig_num, figsize=(6,6))
        plt.clf()
        plt.plot(x, y.real, 'b.')
        plt.plot(x, y.real, 'g')
        plt.xlabel('Sample')
        plt.ylabel('Amp')
        plt.title('Data plot sample vs amplitude')
        plt.grid(True)
        
    #    fignum +=1
    #    plt.figure()
    #    plt.clf()
        #plt.figure()
        self.fig_num = self.fig_num + 1 
        plt.figure(self.fig_num, figsize=(6,6))
        plt.clf()
        plt.plot(x, y.imag, 'r.')
        plt.plot(x, y.imag, 'g')
        plt.xlabel('Sample')
        plt.ylabel('Amp')
        plt.title('Data plot sample vs amplitude')
        plt.grid(True)


    def plot_data_xy(self, x, y, name='' ):
        self.fig_num = self.fig_num + 1 
        plt.figure(self.fig_num, figsize=(6,6))
        plt.clf()
        plt.plot(x, y.real, 'b.')
        plt.plot(x, y.real, 'g')
        plt.xlabel('Sample')
        plt.ylabel('Amp')
        plt.title('XY data plot'+'\n'+name)
        # plt.title('Data plot sample vs amplitude')
        plt.grid(True)




    def plot_data_1figure(self, data, names, start=0, points=False):
        """ 
        function plots n rows of data based on list length
            
        data : list of data to plot
        names : list of names of data
        title : plot name and file name. 
        """
        end = len(data[0])
        if len(data) != len(names):
            print('list of data and names are not equal length')
            return
        
        if len(data) == 1:
            # self.fig_num = self.fig_num + 1 
            fig = plt.figure()
            plt.plot(data[0][start:end])
            if points:
                plt.plot(data[0][start:end], 'r.')
            plt.grid('on')
            plt.title('Data plot'+'\n'+names[0])
            plt.xlabel('real')
            plt.ylabel('imag')
            plt.savefig('data-plot'+names[0])
            plt.show(block=False)
        
        else:
            fig = plt.figure()

            # fig, ax = plt.subplots(len(data), 1)  # N by 1 subplots
            # plt.title(title)
            for i in range(len(data)):
                data_list = list(data[i][start:end])
                plt.plot(data_list)
                if points:
                    plt.plot(data_list, 'r.')
            # plt.legend()
            plt.grid('on')
            plt.title( ' '.join(names))
            plt.xlabel('data')
            plt.savefig('data-plot'+names[0])
            plt.show(block=False)
        return

    def plot_constellation(self, data, start=-110, end=-10, name=''):
        """ 
        function plots n rows of data based on list length
            
        data : list of data to plot
        names : list of names of data
        title : plot name and file name. 
        """
        self.fig_num = self.fig_num + 1 
        # fig = 
        plt.figure(self.fig_num)
        plt.plot(np.real(data[start:end]), np.imag(data[start:end]), 'r.')
        plt.grid('on')
        plt.title('IQ constellation plot'+'\n'+name)
        plt.xlabel('real')
        plt.ylabel('imag')
        plt.savefig('IQ-constellation-plot'+name)
        plt.show(block=False)

    def plot_iq_data2x(self, data1, data2, name1, name2):
        """ 
        function plots 2 rows of data, 1 for real, and 1 for imaginary
        data1,2 : complex data to plot
        """
        # fig = make_subplots(rows=2, cols=1)
        # fig.add_trace(go.Scatter(y=np.real(data1), name=name1 + 'real', mode="lines+markers"), row=1, col=1)
        # fig.add_trace(go.Scatter(y=np.imag(data1), name=name1 + 'imag', mode="lines+markers"), row=1, col=1)
        #
        # fig.add_trace(go.Scatter(y=np.real(data2), name=name2 + 'real', mode="lines+markers"), row=2, col=1)
        # fig.add_trace(go.Scatter(y=np.imag(data2), name=name2 + 'imag', mode="lines+markers"), row=2, col=1)
        # fig.update_layout(title=name1 + " & "+name2 + " data")
        # pyo.plot(fig, filename=name1+"_"+name2+"_IQ_plots.html")

        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(np.real(data1), 'b', label='real')
        ax1.plot(np.real(data1), 'r.')
        ax1.plot(np.imag(data1), 'g', label='imag')
        ax1.plot(np.imag(data1), 'r.')
        ax1.grid('on')
        ax1.legend()
        ax2.plot(np.real(data2), 'b', label='real')
        ax2.plot(np.real(data2), 'r.')
        ax2.plot(np.imag(data2), 'g', label='imag')
        ax2.plot(np.imag(data2), 'r.')
        ax2.grid('on')
        ax2.legend()
        ax1.set_ylabel(name1)
        ax2.set_ylabel(name2)
        # ax1.set_xlim([-Tb, 4 * Tb])
        # ax2.set_xlim([-Tb, 4 * Tb])
        plt.legend()
        plt.show(block=False)

    def plot_psd(self, samples, Fs, mag,  plt_title):
        # Calculate power spectral density (frequency domain version of signal)
        psd = np.abs(np.fft.fftshift(np.fft.fft(samples))) ** 2
        psd = np.clip(psd, 1e-14, None)
        
        if mag == True: 
            psd_dB = 20 * np.log10(psd) # samples are in magnitude
        else:
            psd_dB = 10 * np.log10(psd) # sampes are in power
        
        
        freqs = np.linspace(Fs/-2, Fs/2, len(psd))

        # Plot freq domain
        self.fig_num = self.fig_num + 1 
        #fig = 
        plt.figure(self.fig_num)
        # plt.figure()
        plt.plot(freqs, psd_dB)
        plt.plot(freqs, psd_dB, 'r.' )
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("PSD")
        plt.grid('on')
        plt.title(plt_title)
        plt.show(block=False)
        #plt.show()

    # def plot_iq_psd(self, x, fs, name):
    #     zoom_len = int(len(x)/10)
    #     m5.plot_iq_data2x(x, x[0:zoom_len], name, 'zoom')
    #     m5.plot_psd(x, fs, name + ', ' +str(fs)+' sample rate')


print('visualization end')
