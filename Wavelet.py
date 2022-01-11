import numpy as np
from scipy import fftpack
from pylab import*
N=100000
dt = 1e-5
xa = np.linspace(0, 1, num=N)
xb = np.linspace(0, 1/4, int(N/4))
frequencies = [4, 30, 60, 90]
y1a, y1b = np.sin(2*np.pi*frequencies[0]*xa), np.sin(2*np.pi*frequencies[0]*xb)
y2a, y2b = np.sin(2*np.pi*frequencies[1]*xa), np.sin(2*np.pi*frequencies[1]*xb)
y3a, y3b = np.sin(2*np.pi*frequencies[2]*xa), np.sin(2*np.pi*frequencies[2]*xb)
y4a, y4b = np.sin(2*np.pi*frequencies[3]*xa), np.sin(2*np.pi*frequencies[3]*xb)
def spectrum_wavelet(y):
    Fs = 1 / dt  # sampling rate, Fs = 0,1 MHz
    n = len(y)  # length of the signal
    k = np.arange(n)
    T = n / Fs
    frq = k / T  # two sides frequency range
    frq = frq[range(n // 2)]  # one side frequency range
    Y = fftpack.fft(y) / n  # fft computing and normalization
    Y = Y[range(n // 2)] / max(Y[range(n // 2)])
    # plotting the data
    subplot(2, 1, 1)
    plot(k/N , y, 'b')
    ylabel('Amplitude')
    grid()
    # plotting the spectrum
    subplot(2, 1, 2)
    plot(frq[0:140], abs(Y[0:140]), 'r')
    xlabel('Freq')
    plt.ylabel('|Y(freq)|')
    grid()
y= y1a + y2a + y3a + y4a
spectrum_wavelet(y)
show()