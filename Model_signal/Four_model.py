from math import pi
from os.path import dirname, join as pjoin
from scipy.io import wavfile
import scipy.io
from scipy.integrate import quad
from numpy import *
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft
from pywt import wavedec, integrate_wavelet
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pywt
import pylab

''' Модельные сигналы'''
######### SIN 100 Hertz ####################
# y_sin_100=[sin(2*pi*100*t) for t in arange(0,10,0.001)]
# x_sin_100=[t for t in arange(0,10,0.001)]
# plt.plot(x_sin_100,y_sin_100)
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.title("Signal 1 - sin 100 Hertz")
# plt.show()
# def sin_(t):
#     return sin(2*pi*100*t)
# ######### summ SIN 100 200 1000 Hertz ####################
# y_summ_sin=[sin(2*pi*100*t) + sin(2*pi*200*t) + sin(2*pi*1000*t) for t in arange(0,10,0.0001)]
# x_summ_sin=[t for t in arange(0,10,0.0001)]
# plt.plot(x_summ_sin,y_summ_sin)
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.title("Signal 2 - summ SIN 100 200 1000 Hertz")
# plt.show()

######### chirped 10 100 500 1000Hertz ####################
N=100000
dt = 1e-5
xa = np.linspace(0, 1, num=N)
xb = np.linspace(0, 1/4, num=int(N/4))
frequencies = [10, 100, 500, 1000]
y1a, y1b = np.sin(2*np.pi*frequencies[0]*xa), np.sin(2*np.pi*frequencies[0]*xb)
y2a, y2b = np.sin(2*np.pi*frequencies[1]*xa), np.sin(2*np.pi*frequencies[1]*xb)
y3a, y3b = np.sin(2*np.pi*frequencies[2]*xa), np.sin(2*np.pi*frequencies[2]*xb)
y4a, y4b = np.sin(2*np.pi*frequencies[3]*xa), np.sin(2*np.pi*frequencies[3]*xb)

y_chirped_sin = np.concatenate([y1b, y2b, y3b, y4b])
n = len(y_chirped_sin)  # length of the signal
k = np.arange(n)
x_chirped_sin =10*k/N
# plt.plot(x_chirped_sin , y_chirped_sin, 'b')
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.title("Динамический сигнал")
# plt.show()

######### Heart ####################
# t0 = 6*1e-2
# def A_a_p(t):
#     return (1- exp(-t/8))*(exp(-t/16)*sin(pi*t/60))
# def S_aorta(t):
#     return A_a_p(t)*sin((24.3*t + 451.3*sqrt(fabs(t)))*2*pi*(1e-3))
# def S_pulmonary(t):
#     return 0.5*A_a_p(t-t0)*sin((21.83*(t-t0)+356.34*sqrt(fabs(t-t0)))*2*pi*(1e-3))
#
# def Second_heart_sound_signal(t):
#     return S_aorta(t)+S_pulmonary(t)
# x_Heart=[t for t in arange(0,100,0.001)]
# S_arr_Heart =[Second_heart_sound_signal(i) for i in x_Heart]
# for i in range(len(x_Heart)):
#     x_Heart[i] =x_Heart[i]/10
# plt.plot(x_Heart,S_arr_Heart)
# plt.show()
'''Wavelet transform'''

############## first sin 100 Hertz#################

scales =np.arange(1,300)
waveletname= 'morl'

# dt_sin_100= 0.001
# [coefficients_sin_100, frequencies_sin_100] = pywt.cwt(y_sin_100, scales, waveletname, dt_sin_100)
# power = (abs(coefficients_sin_100)) ** 2
# period = 1. / frequencies_sin_100

# fig, ax = plt.subplots(figsize=(7, 7))
# im = ax.contourf(x_sin_100, frequencies_sin_100,coefficients_sin_100)
# fig.colorbar(im)
# plt.xlabel("Time,s")
# plt.ylabel("Frequency, H")
# plt.title("sin 100 Hertz")
# plt.legend()
# plt.show()
''' Power sin 100'''
# fig, ax = plt.subplots(figsize=(7, 7))
# fig.colorbar(im)
# plt.plot(frequencies_sin_100,power)
# plt.xlabel("Time,s")
# plt.ylabel("Frequency, H")
# plt.title("sin 100 Hertz")
# plt.legend()
# plt.show()
############## second sum sin #################
# dt_summ_sin = 0.0001
# [coefficients_summ_sin, frequencies_summ_sin] = pywt.cwt(y_summ_sin, scales, waveletname, dt_summ_sin)
# power = (abs(coefficients_summ_sin)) ** 2
# period = 1. / frequencies_summ_sin
#
# fig, ax = plt.subplots(figsize=(7, 7))
# im = ax.contourf(x_summ_sin, frequencies_summ_sin,coefficients_summ_sin)
# fig.colorbar(im)
# plt.xlabel("Time,s")
# plt.ylabel("Frequency, H")
# plt.title("sum sin  Hertz")
# plt.legend()
# plt.show()
''' Power second sum sin'''
# fig, ax = plt.subplots(figsize=(7, 7))
# plt.plot(frequencies_summ_sin,power)
# plt.xlabel("Frequency, H")
# plt.ylabel("Amplitude")
# plt.title("summ sin ")
# plt.legend()
# plt.show()
############## third  chirped sin #################
dt_chirped_sin = 1e-5
[coefficients_chirped_sin, frequencies_chirped_sin] = pywt.cwt(y_chirped_sin, scales, waveletname, dt_chirped_sin)
power = (abs(coefficients_chirped_sin)) ** 2
period = 1. / frequencies_chirped_sin
#
# fig, ax = plt.subplots(figsize=(15, 10))
# im = ax.contourf(x_chirped_sin, frequencies_chirped_sin,coefficients_chirped_sin)
# fig.colorbar(im)
# plt.show()

''' Power second sum sin'''
fig, ax = plt.subplots(figsize=(7, 7))
plt.plot(frequencies_chirped_sin,power)
plt.xlabel("Frequency, H")
plt.ylabel("Amplitude")
plt.title("third  chirped sin ")
plt.legend()
plt.show()
############## fourth  Heart signal #################
# dt_heart = 1e-3
# [coefficients_heart, frequencies_heart] = pywt.cwt(S_arr_Heart, scales, waveletname, dt_heart)
# power = (abs(coefficients_heart)) ** 2
# period = 1. / frequencies_heart
#
# fig, ax = plt.subplots(figsize=(15, 10))
# im = ax.contourf(x_Heart, frequencies_heart,coefficients_heart)
# fig.colorbar(im)
# plt.show()

# print(pywt.wavelist(kind='continuous'))
