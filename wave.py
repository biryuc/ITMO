import pywt
from pylab import *
from numpy import *
discrete_wavelets = ['db5', 'sym5', 'coif5', 'haar']
print('discrete_wavelets-%s'%discrete_wavelets )
st='db20'
wavelet = pywt.DiscreteContinuousWavelet(st)
print(wavelet)
i=1
phi, psi, x = wavelet.wavefun(level=i)
subplot(2, 1, 1)
title("График самой вейвлет - функции -%s"%st)
plot(x,psi,linewidth=2, label='level=%s'%i)
grid()
legend(loc='best')
subplot(2, 1, 2)
title("График первообразной -функции -%s"%st)
plt.plot(x,phi,linewidth=2, label='level=%s'%i)
legend(loc='best')
grid()
show()