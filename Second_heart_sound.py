#S2 = Sa + Sb
from math import *
import numpy as np
import matplotlib.pyplot as plt
const_a_= 1
const_p =2
t0 = 10*1e-3
def A_a_p(t):
    return (1- exp(-t/8))*(exp(-t/16)*sin(pi*t/60))
def S_aorta(t):
    return const_a_*A_a_p(t)*sin((24.3*t + 451.4*sqrt(fabs(t)))*2*pi*(1e-3))
def S_pulmonary(t):
    return const_p*A_a_p(t-t0)*sin((21.83*(t-t0)+356.34*sqrt(fabs(t-t0)))*2*pi*(1e-3))

def Second_heart_sound_signal(t):
    return S_aorta(t)+S_pulmonary(t)

t_arr = np.linspace(0,200,200)
S_arr =[Second_heart_sound_signal(i) for i in t_arr]
if __name__ == '__main__':
    fig, ax = plt.subplots()

    ax.plot(t_arr,S_arr)
    ax.set_title('Cигнал второго тона сердца')
    ax.set_xlabel('t,ms')
    ax.set_ylabel('S2')
    plt.show()


