import numpy as np
from utils import PowerSpectralDensity, FFT, freq_PSD, inner_prod, waveform

def deriv_waveform(fdot,t,phi,eps):
    """
    This is a function. It takes in a value of the amplitude $a$, frequency $f$ and frequency derivative $\dot{f}
    and a time vector $t$ and spits out whatever is in the return function. Modify amplitude to improve SNR. 
    Modify frequency range to also affect SNR but also to see if frequencies of the signal are important 
    for the windowing method. We aim to estimate the parameters $a$, $f$ and $\dot{f}$.
    """
    a = 5e-21
    f = 1e-3
    return (a *(np.sin(((2*np.pi)*(f*t + 0.5*fdot * t**2)*(1-eps) + phi))))

def Gaussian(values,mean,std):
    return np.exp(-(values - mean)**2 / (2*std**2))


def llike(fdot,t, data_f, variance_noise_f,eps):
    """
    Computes log likelihood 
    Assumption: Known PSD otherwise need additional term
    Inputs:
    data in frequency domain 
    Proposed signal in frequency domain
    Variance of noise
    """

    signal_prop_t = waveform(fdot,t,eps)
    signal_prop_f = FFT(signal_prop_t)

    inn_prod = sum((abs(data_f - signal_prop_f)**2) / variance_noise_f)

    return(-0.5 * inn_prod)