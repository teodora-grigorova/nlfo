import numpy as np
import scipy as sp
from scipy import optimize
from scipy import fftpack
from scipy.constants import pi, c, mu_0

def chrng(x):
    return (2.0*pi*c)/x

def normbymax(x):
    return x/x.max()

def normtoint1(x): # to be normalisation to integral 1
    return x 

def log10normed(x):
    return 10.0*np.log10(normbymax(x))

    