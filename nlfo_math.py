import numpy as np
import scipy as sp
from scipy import optimize
from scipy import fftpack
from scipy.constants import pi, c, mu_0
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline   


import matplotlib.pyplot as plt

import imports


def chrng(x):
    return (2.0*pi*c)/x


def norm_by_max(x):
    return abs(x)/abs(x).max()


# def norm_to_int1(x): # to be normalisation to integral 1
#     return x 


def log10_normed(x):
    return 10.0*np.log10(norm_by_max(x))


def log_10(x):
    return np.log10(x)


def malus(p, angles):
    return p[0]*np.cos(p[1]*np.deg2rad(angles)+p[2])**2 + p[3]


def malus_fitting(angles, energies, p0=None, showplot=False):
    def fitting_func(p, angles, energies):
        return malus(p, angles) - energies
    if p0 is None:
        p0 = [1.0, 2.0, 48.0, 0.0]
    p_fit, success = optimize.leastsq(fitting_func, p0[:], args=(angles, energies))
    if success:
        if showplot == True:
            angles_fit = np.linspace(angles[0], angles[-1], 200)
            plt.figure()
            plt.plot(angles, energies, "ok", angles_fit, malus(p_fit, angles_fit), "-b")
            plt.show()
        return p_fit
    else:
        print("The fitting was not successful.")


def deg_to_energy(p, angles):
    energies = malus(p, angles)
    return energies


def est_coupled_energy(energies, coupling, core_radius=125e-6, L=3.0):
    if (core_radius == 125e-6) and (L == 3.0):
        transmission = 0.64
    else:
        print("have not yet implemented way to calculate transmission")
    coupled_energies = (1/transmission)*coupling*energies
    return coupled_energies


def extrapolate_freespace(r, Ir):
    r0 = np.zeros(r.shape[0]+1)
    r0[1:] = r
    Ir0 = np.zeros((Ir.shape[0], Ir.shape[1]+1))
    for z in range(0, Ir.shape[0]):
        Ir_interpolation_curve = InterpolatedUnivariateSpline(r, Ir[z, :])
        Ir0[z, :] = Ir_interpolation_curve(r0)
    return r0, Ir0


def spectrogram(simdata, distance, mode=0, τgate_fwhm=2e-15, t_lims=(-50, 150), λ_lims=(200, 2000)):

    gate = lambda time, delay, τgate: np.exp(-(time-delay)**2/(2*τgate**2))

    τgate = τgate_fwhm/(2*np.sqrt(np.log(2)))

    z = simdata["z"]
    ω = simdata["ω"]
    t = simdata["t"]
    # Eω = simdata["Eω"]
    Et = simdata["Et"] 

    z_idx = np.argmin(np.abs(z-distance))

    time_lims = np.asarray([time for time in t if time>t_lims[0]*1e-15 and time<t_lims[1]*1e-15])

    afreq_lims = np.asarray([afreq for afreq in ω if afreq>chrng(λ_lims[1]*1e-9) and afreq<chrng(λ_lims[0]*1e-9)])
    afreq_lims_idx1 = np.argmin(np.abs(ω-afreq_lims[0]))
    afreq_lims_idx2 = np.argmin(np.abs(ω-afreq_lims[-1]))

    spectrogram = np.zeros((len(time_lims), len(afreq_lims)))

    for idx, delay in enumerate(time_lims):
        spec = np.real(abs(sp.fft(Et[z_idx][0][:]*gate(t, delay, τgate)))**2)
        spectrogram[idx,:] = spec[afreq_lims_idx1:afreq_lims_idx2+1]

    spectogram_data = {}
    spectogram_data["t"] = time_lims*1e15
    spectogram_data["λ"] = chrng(afreq_lims)*1e9
    spectogram_data["spectogram"] = np.transpose(spectrogram)

    return spectogram_data

if __name__ == "__main__":
    print(chrng(800e-9))
