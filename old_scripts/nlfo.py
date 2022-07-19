import numpy as np
import scipy as sp
from scipy import optimize
from scipy import fftpack
from scipy.constants import pi, c, mu_0

import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams

import h5py

viridis = plt.cm.viridis
viridis_moded = [viridis(i) for i in range(0, viridis.N, 32)]
viridis_moded[0] = (1.0, 1.0, 1.0, 1.0)
white_viridis = LinearSegmentedColormap.from_list("white_viridis", viridis_moded, 8*viridis.N)

width = 7.08
height = 3.42

def test_import():
    print("nlfo is imported")

def update_rcParams():
    rcParams.update({"figure.autolayout": True}) # tight_layout
    rcParams.update({"font.size": 14})

def chrng(x):
    return (2.0*pi*c)/x

def ρ(p, T, p0=1.0, T0=273.0):
  ρ = (p/p0)*(T0/T)
  return ρ

def n_gas(λ, p, T, gas):
    if gas == "Ar":
        B = [20332.29e-8, 34458.31e-8]
        C = [206.12e-6, 8.066e-3]
    χ = B[0]*(λ*1e6)**2/((λ*1e6)**2 - C[0]) + B[1]*(λ*1e6)**2/((λ*1e6)**2 - C[1])
    ngas = sp.sqrt(1.0 + ρ(p, T)*χ)
    return ngas

def β_capillary(ω, p, core_radius, gas, T):
    β = ω/c*sp.sqrt(n_gas(chrng(ω), p, T, gas)**2 - ((2.4048**2)*(c**2)/((core_radius**2)*ω**2)))
    return β  

def β2(ω, p, core_radius=125e-6, gas="Ar", T=293.0):
    h = 1e13
    return (β_capillary(ω+h, p, core_radius, gas, T)- 2*β_capillary(ω, p, core_radius, gas, T) + β_capillary(ω-h, p, core_radius, gas, T))/(h**2)

def λzd(ω, β2, pressure, gas, core_radius, T):
  return chrng(sp.optimize.brentq(β2, chrng(3000e-9), chrng(100e-9), args=(pressure, core_radius, gas, T)))*1e9

def normbymax(x):
    return x/x.max()

def normtoint1(x): # to be normalisation to integral 1
    return x 

def log10normed(x):
    return 10.0*np.log10(normbymax(x))

def field_envelope(E):
    Eω = sp.fft(E)
    Eω[len(E)//2+1:] = 0.0
    Et = 2.0*sp.ifft(Eω)
    return Et

def get(E):
    Eω = sp.fft(E)
    Eω[len(E)//2+1:] = 0.0
    Et = 2.0*sp.ifft(Eω)
    return Et

def load_sim(filename):
    sim = h5py.File(filename, "r")
    z = np.array(sim["Z"])
    ω = np.array(sim["W"])
    ω_pivot = len(ω)
    t = np.array(sim["T"])
    E = np.array(sim["E"]) 
    steps = E.shape[0]
    modes = E.shape[1]
    Et = np.zeros((steps, modes, len(t)))
    for step in range(0, steps):
        for mode in range(0, modes):
            Et[step, mode, :] = abs(field_envelope(E[step][mode][:]))**2
    Eω = np.zeros((steps, modes, ω_pivot))
    for step in range(0, steps):
        for mode in range(0, modes):
            Eω[step, mode, :] = sp.real(abs(sp.fft(E[step][mode][:])[:ω_pivot])**2)
    pressure = sim["gas_pressure"][()]
    τfwhm = np.array(sim["input_fwhms"])[0]
    sim.close()
    return ω, t, Eω, Et, z, pressure, τfwhm, modes

def plot_zprop_spec(filename, wl_lims=[200, 1600], mode=0, zdw=True, colourbar=True):
    ω, _, Eω, _, z, pressure, τfwhm, modes = load_sim(filename)
    assert modes > mode, "mode to plot set higher than the number of modes in the simulation"
    wl_min = min(wl_lims)
    wl_max = max(wl_lims)
    wls = chrng(ω)*1e9
    wls[np.isinf(wls)] = 6e-4
    Eω = log10normed(Eω)
    print("z-propagated spectrum: " + str(τfwhm*1e15) + " fs pulses, " + str(pressure*1e3) + " mbar Ar, mode HE1" + str(mode+1))
    fig = plt.figure(dpi=300, figsize=(width, height))
    plt.pcolormesh(wls, z, Eω[:, mode], vmin=-30, cmap=white_viridis)
    plt.xlim(wl_min, wl_max)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Propagation length (m)")
    if zdw == True:
        plt.axvline(x=λzd(ω, β2, pressure, "Ar", 125e-6, 293.0), color="grey", linestyle="--", linewidth=3, alpha=0.7)
    if colourbar == True:
        plt.colorbar()
    return fig

def plot_zprop_time(filename, t_lims=[-50, 50], mode=0, colourbar=True):
    _, t, _, Et, z, pressure, τfwhm, modes = load_sim(filename)
    assert modes > mode, "mode to plot set higher than the number of modes in the simulation"
    t_min = min(t_lims)
    t_max = max(t_lims)
    print("z-propagated time: " + str(τfwhm*1e15) + " fs pulses, " + str(pressure*1e3) + " mbar Ar, mode HE1" + str(mode+1))
    fig = plt.figure(dpi=300, figsize=(width, height))
    plt.pcolormesh(t*1e15, z, Et[:, mode], cmap=white_viridis)
    plt.xlim(t_min, t_max)
    plt.xlabel("Time (fs)")
    plt.ylabel("Propagation length (m)")
    if colourbar == True:
        plt.colorbar()
    return fig

def plot_spectrogram(filename, τgate_fwhm=3e-15, wl_lims=[200, 1600], t_lims=[-50, 50], compression=4, mode=0, zdw=True, colourbar=True):
    gate = lambda time, delay, τgate: np.exp(-(time-delay)**2/(2*τgate**2))
    τgate = τgate_fwhm/(2*np.sqrt(np.log(2)))
    ω, t, Eω, Et, z, pressure, τfwhm, modes = load_sim(filename)
    assert modes > mode, "mode to plot set higher than the number of modes in the simulation"
    ω_lims = chrng(wl_limits*1e-9)
    ω_min = min(ω_lims)
    ω_max = min(ω_lims)
    wl_min = min(wl_lims)
    wl_max = max(wl_lims)
    t_min = min(t_lims)
    t_max = max(t_lims)
    time = np.asarray([tt for tt in t if tt>t_min*1e-15 and tt<t_max*1e-15])
    afreq = np.asarray([ww for ww in ω if ww>ω_min and ww<ω_max])
    afreq_idx1 = np.argmin(np.abs(ω-afreq[0]))
    afreq_idx2 = np.argmin(np.abs(ω-afreq[-1]))
    ω_pivot = len(ω)
    spectrogram = np.zeros((len(time[::compression]), len(afreq[::compression])))
    for idxτ, delay in enumerate(time[::t_compression]):
        spec = sp.real(abs(sp.fft(Et[:, mode]*gate(t, delay, τgate))[:ω_pivot])**2)
        spectrogram[idxτ, :] = spec[afreq_idx1:afreq_idx2+1:ω_compression]
    return afreq[::compression], time[::compression], spectrogram, energy, pressure, τfwhm, τgate_fwhm, "HE11"
    
    fig = plt.figure(dpi=300, figsize=(width, height))
    plt.title("z-propagated time: " + str(τfwhm*1e15) + " fs pulses, " + str(pressure*1e3) + " mbar Ar, mode HE1" + str(mode+1))
    plt.pcolormesh(t*1e15, z, Et[:, mode], cmap=white_viridis)
    plt.xlim(t_min, t_max)
    plt.xlabel("Time (fs)")
    plt.ylabel("Propagation length (m)")
    if zdw == True:
        plt.axvline(x=λzd(ω, β2, pressure, "Ar", 125e-6, 293.0), color="grey", linestyle="--", linewidth=3, alpha=0.7)
    if colourbar == True:
        plt.colorbar()
    return fig

if __name__ == '__main__':    
    plot_zprop_spec("Ar467mbar25uJ_forpaper.h5")
    plt.show()