import h5py
import os
import argparse
import glob

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib import rcParams

import numpy as np
import scipy as sp
from scipy import optimize
from scipy.constants import pi, c, mu_0

rcParams.update({'figure.autolayout': True})
rcParams.update({"font.size": 9})

parser = argparse.ArgumentParser()
parser.add_argument("location")
args = parser.parse_args()

core_radius = 125e-6
T = 293.0

try:
    sim_location = os.path.join("C:\PhD\Experiments\HISOL\SPM to soliton regime\simulation", args.location)
    os.chdir(sim_location)
except:
    print("No such dir? Wrong dir?")

try:
    gas = args.location[:2]
    pressure_mbar = int(((args.location).split(gas))[1].split("mbar")[0])
    τfwhm_fs = int(((args.location).split("mbar"))[1].split("fs")[0])
except:
    print("File names are not in the appropriate notation, soz...")

def chrng(x):
    return (2*pi*c)/x

def ρ(p, T, p0=1.0, T0=273.0):
  ρ = (p/p0)*(T0/T)
  return ρ

def n_gas(λ, gas, p, T):
    if gas == "Ar":
        B = [20332.29e-8, 34458.31e-8]
        C = [206.12e-6, 8.066e-3]
    elif gas == "Kr":
        B = [26102.88e-8, 56946.82e-8]
        C = [2.01e-6, 10.043e-3]
    χ = B[0]*(λ*1e6)**2/((λ*1e6)**2 - C[0]) + B[1]*(λ*1e6)**2/((λ*1e6)**2 - C[1])
    ngas = sp.sqrt(1.0 + ρ(p, T)*χ)
    return ngas

def χ3(gas, p, T):
    if gas == "Ar":
        χ3 = 4.0*23.5*3.43e-28
    elif gas == "Kr":
        χ3 = 4.0*64.0*3.43e-28
    χ3gas = ρ(p, T)*χ3
    return χ3gas

def β_capillary(ω, gas, p, core_radius, T):
    β = ω/c*sp.sqrt(n_gas(chrng(ω), gas, p, T)**2 - ((2.4048**2)*(c**2)/((core_radius**2)*ω**2)))
    return β  

def β2(ω, gas, p, core_radius, T):
    h = 1e13
    return (β_capillary(ω+h, gas, p, core_radius, T)- 2*β_capillary(ω, gas, p, core_radius, T) + β_capillary(ω-h, gas, p, core_radius, T))/(h**2)

def λzd(ω, β2, gas, p, core_radius, T):
    return chrng(sp.optimize.brentq(β2, chrng(200e-9), chrng(2000e-9), args=(gas, p, core_radius, T)))*1e9

def τ0(τfwhm):
    return τfwhm/(2*sp.log(1 + sp.sqrt(2)))

def L_dis(p, τfwhm, λpump, gas, core_radius, T):
    return τ0(τfwhm)**2/abs(β2(chrng(λpump), gas, p, core_radius, T))

def P_peak(E, τfwhm):
    return 0.88*E/τfwhm

def γ(gas, p, core_radius, λpump, T):
    return (3/4)*(chrng(λpump)*mu_0*χ3(gas, p, T))/(0.48*sp.pi*core_radius**2*n_gas(λpump, gas, p, T)**2)

def L_nl(gas, p, E, τfwhm, core_radius, λpump, T):
    return 1/(γ(gas, p, core_radius, λpump, T)*P_peak(E, τfwhm))

def Nsol(gas, p, E, τfwhm, λpump, core_radius, T):
    return sp.sqrt(L_dis(p, τfwhm, λpump, gas, core_radius, T)/L_nl(gas, p, E, τfwhm, core_radius, λpump, T))

def field_envelope(E):
    Eω = sp.fft(E)
    Eω[len(E)//2+1:] = 0.0
    Et = 2.0*sp.ifft(Eω)
    return Et

def browse_sims():
    files = os.listdir()
    sim_filenames = []
    for file in files:
        if file[:2] == gas:
            sim_filenames.append(file)
    N = len(sim_filenames)
    return sim_filenames, N

def load_sim(filename):
    sim = h5py.File(filename, "r")
    ω = np.array(sim["W"])
    t = np.array(sim["T"])
    E = np.array(sim["E"]) 
    modes = E.shape[1]
    energy_in = np.array(sim["input_energies"])[0]
    energy_out = np.array(sim["stats_energy_m"])[-1,0]
    energy = np.array([energy_in, energy_out])
    pressure = sim["gas_pressure"][()]
    τfwhm = np.array(sim["input_fwhms"])[0]
    sim.close()
    return ω, t, E, energy, pressure, τfwhm, modes

def plot_spectrogramE(τgate_fwhm, t_lims=[-25e-15, 25e-15], ω_lims=[chrng(1600e-9), chrng(200e-9)], t_compression_ratio=16, ω_compression_ratio=8):
    gate = lambda time, delay, τgate: np.exp(-(time-delay)**2/(2*τgate**2))
    τgate = τgate_fwhm/(2*np.sqrt(np.log(2)))
    sim_filenames, N = browse_sims()
    ω, t, E, energy, pressure, τfwhm, modes = load_sim(sim_filenames[0])
    time = t[::t_compression_ratio]
    time_lims = np.asarray([tt for tt in t if tt>t_lims[0] and tt<t_lims[1]])
    ω_pivot = len(ω)
    afreq = ω[::ω_compression_ratio]
    afreq_lims = np.asarray([ww for ww in ω if ww>ω_lims[0] and ww<ω_lims[1]])
    afreq_lims_idx1 = np.argmin(np.abs(ω-afreq_lims[0]))
    afreq_lims_idx2 = np.argmin(np.abs(ω-afreq_lims[-1]))
    wl = chrng(afreq)*1e9
    wl[0] = 6e5
    energies = np.zeros(N)
    for idx, file in enumerate(sim_filenames):
        energy_value = float(((file).split("E="))[1].split("e")[0])
        energy_exponent = int(((file).split("e"))[1].split("_")[0])
        energy = energy_value*10**energy_exponent
        energies[idx] = energy
    Elim_lf = float(input("What energy do you want to plot? "))
    Elim_idx = np.argmin(np.abs(energies-Elim_lf))
    Elim = energies[Elim_idx]
    filename = (args.location).split("10fs")[0]+"_i"+(10*"0"+str(Elim_idx))[len(str(Elim_idx)+10*"0")-10:]+"*"
    filename = glob.glob(filename)[0]
    _, _, E, _, _, _, _ = load_sim(filename)
    # zdw = λzd(afreq, β2, gas, pressure, core_radius, T)
    spectrogram = np.zeros((len(time_lims), len(afreq_lims)))
    for idx, delay in enumerate(time_lims):
        spec = sp.real(abs(sp.fft(E[-1][0][:]*gate(t, delay, τgate))[:ω_pivot])**2)
        spectrogram[idx,:] = spec[afreq_lims_idx1:afreq_lims_idx2+1]
        # spectrogram[idx,:] = 10*np.log10(spec[afreq_lims_idx1:afreq_lims_idx2+1]/spec[afreq_lims_idx1:afreq_lims_idx2+1].max())
    plt.figure(dpi=250)
    plt.pcolormesh(time_lims*1e15, chrng(afreq_lims)*1e9, np.transpose(spectrogram), rasterized=True)
    # plt.pcolormesh(time_lims*1e15, chrng(afreq_lims)*1e9, np.transpose(spectrogram), vmin=-30)
    # plt.title("Spectrogram simulation: " + str(τfwhm_fs) + " fs input duration, " + str(Elim*1e6) + " μJ input energy, " + str(pressure_mbar) + " mbar Ar, mode HE1" + str(mode+1))
    # plt.axhline(y=800, color="white", linestyle=":", linewidth=0.5)
    # plt.axhline(y=zdw, color="white", linestyle="--", linewidth=0.5)
    plt.title("Spectrogram simulation: \n" + str(τfwhm_fs) + " fs input duration, " + str(round(Elim*1e6, 2)) + " μJ input energy, " + str(pressure_mbar) + " mbar Ar, mode HE11", fontsize=8)
    plt.xlabel("Delay [fs]")
    plt.ylabel("Wavelength [nm]")
    plt.yticks([200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])

#def plot_spectrogramZ

#def animate_spectrogram

plot_spectrogramE(3e-15, [-20e-15, 50e-15], [chrng(2000e-9), chrng(200e-9)])

plt.show()