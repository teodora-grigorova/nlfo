import h5py
import os
import argparse
import pickle

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams

import numpy as np
import scipy as sp
from scipy import optimize
from scipy.constants import pi, c, mu_0

rcParams.update({'figure.autolayout': True})
rcParams.update({"font.size": 9})
rcParams.update({"figure.dpi": 300})

viridis = cm.viridis
viridis_moded = [viridis(i) for i in range(0, viridis.N, 32)]
viridis_moded[0] = (1.0, 1.0, 1.0, 1.0)
white_viridis = LinearSegmentedColormap.from_list("white_viridis", viridis_moded, 8*viridis.N)

parser = argparse.ArgumentParser()
parser.add_argument("location")
args = parser.parse_args()

core_radius = 125e-6
T = 293.0
coupling = 0.2

try:
    sim_location = os.path.join("C:\\PhD\\Experiments\\HISOL\\SPM to soliton regime\\escan simulation", args.location)
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
    elif gas == "Ne":
        B = [9154.48e-8, 4018.63e-8]
        C = [656.97e-6, 5.728e-3]
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

def β1(ω, gas, p, core_radius, T):
    h = 1e13
    return (β_capillary(ω+h, gas, p, core_radius, T) - β_capillary(ω, gas, p, core_radius, T))/(h)

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

def malus(p, angle):
    return p[0]*np.cos(p[1]*np.deg2rad(angle) + p[2])**2 + p[3]

def load_escan(path):
    with open(path, "rb") as file:
        escan = pickle.load(file, encoding="latin1")
    data = {}
    p_fit = np.asarray(escan["fit parameters"])
    out_angles = np.asarray(escan["angles"])
    data["est coupled power"] = 1.5625*coupling*malus(p_fit, out_angles)
    data["out powers"] = np.asarray(escan["output powers"])
    return data

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
    #z = np.array(sim["Z"])
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
    energy_in = np.array(sim["input_energies"])[0]
    energy_out = np.array(sim["stats_energy_m"])[-1,0]
    energy = np.array([energy_in, energy_out])
    pressure = sim["gas_pressure"][()]
    τfwhm = np.array(sim["input_fwhms"])[0]
    sim.close()
    #return z, ω, t, Eω, Et, energy, pressure, τfwhm, modes
    return ω, t, Eω, Et, energy, pressure, τfwhm, modes

def plot_escan(t_compression_ratio, ω_compression_ratio):
    sim_filenames, N = browse_sims()
    ω, t, Eω, Et, energy, pressure, τfwhm, modes = load_sim(sim_filenames[0])
    #ω, t, Eω, _, energy, pressure, τfwhm, modes = load_sim(sim_filenames[0])
    time = t[::t_compression_ratio]
    afreq = ω[::ω_compression_ratio]
    Etime = np.zeros((N, modes, len(t)//t_compression_ratio))
    Espec = np.zeros((N, modes, len(t)//(2*ω_compression_ratio) + 1))
    Espec_full = np.zeros((N, len(t)//(2*ω_compression_ratio) + 1))
    energies = np.zeros((N, len(energy)))
    wl = chrng(afreq)*1e9
    wl[0] = 6e5
    print("Simulation has " + str(modes) + " modes.")
    modes_to_plot = int(input("How many modes do you want to plot? "))
    Elim = float(input("Up to what input energy do you want to plot? "))
    for idx, file in enumerate(sim_filenames):
        #_, _, Eω, Et, energy, _, _, _ = load_sim(file)
        _, _, Eω, Et, energy, _, _, _ = load_sim(file)
        for mode in range(0, modes_to_plot):
            energies[idx] = energy
            #Etime[idx,mode,:] = Et[-1,mode,::t_compression_ratio]/Et[-1,mode,::t_compression_ratio].max()
            Etime[idx,mode,:] = Et[-1,mode,::t_compression_ratio]
            Espec[idx,mode,:] = ((afreq**2/(4*sp.pi**2*c))*Eω[-1,mode,::ω_compression_ratio])/((afreq**2/(4*sp.pi**2*c))*Eω[-1,0,::ω_compression_ratio]).max()
        Espec_full[idx,:] = 10*np.log10(np.sum(Espec[idx], axis=0))
    Espec = 10*np.log10(Espec)
    idx_lim = np.abs(energies[:,0] - Elim).argmin()
    Nsim = Nsol(gas, pressure, energies[:idx_lim,0], τfwhm, 800e-9, core_radius, T)
    # for mode in range(0, modes_to_plot+1):
    #     if mode < modes_to_plot:
    #         if mode == 0:
    #             plt.figure(dpi=250)
    #             plt.title("Energy scan spectrum simulation: " + str(τfwhm_fs) + " fs pulses, " + str(pressure_mbar) + " mbar Ar, mode HE1" + str(mode+1))
    #             #plt.pcolormesh(wl, Nsim, Espec[:idx_lim,mode,:], vmin=-30)
    #             plt.pcolormesh(wl, energies[:idx_lim,0]*1e6, Espec[:idx_lim,mode,:], vmin=-30, rasterized=True)
    #             plt.xlim(200, 1600)
    #             plt.xlabel("Wavelength (nm)")
    #             #plt.ylabel("Soliton order")
    #             plt.ylabel("Input energy (μJ)")
    #             #plt.axvline(x=λzd(afreq, β2, gas, pressure, core_radius, T), color="white", linestyle="--")
    # #         else:
    #             plt.figure(dpi=250)
    #             plt.title("Energy scan spectrum simulation: " + str(τfwhm_fs) + " fs pulses, " + str(pressure_mbar) + " mbar Ar, mode HE1" + str(mode+1))
    #             plt.pcolormesh(wl, energies[:idx_lim,0]*1e6, Espec[:idx_lim,mode,:], vmin=-30)
    #             plt.xlim(200, 1600)
    #             plt.xlabel("Wavelength [nm]")
    #             plt.ylabel("Input energy (μJ)")
    fig = plt.figure()
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.title(str(τfwhm_fs) + " fs pulses, " + str(pressure_mbar) + " mbar Ar, " + str(modes_to_plot) + " modes")
    plt.pcolormesh(wl, energies[:idx_lim,0]*1e6, Espec_full[:idx_lim,:], vmin=-30, rasterized=True, cmap=white_viridis)
    #plt.pcolormesh(wl, Nsim, Espec_full[:idx_lim,:], vmin=-30, rasterized=True, cmap=white_viridis)
    plt.xlim(200, 1600)
    #plt.ylabel_position("right")
    #plt.ytick_right()
    #plt.yticks([50, 100, 150, 200, 250, 300])
    plt.colorbar()
    plt.text(1750, energies[idx_lim,0]*1e6, "dB",  color="black")
    #plt.text(200, energies[idx_lim,0]*1e6, "d)",  color="black")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Input energy (μJ)")
    plt.axvline(x=λzd(afreq, β2, gas, pressure, core_radius, T), color="black", linestyle="--")
    fig = plt.figure()
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.title(str(τfwhm_fs) + " fs pulses, " + str(pressure_mbar) + " mbar Ar, mode HE11")
    plt.pcolormesh(time*1e15, energies[:idx_lim,0]*1e6, Etime[:idx_lim,0,:], rasterized=True, cmap="magma_r", vmin=0, vmax=37)
    plt.pcolormesh(time*1e15, energies[:idx_lim,0]*1e6, Etime[:idx_lim,0,:]*1e-9, rasterized=True, cmap=white_viridis)
    plt.xlim(-50, 200)
    #plt.yticks([50, 100, 150, 200])
    plt.text(230, energies[idx_lim,0]*1e6, "GW",  color="black")
    #plt.text(-25, energies[idx_lim,0]*1e6, "f)",  color="black")
    plt.xlabel("Time (fs)")
    plt.ylabel("Input energy (μJ)")
    plt.colorbar()
    filename = "escan_" + str(pressure_mbar) + "mbarAr.pkl"
    path = os.path.join("C:\\PhD\\Experiments\\HISOL\\SPM to soliton regime\\experimental data", filename)
    data = load_escan(path)
    x = np.linspace(0, energies[:idx_lim,0].max()*1e6, 200)
    straight_line_x = 0.64*x
    marker_colour = cm.viridis(150)
    marker_colour2 = cm.viridis(0)
    fig = plt.figure(dpi=250)
    fig.tight_layout(rect=[0, 0.5, 1, 0.95])
    #plt.title("Energy scan throughput simulation: " + str(τfwhm_fs) + " fs pulses, " + str(pressure_mbar) + " mbar Ar, mode HE11")
    plt.title(str(pressure_mbar) + " mbar Ar")
    #plt.plot(energies[:idx_lim,0]*1e6, energies[:idx_lim,1]*1e6, "o", markersize=3, color=marker_colour, label="Output energy")
    plt.plot(x, straight_line_x, "--k", linewidth=1.0, label="linear throughput")
    plt.plot(data["est coupled power"][1:]*1e3, data["out powers"][1:]*1e3, "o", markersize=3, color=marker_colour2, label="experiment")
    plt.plot(energies[:idx_lim,0]*1e6, energies[:idx_lim,1]*1e6, linestyle="-", color=marker_colour, label="simulation")
    plt.xlim([0, data["est coupled power"][1:].max()*1e3+10])
    plt.ylim([0, straight_line_x.max()+10])
    #plt.text(0, 100, "h)",  color="black")
    #plt.yticks([0, 100, 200])
    plt.xlabel("Input energy (μJ)")
    plt.ylabel("Output energy (μJ)")
    plt.legend()

plot_escan(4, 4)

#print(β2(chrng(800e-9), "Ar", 1.158, core_radius, T))

# λ = np.linspace(200e-9, 1600e-9, 1000)
# print(λ)
# ω = chrng(λ)
# gv = β1(ω, "Ar", 467e-3, core_radius, T)

# plt.plot(λ*1e9, gv)

plt.show()

#wls = np.linspace(200e-9, 1500e-9, 1000)

#λ = λzd(chrng(wls), β2, "Kr", 12.4e-3, 160e-6, T)
#N = Nsol("Ne", 45e-3, 700e-6, 10e-15, 800e-9, 125e-6, T)
#Nsol(gas, p, E, τfwhm, λpump, core_radius, T)
#print(λ)