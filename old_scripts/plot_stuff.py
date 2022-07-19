import h5py
import os
import argparse

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib import rcParams

import numpy as np
import scipy as sp
from scipy import optimize
from scipy.constants import pi, c, mu_0
import scipy.fftpack as fft

rcParams.update({'figure.autolayout': True})
rcParams.update({"font.size": 9})

# parser = argparse.ArgumentParser()
# parser.add_argument("location")
# args = parser.parse_args()

core_radius = 125e-6
T = 293.0

# sim_location = os.path.join("C:\\PhD\\Experiments\\HISOL\\SPM to soliton regime\\simulation", args.location)
# os.chdir(sim_location)

# try:
#     sim_location = os.path.join("C:\\Users\\Setaou\\Documents\\PhD\\Experiments\\HISOL\\SPM to soliton regime\\simulation\\spectrogram", args.location + "\\ionONshockON")
#     os.chdir(sim_location)
# except:
#     print("No such dir? Wrong dir?")

# try:
#     gas = args.location[:2]
#     pressure_mbar = int(((args.location).split(gas))[1].split("mbar")[0])
#     τfwhm_fs = int(((args.location).split("mbar"))[1].split("fs")[0])
# except:
#     print("File names are not in the appropriate notation, soz...")

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
    energies = []
    for file in files:
        if file[:2] == gas:
            sim_filenames.append(file)
            energy = float((file.split("E="))[1].split("e-")[0])
            energy_mag = int((file.split("e-"))[1].split("_0000")[0])
            energies.append(energy*10**(-energy_mag))
    N = len(sim_filenames)
    return sim_filenames, N, energies

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

def load_sim_prop(filename):
    sim = h5py.File(filename, "r")
    z = np.array(sim["Z"])
    ω = np.array(sim["W"])
    ω_pivot = len(ω)
    t = np.array(sim["T"])
    E = np.array(sim["E"]) 
    steps = E.shape[0]
    modes = E.shape[1]
    Et = np.zeros((steps, modes, len(t)), dtype=np.cdouble)
    for step in range(0, steps):
        for mode in range(0, modes):
            Et[step, mode, :] = field_envelope(E[step][mode][:])
    Eω = np.zeros((steps, modes, ω_pivot), dtype=np.cdouble)
    for step in range(0, steps):
        for mode in range(0, modes):
            Eω[step, mode, :] = sp.fft(E[step][mode][:])[:ω_pivot]
    stats = {}
    pressure = sim["gas_pressure"][()]
    stats["pressure"] = pressure
    energy_out = np.array(sim["stats_energy"])[-1]
    stats["output energy"] = energy_out
    energy_in = np.array(sim["input_energies"])[0]
    stats["input energy"] = energy_in
    ne = np.array(sim["stats_peak_Ne"])
    stats["free electron density"] = ne
    z_solver = np.array(sim["stats_z"])
    stats["z"] = z_solver
    stats["neutral gas density"] = sim["gas_N0"][()]
    #stats["t_fwhm_compression"] = sim["stats_fwhm_t"][-1]
    sim.close()
    return z, ω, t, Eω, Et, modes, stats
    # return z, ω, t, Et, modes, stats

def plot_zprop_with_ion(t_compression_ratio=16, ω_compression_ratio=1):
    sim_filenames, N, energies = browse_sims()
    z, ω, t, Eω, Et, modes, stats = load_sim_prop(sim_filenames[0])
    time = t[::t_compression_ratio]
    afreq = ω[::ω_compression_ratio]
    Etime = np.zeros((len(z), modes, len(t)//t_compression_ratio))
    Espec = np.zeros((len(z), modes, len(t)//(2*ω_compression_ratio) + 1))
    Espec_full = np.zeros((len(t)//(2*ω_compression_ratio) + 1))
    wl = chrng(afreq)*1e9
    wl[0] = 6e5
    modes_to_plot = int(input("How many modes do you want to plot? "))
    Eplot = float(input("What input energy do you want to plot? "))
    idx_plot = np.abs(np.asarray(energies) - Eplot).argmin()
    _, _, _, Eω, Et, _, stats = load_sim_prop(sim_filenames[idx_plot])
    for mode in range(0, modes_to_plot):
        Etime[:, mode, :] = Et[:,mode,::t_compression_ratio]/Et[:,mode,::t_compression_ratio].max()
        Espec[:, mode, :] = Eω[:,mode,::ω_compression_ratio]/Eω[:,0,::ω_compression_ratio].max()
    #Espec_full = 10*np.log10(np.sum(Espec, axis=0))
    Espec = 10*np.log10(Espec)
    #Nsim = Nsol(gas, pressure, energies[:idx_lim,0], τfwhm, 800e-9, core_radius, T)
    fig = plt.figure(dpi=300)
    gs = gridspec.GridSpec(1, 3, width_ratios=[2, 2, 1], wspace=0.1, hspace=0.1, left=0.11, right=0.92, bottom=0.15, top=0.98) 
    plt.rcParams.update({"font.size": 9})
    for mode in range(0, modes+1):
        if mode < modes:
            if mode == 0:
                ax = fig.add_subplot(gs[0])
                #plt.title("Propagation simulation: " + str(τfwhm_fs) + " fs pulses, " + str(pressure_mbar) + " mbar Ar, mode HE1" + str(mode+1))
                #plt.pcolormesh(wl, Nsim, Espec[:idx_lim,mode,:], vmin=-30)
                ax.pcolormesh(afreq*1e-15, z, Espec[:, mode, :], vmin=-30, rasterized=True)
                ax.set_xlim(chrng(2500e-9)*1e-15, chrng(270e-9)*1e-15)
                ax.set_xlabel("Angular frequency [PHz]")
                #plt.ylabel("Soliton order")
                ax.set_ylabel("Propagation distance [m]")
                #plt.axvline(x=λzd(afreq, β2, gas, pressure, core_radius, T), color="white", linestyle="--")
            # else:
            #     plt.figure(dpi=250)
            #     plt.title("Energy scan spectrum simulation: " + str(τfwhm_fs) + " fs pulses, " + str(pressure_mbar) + " mbar Ar, mode HE1" + str(mode+1))
            #     plt.pcolormesh(wl, energies[:,0]*1e6, Espec[:,mode,:], vmin=-30)
            #     plt.xlim(200, 1600)
            #     plt.xlabel("Wavelength [nm]")
            #     plt.ylabel("Soliton order")
        # else:
        #     plt.figure(dpi=250)
        #     plt.title("Energy scan spectrum simulation: " + str(τfwhm_fs) + " fs pulses, " + str(pressure_mbar) + " mbar Ar, " + str(modes_to_plot) + " modes")
        #     plt.pcolormesh(wl, z, Espec_full, vmin=-30)
        #     plt.xlim(200, 1600)
        #     plt.xlabel("Wavelength [nm]")
            # plt.ylabel("Soliton order")
    bx = fig.add_subplot(gs[1])
    # plt.title("Energy scan time simulation: " + str(τfwhm_fs) + " fs pulses, " + str(pressure_mbar) + " mbar Ar, mode HE11")
    bx.pcolormesh(time*1e15, z, Etime[:, 0, :], rasterized=True)
    bx.set_xlim(-15, 50)
    bx.set_yticks([], [])
    bx.set_xlabel("Time [fs]")
    cx = fig.add_subplot(gs[2])
    cx.plot(stats["free electron density"]/stats["neutral gas density"]*100, stats["z"], color=cm.viridis(0), linewidth=1.0)
    #plt.set_yticks([], [])
    cx.set_xlabel("Ionization fraction [%]")
    cx.set_yticks([], [])
    cx.set_ylim(0, 3)
    #cx.set_xlim(0, 4)
#     # plt.ylabel("Soliton order")
#     # x = np.linspace(0, energies[:idx_lim,0].max()*1e6, 200)
#     # straight_line_x = 0.64*x
#     # marker_colour = cm.viridis(0)
#     # plt.figure(dpi=250)
#     # plt.title("Energy scan energy simulation: " + str(τfwhm_fs) + " fs pulses, " + str(pressure_mbar) + " mbar Ar, mode HE11")
#     # plt.plot(energies[:idx_lim,0]*1e6, energies[:idx_lim,1]*1e6, "o", markersize=3, color=marker_colour, label="Output energy")
#     # plt.plot(x, straight_line_x, "--k", linewidth=1.0, label="Linear output")
#     # plt.xlabel("Input energy [μJ]")
#     # plt.ylabel("Output energy [μJ]")
#     # plt.legend()

def plot_zprop(filename, t_compression_ratio=16, ω_compression_ratio=1):
    z, ω, t, Eω, Et, modes, stats = load_sim_prop(filename)
    fig = plt.figure(dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.1, hspace=0.1, left=0.11, right=0.92, bottom=0.15, top=0.98) 
    plt.rcParams.update({"font.size": 9})
    ax = fig.add_subplot(gs[0])
    #plt.title("Propagation simulation: " + str(τfwhm_fs) + " fs pulses, " + str(pressure_mbar) + " mbar Ar, mode HE1" + str(mode+1))
    #plt.pcolormesh(wl, Nsim, Espec[:idx_lim,mode,:], vmin=-30)
    ax.pcolormesh(ω[::ω_compression_ratio]*1e-15, z, 10*np.log10(abs(Eω[:, 0, ::ω_compression_ratio])**2/abs(Eω[:, 0, ::ω_compression_ratio].max())**2), vmin=-30, rasterized=True)
    ax.set_xlim(0.1, 6)
    ax.set_xlabel("Angular frequency (rad/fs)")
    #plt.ylabel("Soliton order")
    ax.set_ylabel("Propagation distance (m)")
    ax.axvline(x=chrng(λzd(ω, β2, "Ar", 467e-3, core_radius, T)*1e-9)*1e-15, color="white", linestyle="--")
    bx = fig.add_subplot(gs[1])
    # plt.title("Energy scan time simulation: " + str(τfwhm_fs) + " fs pulses, " + str(pressure_mbar) + " mbar Ar, mode HE11")
    bx.pcolormesh(t[::t_compression_ratio]*1e15, z, abs(Et[:, 0, ::t_compression_ratio])**2, vmin=-30, rasterized=True)
    bx.set_xlim(-70, 150)
    bx.set_xticks([-50, 0, 50, 100, 150])
    bx.set_yticks([], [])
    bx.set_xlabel("Time (fs)")

def plot_escan(t_compression_ratio=16, ω_compression_ratio=1):
    sim_filenames, N, _ = browse_sims()
    ω, t, Eω, Et, energy, pressure, τfwhm, modes = load_sim(sim_filenames[0])
    time = t[::t_compression_ratio]
    afreq = ω[::ω_compression_ratio]
    Etime = np.zeros((N, modes, len(t)//t_compression_ratio))
    Espec = np.zeros((N, modes, len(t)//(2*ω_compression_ratio) + 1))
    Espec_full = np.zeros((N, len(t)//(2*ω_compression_ratio) + 1))
    energies = np.zeros((N, len(energy)))
    wl = chrng(afreq)*1e9
    wl[0] = 6e5
    modes_to_plot = int(input("How many modes do you want to plot? "))
    Elim = float(input("Up to what input energy do you want to plot? "))
    for idx, file in enumerate(sim_filenames):
        _, _, Eω, Et, energy, _, _, _ = load_sim(file)
        for mode in range(0, modes_to_plot):
            energies[idx] = energy
            Etime[idx,mode,:] = Et[-1,mode,::t_compression_ratio]/Et[-1,mode,::t_compression_ratio].max()
            Espec[idx,mode,:] = Eω[-1,mode,::ω_compression_ratio]/Eω[-1,0,::ω_compression_ratio].max()
        Espec_full[idx,:] = 10*np.log10(np.sum(Espec[idx], axis=0))
    Espec = 10*np.log10(Espec)
    idx_lim = np.abs(energies[:,0] - Elim).argmin()
    Nsim = Nsol(gas, pressure, energies[:idx_lim,0], τfwhm, 800e-9, core_radius, T)
    for mode in range(0, modes+1):
        if mode < modes:
            if mode == 0:
                plt.figure(dpi=200)
                plt.title("Energy scan spectrum simulation: " + str(τfwhm_fs) + " fs pulses, " + str(pressure_mbar) + " mbar Ar, mode HE1" + str(mode+1))
                #plt.pcolormesh(wl, Nsim, Espec[:idx_lim,mode,:], vmin=-30)
                plt.pcolormesh(wl, energies[:idx_lim,0]*1e6, Espec[:idx_lim,mode,:], vmin=-30)
                plt.xlim(120, 1600)
                plt.xlabel("Wavelength [nm]")
                #plt.ylabel("Soliton order")
                plt.ylabel("Input energy [μJ]")
                plt.axvline(x=λzd(afreq, β2, gas, pressure, core_radius, T), color="white", linestyle="--")
            # else:
            #     plt.figure(dpi=250)
            #     plt.title("Energy scan spectrum simulation: " + str(τfwhm_fs) + " fs pulses, " + str(pressure_mbar) + " mbar Ar, mode HE1" + str(mode+1))
            #     plt.pcolormesh(wl, energies[:,0]*1e6, Espec[:,mode,:], vmin=-30)
            #     plt.xlim(200, 1600)
            #     plt.xlabel("Wavelength [nm]")
            #     plt.ylabel("Soliton order")
        else:
            plt.figure(dpi=200)
            plt.title("Energy scan spectrum simulation: " + str(τfwhm_fs) + " fs pulses, " + str(pressure_mbar) + " mbar Ar, " + str(modes_to_plot) + " modes")
            plt.pcolormesh(wl, energies[:idx_lim,0]*1e6, Espec_full[:idx_lim,:], vmin=-30)
            # plt.pcolormesh(wl, Nsim, Espec_full[:idx_lim,:], vmin=-30)
            plt.xlim(120, 1600)
            plt.xlabel("Wavelength [nm]")
            # plt.ylabel("Soliton order")
            plt.ylabel("Input energy [μJ]")
    plt.figure(dpi=200)
    plt.title("Energy scan time simulation: " + str(τfwhm_fs) + " fs pulses, " + str(pressure_mbar) + " mbar Ar, mode HE11")
    # plt.pcolormesh(time*1e15, Nsim, Etime[:idx_lim,0,:])
    plt.pcolormesh(time*1e15, energies[:idx_lim,0]*1e6, Etime[:idx_lim,0,:])
    plt.xlim(-25, 50)
    plt.xlabel("Time [fs]")
    # plt.ylabel("Soliton order")
    plt.ylabel("Input energy [μJ]")
    x = np.linspace(0, energies[:idx_lim,0].max()*1e6, 200)
    straight_line_x = 0.64*x
    marker_colour = cm.viridis(0)
    plt.figure(dpi=200)
    plt.title("Energy scan energy simulation: " + str(τfwhm_fs) + " fs pulses, " + str(pressure_mbar) + " mbar Ar, mode HE11")
    plt.plot(energies[:idx_lim,0]*1e6, energies[:idx_lim,1]*1e6, "o", markersize=3, color=marker_colour, label="Output energy")
    plt.plot(x, straight_line_x, "--k", linewidth=1.0, label="Linear output")
    plt.xlabel("Input energy [μJ]")
    plt.ylabel("Output energy [μJ]")
    plt.legend()

def plot_1spec(filename):
    # sim_filenames, N, energies = browse_sims()
    # Eplot = float(input("What input energy do you want to plot? "))
    # idx_plot = np.abs(np.asarray(energies) - Eplot).argmin()
    # ω, t, Eω, Et, _, pressure, _, _ = load_sim(sim_filenames[idx_plot])
    ω, t, Eω, Et, _, pressure, _, _ = load_sim(filename)
    wl = chrng(ω)*1e9
    wl[0] = 6e5
    plt.figure(dpi=200)
    colour = cm.viridis(0)
    plt.plot(ω*1e-15, Eω[-1, 0, :]/Eω[-1, 0, :].max(0), color=colour, label=str(int(Eplot*1e6))+ " μJ")
    plt.axvline(x=chrng(λzd(ω, β2, gas, pressure, core_radius, T)*1e-9)*1e-15, color="black", linestyle="--")
    plt.xlim(0, 10)
    plt.xlabel("Angular frequency [PHz]")
    plt.ylim(0, 1.1)
    plt.ylabel("SPD [a.u.]")
    plt.legend()
    plt.figure(dpi=200)
    plt.plot(wl, (Eω[-1, 0, :]*ω**2/c)/(Eω[-1, 0, :]*ω**2/c).max(0), color=colour, label=str(int(Eplot*1e6))+ " μJ")
    plt.axvline(x=λzd(ω, β2, gas, pressure, core_radius, T), color="black", linestyle="--")
    plt.xlim(120, 2500)
    plt.xlabel("Wavelength [nm]")
    plt.ylim(0, 1.1)
    plt.ylabel("SPD [a.u.]")
    plt.legend()

def plot_filtered2(filename, λ0IR, ΔλIR, λ0VIS, ΔλVIS, supergauss_power, plot_filtered=True):
    def gauss_filter(x, x0, σ, supergauss_power):
        return sp.exp(-0.5*((x-x0)/σ)**supergauss_power)
    def chΔλ(Δλ, λ0):
        return (c/λ0**2)*Δλ
    z, ω, t, Eω, Et, _, _ = load_sim_prop(filename)
    # stats: "pressure", "output energy", "input energy", "free electron density", "z", "neutral gas density"
    # # wl = chrng(ω)*1e9
    # # wl[0] = 6e5
    Nt = len(t)
    T = Nt//2-1
    Nz = len(z)
    Nw = len(ω)
    gaussIR = np.zeros(Nt)
    gaussIR[T:] = gauss_filter(ω, chrng(λ0IR), chΔλ(ΔλIR, λ0IR), supergauss_power)
    gaussVIS = np.zeros(Nt)
    gaussVIS[T:] = gauss_filter(ω, chrng(λ0VIS), chΔλ(ΔλVIS, λ0VIS), supergauss_power)
    filteredEωIR = np.zeros((Nz, Nt), dtype=np.cdouble)
    filteredEωIRprop = np.zeros((Nz, Nw))
    filteredEtIR = np.zeros((Nz, Nt))
    ###### here this array if of real values only, because otherwise the printing complains; if I need phase I need to change it
    filteredEωVIS = np.zeros((Nz, Nt), dtype=np.cdouble)
    filteredEωVISprop = np.zeros((Nz, Nw))
    filteredEtVIS = np.zeros((Nz, Nt))
    for step in range(0, len(z)):
        filteredEωIR[step, T:] = gaussIR[T:]*Eω[step, 0, :]
        filteredEωIRprop[step, :] = abs(filteredEωIR[step, T:]/filteredEωIR[0, T:].max())**2
        filteredEtIR[step, :] = abs(2.0*sp.ifft(filteredEωIR[step, :]))**2
        filteredEωVIS[step, T:] = gaussVIS[T:]*Eω[step, 0, :]
        filteredEωVISprop[step, :] = abs(filteredEωVIS[step, T:]/filteredEωVIS[0, T:].max())**2
        filteredEtVIS[step, :] = abs(2.0*sp.ifft(filteredEωVIS[step, :]))**2
    colour = cm.viridis(0)
    colourIR = cm.viridis(100)
    colourVIS = cm.viridis(200)
    colourUV = cm.viridis(300)
    if plot_filtered == True:
        filteredEωIR = gaussIR[T:]*(Eω[-1, 0, :]/Eω[-1, 0, :].max())
        filteredEωVIS = gaussVIS[T:]*(Eω[-1, 0, :]/Eω[-1, 0, :].max())
        plt.figure(dpi=200)
        plt.plot(ω*1e-15, abs((Eω[-1, 0, :]))**2/(abs(Eω[-1, 0, :])**2).max(), color=colour, label="Full time")
        #plt.plot(ω*1e-15, abs(filteredEωIR)**2, color=colourIR, linestyle="--", label="Filtered IR")
        #plt.plot(ω*1e-15, abs(filteredEωVIS)**2, color=colourVIS, linestyle="--", label="Filtered VIS")
        plt.xlabel("Ang. frequency (rad/fs)")
        plt.ylabel("SPD (a.u.)")
        plt.axvline(x=2.35, color="black", linestyle="--")
        plt.xlim(0.1, 7)
    else:
        plt.figure(dpi=200)
        plt.plot(t*1e15, 1e-9*abs(Et[-1, 0, :])**2, color=colour, label="Full time")
        plt.plot(t*1e15, filteredEtIR[-1, :]*1e-9, color=colourIR, label="Filtered IR")
        plt.plot(t*1e15, filteredEtVIS[-1, :]*1e-9, color=colourVIS, label="Filtered VIS")
        plt.xlim(-30, 50)
        plt.xlabel("Time (fs)")
        plt.ylabel("Power (GW)")
        plt.legend()
    plt.show()

def plot_filtered(filename, λ0IR, ΔλIR, λ0VIS, ΔλVIS, λ0UV, ΔλUV, supergauss_power, plot_filtered=True):
    def gauss_filter(x, x0, σ, supergauss_power):
        return sp.exp(-0.5*((x-x0)/σ)**supergauss_power)
    def chΔλ(Δλ, λ0):
        return (c/λ0**2)*Δλ
    z, ω, t, Eω, Et, _, _ = load_sim_prop(filename)
    # stats: "pressure", "output energy", "input energy", "free electron density", "z", "neutral gas density"
    # # wl = chrng(ω)*1e9
    # # wl[0] = 6e5
    Nt = len(t)
    T = Nt//2-1
    Nz = len(z)
    Nw = len(ω)
    gaussIR = np.zeros(Nt)
    gaussIR[T:] = gauss_filter(ω, chrng(λ0IR), chΔλ(ΔλIR, λ0IR), supergauss_power)
    gaussVIS = np.zeros(Nt)
    gaussVIS[T:] = gauss_filter(ω, chrng(λ0VIS), chΔλ(ΔλVIS, λ0VIS), supergauss_power)
    gaussUV = np.zeros(Nt)
    gaussUV[T:] = gauss_filter(ω, chrng(λ0UV), chΔλ(ΔλUV, λ0UV), supergauss_power)
    filteredEωIR = np.zeros((Nz, Nt), dtype=np.cdouble)
    filteredEωIRprop = np.zeros((Nz, Nw))
    filteredEtIR = np.zeros((Nz, Nt))
    ###### here this array if of real values only, because otherwise the printing complains; if I need phase I need to change it
    filteredEωVIS = np.zeros((Nz, Nt), dtype=np.cdouble)
    filteredEωVISprop = np.zeros((Nz, Nw))
    filteredEtVIS = np.zeros((Nz, Nt))
    filteredEωUV = np.zeros((Nz, Nt), dtype=np.cdouble)
    filteredEωUVprop = np.zeros((Nz, Nw))
    filteredEtUV = np.zeros((Nz, Nt))
    maxIR = np.zeros((Nz))
    maxUV = np.zeros((Nz))
    energyIR = np.zeros((Nz))
    energyUV = np.zeros((Nz))
    for step in range(0, len(z)):
        filteredEωIR[step, T:] = gaussIR[T:]*Eω[step, 0, :]
        filteredEωIRprop[step, :] = abs(filteredEωIR[step, T:]/filteredEωIR[0, T:].max())**2
        maxIRidx = np.where(filteredEωIR[step, T:] == filteredEωIR[step, T:].max())
        maxIR[step] = ω[maxIRidx[0]]
        filteredEtIR[step, :] = abs(sp.ifft(filteredEωIR[step, :]))**2
        energyIR[step] = np.trapz(filteredEtIR[step, :], t)
        filteredEωVIS[step, T:] = gaussVIS[T:]*Eω[step, 0, :]
        filteredEωVISprop[step, :] = abs(filteredEωVIS[step, T:]/filteredEωVIS[0, T:].max())**2
        filteredEtVIS[step, :] = abs(2.0*sp.ifft(filteredEωVIS[step, :]))**2
        filteredEωUV[step, T:] = gaussUV[T:]*Eω[step, 0, :]
        filteredEωUVprop[step, :] = abs(filteredEωUV[step, T:]/filteredEωIR[0, T:].max())**2
        maxUVidx = np.where(filteredEωUV[step, T:] == filteredEωUV[step, T:].max())
        maxUV[step] = ω[maxUVidx[0]]
        filteredEtUV[step, :] = abs(2.0*sp.ifft(filteredEωUV[step, :]))**2
        energyUV[step] = np.trapz(filteredEtUV[step, :], t)
    colour = cm.viridis(0)
    colourIR = cm.viridis(100)
    colourVIS = cm.viridis(200)
    colourUV = cm.viridis(300)
    if plot_filtered == True:
        filteredEωIR = gaussIR[T:]*(Eω[-1, 0, :]/Eω[-1, 0, :].max())
        filteredEωVIS = gaussVIS[T:]*(Eω[-1, 0, :]/Eω[-1, 0, :].max())
        filteredEωUV = gaussUV[T:]*(Eω[-1, 0, :]/Eω[-1, 0, :].max())
        plt.figure(dpi=200)
        plt.plot(ω*1e-15, Eω[-1, 0, :]/Eω[-1, 0, :].max(), color=colour, label="Full time")
        plt.plot(ω*1e-15, filteredEωIR, color=colourIR, linestyle="--", label="Filtered IR")
        plt.plot(ω*1e-15, filteredEωVIS, color=colourVIS, linestyle="--", label="Filtered VIS")
        plt.plot(ω*1e-15, filteredEωUV, color=colourUV, linestyle="--", label="Filtered UV")
        plt.axvline(x=λzd(ω, β2, "Ar", 467e-3, core_radius, T), color="black", linestyle="--")
        plt.xlim(0.1, 8)
        plt.ylim(0, 1.1)
        plt.legend()
    else:
        plt.figure(dpi=200)
        plt.plot(chrng(ω)*1e9, Eω[-1, 0, :]/Eω[-1, 0, :].max(), color=colour)
        plt.axvline(x=λzd(ω, β2, "Ar", 467e-3, core_radius, T), color="black", linestyle="--")
        # plt.figure(dpi=200)
        # plt.subplot(1,2,1)
        # plt.plot(z, chrng(maxIR)*1e9, "ro", label="IR")
        # plt.plot(z, chrng(maxUV)*1e9, "ko", label="UV")
        # plt.legend()
        # plt.subplot(1,2,2)
        # plt.plot(z, energyIR*1e6, "ro", label="IR")
        # plt.plot(z, energyUV*1e6, "ko", label="UV")
        # plt.legend()
        # plt.figure(dpi=200)
        # plt.subplot(2,3,1)
        # plt.title("Filtered IR - spectrum")
        # filteredEωIRproplog = 10*np.log10(filteredEωIRprop)
        # filteredEωIRproplog[np.isinf(filteredEωIRproplog)] = -30
        # plt.pcolormesh(ω*1e-15, z, filteredEωIRproplog, vmin=-30, rasterized=True)
        # plt.xlim(0.1, 7)
        # plt.xlabel("Ang. frequency (rad/fs)")
        # plt.ylabel("Propagation distance (m)")
        # #plt.figure(dpi=200)
        # plt.subplot(2,3,4)
        # plt.title("Filtered IR - time")
        # plt.pcolormesh(t*1e15, z, filteredEtIR, rasterized=True)
        # plt.xlim(-25, 150)
        # plt.xlabel("Time (fs)")
        # plt.ylabel("Propagation distance (m)")
        # #plt.figure(dpi=200)
        # plt.subplot(2,3,2)
        # plt.title("Filtered VIS - spectrum")
        # filteredEωVISproplog = 10*np.log10(filteredEωVISprop)
        # filteredEωVISproplog[np.isinf(filteredEωVISproplog)] = -30
        # plt.pcolormesh(ω*1e-15, z, filteredEωVISproplog, vmin=-30, rasterized=True)
        # plt.xlim(0.1, 7)
        # plt.xlabel("Ang. frequency (rad/fs)")
        # plt.ylabel("Propagation distance (m)")
        # #plt.figure(dpi=200)
        # plt.subplot(2,3,5)
        # plt.title("Filtered VIS - time")
        # plt.pcolormesh(t*1e15, z, filteredEtVIS, rasterized=True)
        # plt.xlim(-25, 150)
        # plt.xlabel("Time [fs]")
        # plt.ylabel("Propagation distance (m)")
        # #plt.figure(dpi=200)
        # plt.subplot(2,3,3)
        # plt.title("Filtered UV - spectrum")
        # filteredEωUVproplog = 10*np.log10(filteredEωUVprop)
        # filteredEωUVproplog[np.isinf(filteredEωUVproplog)] = -30
        # plt.pcolormesh(ω*1e-15, z, filteredEωUVproplog, vmin=-30, vmax=0, rasterized=True)
        # plt.xlim(0.1, 7)
        # plt.xlabel("Ang. frequency (rad/fs)")
        # plt.ylabel("Propagation distance (m)")
        # plt.colorbar()
        # #plt.figure(dpi=200)
        # plt.subplot(2,3,6)
        # plt.title("Filtered UV - time")
        # plt.pcolormesh(t*1e15, z, filteredEtUV, rasterized=True)
        # plt.xlim(-25, 150)
        # plt.xlabel("Time (fs)")
        # plt.ylabel("Propagation distance (m)")
        # plt.colorbar()
    plt.show()

#plot_1spec()
#plot_filtered("Ar467mbar10fs170uJ.h5", 1250e-9, 5000e-9, 550e-9, 1100e-9, 200e-9, 650e-9, 10, False)
plot_filtered2("Ar467mbar_i0000000017_E=3.50854271e-05_0000.h5", 1320e-9, 5000e-9, 570e-9, 1100e-9, 30, True)
#plot_escan(8, 4)
#plot_zprop("collision.h5", 1, 4)

plt.show()