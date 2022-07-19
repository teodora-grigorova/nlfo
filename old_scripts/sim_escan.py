import h5py
import os

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec

import numpy as np
import scipy as sp
from scipy.constants import pi, c

files = os.listdir()
sim_filenames = []
energies = []

for file in files:
    if file[:2] == "Ar":
        energy_value = float((file.split("="))[1].split("e")[0])
        energy_exponent = int((((file.split("="))[1].split("e")[1]).split("_")[0]).split("0")[1])
        energy = energy_value*10**(-float(energy_exponent))
        sim_filenames.append(file)
        energies.append(energy)

def chrng(x):
    return (2*pi*c)/x

def field_envelope(E):
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
    E = np.array(sim["E"]) # This E is field: to get the envelope need to find the analytical signal?
    steps = E.shape[0]
    Et = np.zeros((steps, len(t)))
    for step in range(0, steps):
        Et[step,:] = abs(field_envelope(E[step][0][:]))**2
    Eω = np.zeros((steps, ω_pivot))
    for step in range(0, steps):
        Eω[step,:] = sp.real(abs(sp.fft(E[step][0][:])[:ω_pivot])**2)
    # if E.shape[1] == 1:
    #     for step in range(0, steps):
    #         Et[step,:] = abs(field_envelope(E[step][0][:]))**2
    # else:
    #     print("The middle index of E field is not 1.")
    # Eω = np.zeros((steps, ω_pivot))
    # if E.shape[1] == 1:
    #     for step in range(0, steps):
    #         Eω[step,:] = sp.real(abs(sp.fft(E[step][0][:])[:ω_pivot])**2)
    # else:
    #     print("The middle index of E field is not 1.")
    energy = np.array(sim["stats_energy_m"])
    return z, ω, t, Eω, Et, energy 

N = len(sim_filenames)
z, ω, t, Eω, Et, energy = load_sim(sim_filenames[0])
Etime = np.zeros((N, len(t)))
# Espec = np.zeros((N, len(t)//2 + 1))
time = t
# afreq = ω
# out_energies = np.zeros(N)
# wl = chrng(afreq)*1e9
# wl[0] = 6e5

nn = 0
for file in sim_filenames:
    _, _, t, _, Et, _ = load_sim(file)
    # _, ω, _, Eω, _, _ = load_sim(file)
    # _, _, _, _, _, energy = load_sim(file)
    # out_energies[nn] = energy[-1]
    Etime[nn,:] = Et[-1,:]/Et[-1,:].max()
    # Espec[nn,:] = Eω[-1,:]/Eω[-1,:].max()
    nn += 1
# Espec = 10*np.log10(Espec)

# z, ω, t, Eω, Et = load_sim(args.filename)

# plt.plot(energies, out_energies*1e6, "ok")
# plt.xlabel("Input energy [μJ]")
# plt.ylabel("Output energy [μJ]")
# plt.xlim(0, 560)
# plt.ylim(0, 300)

fig = plt.figure(dpi=250)
plt.rcParams.update({"font.size": 6.8})
plt.suptitle("Energy scan simulation: 10 fs pulses, 143 mbar Ar")

# plt.pcolormesh(wl, energies, Espec, vmin=-30)
# plt.xlim(200, 1600)
# plt.xlabel("Wavelength [nm]")
# plt.ylabel("Pulse energy [μJ]")
# # #plt.yticks([1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550])
# plt.axvline(x=600, color="white", linestyle="--")

# plt.pcolormesh(time*1e15, energies, Etime)
# plt.xlim(-100, 100)
# plt.xlabel("Time [fs]")
# plt.ylabel("Pulse energy [μJ]")
# plt.yticks([1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550])
# plt.axvline(x=400, color="white", linestyle="--")

plt.show()