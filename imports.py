import h5py
import pickle
import os

import numpy as np
import scipy as sp
import scipy.fftpack as fft
from scipy.constants import pi, c, mu_0
import scipy.integrate as integrate
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline   
    
import nlfo_math
import nlfo_functions
import plotting



def load_escan(escan_filepath, scan_type, coupling=0.2, ratio_int_time=1.0, core_radius=125e-6, fibre_length=3.0, gas="Ar"):

    if scan_type == "vuv":
        escan_data = h5py.File(escan_filepath, "r")
        wavelengths = np.array(escan_data.get("wavelengths"))
        angles = np.array(escan_data.get("hwp_angles"))
        escan =  np.array(escan_data.get("spectra"))

        data = {}
        data["λ"] = wavelengths
        data["angles"] = angles
        data["Iλ"] = escan
        #data["energies"] = np.asarray([coupled_energies, out_energies])
        #data["stats"] = {"pressure": pressure, "gas": gas, "core radius": core_radius, "length": fibre_length, "ratio integration time": ratio_int_time, "coupling": coupling}

        return data

    elif scan_type == "vis":
        escan_data = h5py.File(escan_filepath, "r")
        wavelengths = np.array(escan_data.get("wavelengths"))
        angles = np.array(escan_data.get("hwp_angles"))
        escan =  np.array(escan_data.get("spectra"))

        data = {}
        data["λ"] = wavelengths
        data["angles"] = angles
        data["Iλ"] = escan
        #data["energies"] = np.asarray([coupled_energies, out_energies])
        #data["stats"] = {"pressure": pressure, "gas": gas, "core radius": core_radius, "length": fibre_length, "ratio integration time": ratio_int_time, "coupling": coupling}

        return data

    elif scan_type == "vis+ir":

        with open(escan_filepath, "rb") as file:
            escan = pickle.load(file, encoding="latin1")

        wls_vis = np.asarray(escan["wavelength"][11:])
        wls_ir = np.asarray(escan["IR wavelengths"])
        spec_vis = np.asarray(escan["spectra"]) 
        spec_ir = np.asarray(escan["IR spectra"])

        spec_vis_calib = np.asarray([plotting.calib(wls_vis, spec_vis[spec_idx, :], calibration="vis") for spec_idx in range(0, spec_vis.shape[0])])
        spec_ir_calib = np.asarray([plotting.calib(wls_ir, spec_ir[spec_idx, :], calibration="nir") for spec_idx in range(0, spec_ir.shape[0])])

        overlap_vis = wls_vis[wls_vis>wls_ir[0]] 
        length_full_spec = (wls_vis.shape[0]-overlap_vis.shape[0]) + wls_ir.shape[0]
        wls_full = np.zeros(length_full_spec)
        wls_full[0:(wls_vis.shape[0]-overlap_vis.shape[0])] = wls_vis[0:(wls_vis.shape[0]-overlap_vis.shape[0])]
        wls_full[(wls_vis.shape[0]-overlap_vis.shape[0]):] = wls_ir

        full_spec = np.zeros((spec_vis_calib.shape[0], length_full_spec))
        full_spec[:,0:(wls_vis.shape[0]-overlap_vis.shape[0])] = spec_vis_calib[:,0:(wls_vis.shape[0]-overlap_vis.shape[0])]
        full_spec[:,(wls_vis.shape[0]-overlap_vis.shape[0]):] = spec_ir_calib[:,]*ratio_int_time

        pressure = float(escan_filepath.split("escan_")[1].split("mbar")[0])*1e-3
        p_fit = np.asarray(escan["fit parameters"])
        angles = np.asarray(escan["angles"])
        in_energies = nlfo_math.malus(p_fit, angles)
        coupled_energies = nlfo_math.est_coupled_energy(in_energies, coupling, core_radius=core_radius, L=fibre_length)*1e-3
        out_energies = np.asarray(escan["output powers"])*1e-3
        # often the first value of the output energies data is wrong, e.g. 1000 uJ, so discard it:
        out_energies[0] = 1e-6

        data = {}
        data["λ"] = wls_full*1e-9
        data["Iλ"] = full_spec
        data["energies"] = np.asarray([coupled_energies, out_energies])
        data["stats"] = {"pressure": pressure, "gas": gas, "core radius": core_radius, "length": fibre_length, "ratio integration time": ratio_int_time, "coupling": coupling}

        return data

    else:
        raise AttributeError("Scan type should be given: 'vuv', 'vis', or 'vis+ir'.")


def browse_sims(common_filename="Ar"):
    files = os.listdir()

    sim_filenames = []

    idx = len(common_filename)

    for file in files:
        if file[:idx] == common_filename:
            sim_filenames.append(file)

    N = len(sim_filenames)

    return sim_filenames, N


def load_fnfep_sim(filename):

    sim = h5py.File(filename, "r")

    z = np.array(sim["Z"])
    zsteps = len(z)
    ω = np.array(sim["W"])
    ω_pivot = len(ω)
    t = np.array(sim["T"])
    E = np.array(sim["E"])

    energy_in = np.array(sim["input_energies"])[0]
    energy_out = np.array(sim["stats_energy_m"])
    # energy_out = np.array(sim["stats_energy_m"])[-1,0]
    energies = np.array([energy_in, energy_out])
    pressure = np.array(sim["stats_pressure"])
    # pressure = sim["gas_pressure"][()]
    T = sim["gas_temperature"][()]
    core_radius = sim["fibre_radius"][()]
    L = sim["fibre_length"][()]
    τfwhm = np.array(sim["input_fwhms"])[0]
    gas = sim["gas"][()].decode("utf-8")
    electron_density = np.array(sim["ne"])[:, -1]
    # electron_density = sim["ne"][()]
    gas_density = nlfo_functions.gas_density(gas, pressure)
    z_stats = np.array(sim["stats_z"])

    sim.close()

    modes = E.shape[1]
    
    pressure_interpolation_curve = InterpolatedUnivariateSpline(z_stats, pressure)
    pressure = pressure_interpolation_curve(z)

    gas_density = nlfo_functions.gas_density("He", pressure)
    gas_density[np.isinf(gas_density)] = 0.0

    ionisation_fraction = electron_density/gas_density[:-1]
    
    Et = np.zeros((zsteps, modes, len(t)), dtype=complex)
    It = np.zeros((zsteps, modes, len(t)))

    for step in range(0, zsteps):
        for mode in range(0, modes):
            Et[step, mode, :] = nlfo_functions.field_envelope(E[step][mode][:])
            It[step, mode, :] = abs(nlfo_functions.field_envelope(E[step][mode][:]))**2

    Eω = np.zeros((zsteps, modes, ω_pivot), dtype=complex)
    Iω = np.zeros((zsteps, modes, ω_pivot))

    for step in range(0, zsteps):
        for mode in range(0, modes):
            Eω[step, mode, :] = sp.fft(E[step][mode][:])[:ω_pivot]
            Iω[step, mode, :] = np.real(abs(sp.fft(E[step][mode][:])[:ω_pivot])**2)

    ω[ω==0] = 1e12

    simdata = {}

    simdata["z"] = z
    simdata["ω"] = ω
    simdata["t"] = t
    simdata["Eω_modal"] = Eω
    simdata["Iω_modal"] = Iω
    simdata["Iω"] = np.sum(Iω, axis=1)
    simdata["Et_modal"] = Et
    simdata["It_modal"] = It
    simdata["It"] = np.sum(It, axis=1)
    simdata["energies"] = energies
    simdata["stats"] = {"pressure": pressure, "temprerature": T, "gas": gas, "core radius": core_radius, "fibre length": L, "pulse fwhm duration": τfwhm, "number of modes": modes, "electron density": electron_density, "gas density": gas_density, "ionisation fraction": ionisation_fraction}

    return simdata


def load_luna_sim(filename, stats=False):

    sim = h5py.File(filename, "r")

    Eω = sim["Eω"][:]
    z = sim["z"][:]
    ω = sim["grid"]["ω"][:]
    t = sim["grid"]["t"][:]

    if stats == True:
        pressure = sim["stats"]["pressure"][()]
        energy = sim["stats"]["energy"][()]
        density = sim["stats"]["density"][()]
        electron_density = sim["stats"]["electrondensity"][()]
        zdw = sim["stats"]["zdw"][()]
        z_prop = sim["stats"]["z"][()]

    L = sim["grid"]["zmax"][()]

    sim.close()

    modes = Eω.shape[1]
    zsteps = len(z)
    ω[ω == 0.0] = 1e12
    Et = np.zeros((zsteps, modes, len(t)), dtype=complex)
    It = np.zeros((zsteps, modes, len(t)))

    for step in range(0, zsteps):
        for mode in range(0, modes):
            Et[step, mode, :] = nlfo_functions.analytical_signal(t, Eω[step, mode, :])
    It = abs(Et)**2

    Espec = np.zeros((zsteps, modes, len(ω)), dtype=complex)
    Iω = np.zeros((zsteps, modes, len(ω)))

    engs = np.zeros((zsteps, modes))

    for step in range(0, zsteps):
        engs[step, :] = [integrate.simps(It[step, mode, :], t) for mode in range(0, modes)]
        for mode in range(0, modes):
            Espec[step, mode, :] = Eω[step, mode, :]
    Iω = abs(Espec)**2

    energies = engs

    # energy_in = [integrate.simps(It[0, mode, :], t) for mode in range(0, modes)]
    # energy_out = [integrate.simps(It[-1, mode, :], t) for mode in range(0, modes)]
    # energies = np.array([energy_in, energy_out])

    simdata = {}

    simdata["z"] = z
    simdata["ω"] = ω
    simdata["t"] = t
    simdata["Eω_modal"] = Espec
    simdata["Iω_modal"] = Iω
    simdata["Iω"] = np.sum(Iω, axis=1)
    simdata["Et_modal"] = Et
    simdata["It_modal"] = It
    simdata["It"] = np.sum(It, axis=1)
    simdata["energies"] = energies

    if stats==True:
        simdata["stats"] = {"fibre length": L, "number of modes": modes, "pressure": pressure, "energy": energy, "density": density, "electron_density": electron_density, "zdw": zdw, "z_propagation": z_prop}
    else:
        simdata["stats"] = {"fibre length": L, "number of modes": modes}

    return simdata


def load_luna_freespacesim(filename, stats=False):

    sim = h5py.File(filename, "r")

    Eω = sim["Eω"][:]
    z = sim["z"][:]
    ω = sim["grid"]["ω"][:]
    t = sim["grid"]["t"][:]

    if stats == True:
        pressure = sim["stats"]["pressure"][()]
        energy = sim["stats"]["energy"][()]
        density = sim["stats"]["density"][()]
        electron_density = sim["stats"]["electrondensity"][()]
        zdw = sim["stats"]["zdw"][()]
        z_prop = sim["stats"]["z"][()]

    L = sim["grid"]["zmax"][()]

    sim.close()

    rs = Eω.shape[1]
    zsteps = len(z)
    ω[ω == 0.0] = 1e12
    Et = np.zeros((zsteps, rs, len(t)), dtype=complex)
    It = np.zeros((zsteps, rs, len(t)))

    for step in range(0, zsteps):
        for r in range(0, rs):
            Et[step, r, :] = nlfo_functions.analytical_signal(t, Eω[step, r, :])
    It = abs(Et)**2

    Espec = np.zeros((zsteps, rs, len(ω)), dtype=complex)
    Iω = np.zeros((zsteps, rs, len(ω)))

    for step in range(0, zsteps):
        for r in range(0, rs):
            Espec[step, r, :] = Eω[step, r, :]
    Iω = abs(Espec)**2

    energy_in = [integrate.simps(It[0, r, :], t) for r in range(0, rs)]
    energy_out = [integrate.simps(It[-1, r, :], t) for r in range(0, rs)]
    energies = np.array([energy_in, energy_out])

    simdata = {}

    simdata["z"] = z
    simdata["ω"] = ω
    simdata["t"] = t
    simdata["Eω_modal"] = Espec
    simdata["Iω_modal"] = Iω
    simdata["Iω"] = np.sum(Iω, axis=1)
    simdata["Et_modal"] = Et
    simdata["It_modal"] = It
    simdata["It"] = np.sum(It, axis=1)
    simdata["energies"] = energies

    if stats==True:
        simdata["stats"] = {"fibre length": L, "r samples": rs, "pressure": pressure, "energy": energy, "density": density, "electron_density": electron_density, "zdw": zdw, "z_propagation": z_prop}
    else:
        simdata["stats"] = {"fibre length": L, "r samples": rs}

    return simdata


def load_fnfep_prop(filename, t_lims, λ_lims):

    simdata = load_fnfep_sim(filename)

    z = simdata["z"]
    ω = simdata["ω"]
    t = simdata["t"]
    Iω = simdata["Iω"]
    It = simdata["It"]
    modes = simdata["stats"]["number of modes"]

    propdata = {}

    time_min = np.where(np.isclose(t, t_lims[0]*1e-15, atol=5e-16))[0][0]
    time_max = np.where(np.isclose(t, t_lims[1]*1e-15, atol=5e-16))[0][0]
    time = t[time_min:time_max]

    afreq_max = np.where(np.isclose(ω, nlfo_math.chrng(λ_lims[0]*1e-9), atol=1e13))[0][0]
    afreq_min = np.where(np.isclose(ω, nlfo_math.chrng(λ_lims[1]*1e-9), atol=5e12))[0][0]
    afreq = ω[afreq_min:afreq_max]

    steps = len(z)

    Itime = np.zeros((steps, modes, len(time)))
    Ispec = np.zeros((steps, modes, len(afreq)))
    Itime_full = np.zeros((steps, len(time)))
    Ispec_full = np.zeros((steps, len(afreq)))

    for mode in range(0, modes):
        Itime[:, mode, :] = It[:, mode, time_min:time_max]
        Ispec[:, mode, :] = ((afreq**2/(4*pi**2*c))*Iω[:, mode, afreq_min:afreq_max])/((afreq**2/(4*pi**2*c))*Iω[:, 0, afreq_min:afreq_max]).max()
    Itime_full = np.sum(Itime, axis=1)
    Ispec_full = np.sum(Ispec, axis=1)

    propdata["z"] = z
    propdata["λ"] = nlfo_math.chrng(afreq)*1e9
    propdata["t"] = time*1e15
    propdata["Iω"] = Ispec_full
    propdata["It"] = Itime_full
    propdata["Iω_modal"] = Ispec
    propdata["It_modal"] = Itime
    propdata["stats"] = simdata["stats"]


    return propdata


def load_fnfep_escan(path, t_lims, λ_lims):

    os.chdir(path)

    sim_filenames, N = browse_sims()

    simdata = load_fnfep_sim(sim_filenames[0])

    ω = simdata["ω"]
    t = simdata["t"]
    Iω = simdata["Iω"]
    It = simdata["It"]
    stats = simdata["stats"]
    modes = stats["number of modes"]

    scandata = {}

    time_min = np.where(np.isclose(t, t_lims[0]*1e-15, atol=5e-16))[0][0]
    time_max = np.where(np.isclose(t, t_lims[1]*1e-15, atol=5e-16))[0][0]
    time = t[time_min:time_max]

    afreq_max = np.where(np.isclose(ω, nlfo_math.chrng(λ_lims[0]*1e-9), atol=1e13))[0][0]
    afreq_min = np.where(np.isclose(ω, nlfo_math.chrng(λ_lims[1]*1e-9), atol=5e12))[0][0]
    afreq = ω[afreq_min:afreq_max]

    Itime = np.zeros((N, modes, len(time)))
    Ispec = np.zeros((N, modes, len(afreq)))
    Itime_full = np.zeros((N, len(time)))
    Ispec_full = np.zeros((N, len(afreq)))

    energies = np.zeros((N,2))

    for idx, file in enumerate(sim_filenames):
        simdata = load_fnfep_sim(file)
        for mode in range(0, modes):
            energies[idx] = simdata["energies"]
            Itime[idx,mode,:] = simdata["It"][-1, mode, time_min:time_max]
            Ispec[idx,mode,:] = ((afreq**2/(4*pi**2*c))*simdata["Iω"][-1, mode, afreq_min:afreq_max])/((afreq**2/(4*pi**2*c))*simdata["Iω"][-1, 0, afreq_min:afreq_max]).max()
        Itime_full[idx,:] = np.sum(Itime[idx], axis=0)
        Ispec_full[idx,:] = np.sum(Ispec[idx], axis=0)

    scandata["ω"] = afreq
    scandata["t"] = time
    scandata["Iω"] = Ispec_full
    scandata["Iω_modal"] = Ispec
    scandata["It"] = Itime_full
    scandata["It_modal"] = Itime
    scandata["energies"] = energies
    scandata["stats"] = stats

    return scandata


def load_luna_escan(path, common_filename, t_lims, λ_lims):

    os.chdir(path)

    sim_filenames, N = browse_sims(common_filename=common_filename)

    simdata = load_luna_sim(sim_filenames[0])

    ω = simdata["ω"]
    t = simdata["t"]
    modes = simdata["stats"]["number of modes"]

    scandata = {}

    time_min = np.where(np.isclose(t, t_lims[0]*1e-15, atol=5e-16))[0][0]
    time_max = np.where(np.isclose(t, t_lims[1]*1e-15, atol=5e-16))[0][0]
    time = t[time_min:time_max]

    afreq_max = np.where(np.isclose(ω, nlfo_math.chrng(λ_lims[0]*1e-9), atol=1e13))[0][0]
    afreq_min = np.where(np.isclose(ω, nlfo_math.chrng(λ_lims[1]*1e-9), atol=5e12))[0][0]
    afreq = ω[afreq_min:afreq_max]

    Itime = np.zeros((N, modes, len(time)))
    Ispec = np.zeros((N, modes, len(afreq)))
    Itime_full = np.zeros((N, len(time)))
    Ispec_full = np.zeros((N, len(afreq)))

    energies = np.zeros((N, 2, modes))

    for idx, file in enumerate(sim_filenames):
        simdata = load_luna_sim(file)
        for mode in range(0, modes):
            energies[idx, 0, :] = simdata["energies"][0]
            energies[idx, 1, :] = simdata["energies"][-1]
            Itime[idx,mode,:] = simdata["It_modal"][-1, mode, time_min:time_max]
            Ispec[idx,mode,:] = ((afreq**2/(4*pi**2*c))*simdata["Iω_modal"][-1, mode, afreq_min:afreq_max])/((afreq**2/(4*pi**2*c))*simdata["Iω_modal"][-1, 0, afreq_min:afreq_max]).max()
        Itime_full[idx,:] = np.sum(Itime[idx], axis=0)
        Ispec_full[idx,:] = np.sum(Ispec[idx], axis=0)

    scandata["wls"] = nlfo_math.chrng(afreq)*1e9
    scandata["time"] = time*1e15
    scandata["Iω_modal"] = Ispec
    scandata["Iω"] = Ispec_full
    scandata["It_modal"] = Itime
    scandata["It"] = Itime_full
    scandata["energies"] = energies

    return scandata


def load_luna_freespaceescan(path, common_filename, t_lims, λ_lims):

    os.chdir(path)

    sim_filenames, N = browse_sims(common_filename=common_filename)

    simdata = load_luna_freespacesim(sim_filenames[0])

    ω = simdata["ω"]
    t = simdata["t"]
    rs = simdata["stats"]["r samples"]

    scandata = {}

    time_min = np.where(np.isclose(t, t_lims[0]*1e-15, atol=5e-16))[0][0]
    time_max = np.where(np.isclose(t, t_lims[1]*1e-15, atol=5e-16))[0][0]
    time = t[time_min:time_max]

    afreq_max = np.where(np.isclose(ω, nlfo_math.chrng(λ_lims[0]*1e-9), atol=1e13))[0][0]
    afreq_min = np.where(np.isclose(ω, nlfo_math.chrng(λ_lims[1]*1e-9), atol=5e12))[0][0]
    afreq = ω[afreq_min:afreq_max]

    Itime = np.zeros((N, rs, len(time)))
    Ispec = np.zeros((N, rs, len(afreq)))
    Itime_full = np.zeros((N, len(time)))
    Ispec_full = np.zeros((N, len(afreq)))
    Ispec_z = np.zeros((N, rs, len(afreq)))

    energies = np.zeros((N, 2, rs))

    for idx, file in enumerate(sim_filenames):
        simdata = load_luna_freespacesim(file)
        for r in range(0, rs):
            energies[idx,:] = simdata["energies"]
            Itime[idx,r,:] = simdata["It_modal"][-1, r, time_min:time_max]
            Ispec[idx,r,:] = (afreq**2/(4*pi**2*c))*simdata["Iω_modal"][-1, r, afreq_min:afreq_max]
            Ispec_z[idx,r,:] = (afreq**2/(4*pi**2*c))*simdata["Iω_modal"][0, r, afreq_min:afreq_max]
        Itime_full[idx,:] = np.sum(Itime[idx], axis=0)
        Ispec_full[idx,:] = np.sum(Ispec[idx], axis=0)

    scandata["wls"] = nlfo_math.chrng(afreq)*1e9
    scandata["time"] = time*1e15
    scandata["Iω_modal"] = Ispec
    scandata["Iω_modal_z"] = Ispec_z
    scandata["Iω"] = Ispec_full
    scandata["It_modal"] = Itime
    scandata["It"] = Itime_full
    scandata["energies"] = energies

    return scandata


if __name__ == "__main__":

    pass