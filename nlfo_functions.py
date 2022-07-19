import numpy as np
import scipy as sp
from scipy import optimize
from scipy.constants import pi, c, mu_0, epsilon_0, e, m_e, N_A
import scipy.special as special
from scipy.misc import derivative

from CoolProp.CoolProp import PropsSI

import nlfo_math


def field_envelope(E):
    Eω = sp.fft(E)
    Eω[len(E)//2+1:] = 0.0
    Et = 2.0*sp.ifft(Eω)
    return Et


def analytical_signal(t, Eω):
    Et = 2.0*sp.ifft(Eω, len(t))
    return Et


def rho(pressure, T=293.0, pressure0=1.0, T0=273.0):
    ρ = (pressure/pressure0)*(T0/T)
    return ρ


def bar_to_pascal(pressure):
    return pressure*1e5


def gas_density(gas, pressure, temperature=293.0):
    return PropsSI("DMOLAR", "T", temperature, "P", bar_to_pascal(pressure), gas)*N_A


def χ_gas(wls, gas):

    if gas == "He":
        B = [4977.77e-8, 1856.94e-8]
        C = [28.54e-6, 7.760e-3]
    elif gas == "He_JCT":
        B = [2.16463842e-05, 2.10561127e-07, 4.75092720e-05]
        C = [-6.80769781e-04, 5.13251289e-03, 3.18621354e-03]
    elif gas == "Ne":
        B = [9154.48e-8, 4018.63e-8]
        C = [656.97e-6, 5.728e-3]
    elif gas == "Ar":
        B = [20332.29e-8, 34458.31e-8]
        C = [206.12e-6, 8.066e-3]
    elif gas == "Kr":
        B = [26102.88e-8, 56946.82e-8]
        C = [2.01e-6, 10.043e-3]
    elif gas == "Xe":
        B = [103701.61e-8, 31228.61e-8]
        C = [12.75e-3, 0.561e-3]
    elif gas == "N2":
        B = [39209.95e-8, 18806.48e-8]
        C = [1146.24e-6, 13.476e-3]
    else:
        print("gas not implemented")

    if gas == "He_JCT":
        return B[0]*(wls*1e6)**2/((wls*1e6)**2 - C[0]) + B[1]*(wls*1e6)**2/((wls*1e6)**2 - C[1]) + B[2]*(wls*1e6)**2/((wls*1e6)**2 - C[2])
    else:
        return B[0]*(wls*1e6)**2/((wls*1e6)**2 - C[0]) + B[1]*(wls*1e6)**2/((wls*1e6)**2 - C[1])


def n_gas(wls, pressure, gas, T=293.0):
    ngas = np.sqrt(1.0 + rho(pressure, T)*χ_gas(wls, gas))
    return ngas


def chi3(pressure, gas, T=293.0):
    if gas == "He":
        χ3 = 4.0*3.43e-28
    elif gas == "He_JCT":
        χ3 = 4.0*3.43e-28
    elif gas == "Ne":
        χ3 = 4.0*1.8*3.43e-28
    elif gas == "Ar":
        χ3 = 4.0*23.5*3.43e-28
    elif gas == "Kr":
        χ3 = 4.0*64.0*3.43e-28
    elif gas == "Xe":
        χ3 = 4.0*188.2*3.43e-28
    elif gas == "N2":
        χ3 = 4.0*21.1*3.43e-28
    else:
        print("gas not implemented")
    χ3gas = rho(pressure, T)*χ3
    return χ3gas


def plasma_frequency(ne):
    return np.sqrt((ne*e**2)/(epsilon_0*m_e))


def ne(gas="Ar", percentage=0.0, pressure=1.0, T=293.0):
    if gas == "Ar":
        density = 1.7837
        m_atom = 6.6335209e-26
    else:
        print("only ionisation fraction for Ar for now")
    n0 = density*rho(pressure, T)/m_atom
    return percentage*n0


def beta(omega, pressure=100e-3, gas="Ar", core_radius=125e-6, plasma=(False, None), mode="he11", T=293.0):
    λ = nlfo_math.chrng(omega)
    if core_radius:
        if plasma[0]:
            β = omega/c*np.sqrt(n_gas(λ, pressure, gas, T)**2 - ((special.jn_zeros(0, 1)[0]**2)*(c**2)/((core_radius**2)*omega**2)) - plasma_frequency(ne(gas=gas, percentage=plasma[1], pressure=pressure, T=T))**2/(2*omega**2))
            return β
        else:
            n = int(mode[-2]) - 1
            m = int(mode[-1])
            β = omega/c*np.sqrt(n_gas(λ, pressure, gas, T)**2 - ((special.jn_zeros(n, m)[-1]**2)*(c**2)/((core_radius**2)*omega**2)))
            return β 
    else:
        β = omega/c*n_gas(λ, pressure, gas, T)
        return β


def β(omega, pressure=100e-3, gas="Ar", core_radius=125e-6):
    λ = nlfo_math.chrng(omega)
    if core_radius:
        β = omega/c*np.sqrt(n_gas(λ, pressure, gas)**2 - ((special.jn_zeros(0, 1)[-1]**2)*(c**2)/((core_radius**2)*omega**2)))
        return β 
    else:
        β = omega/c*n_gas(λ, pressure, gas)
        return β


def gdd(omega, pressure=100e-3, gas="Ar", core_radius=125e-6, plasma=(False, None), mode="he11", T=293.0):
    if type(omega) == float:
        Δω = 1e13
    else:
        Δω = abs(omega[1] - omega[0])/1e2
    β1 = derivative(β, omega, dx=Δω, n=1, args=(pressure, gas, core_radius, plasma, mode, T), order=3)
    return β1


def β1(omega, pressure=100e-3, gas="Ar", core_radius=125e-6):
    if type(omega) == float:
        Δω = 1e13
    else:
        Δω = abs(omega[1] - omega[0])/1e2
    β1 = derivative(beta, omega, dx=Δω, n=1, args=(pressure, gas, core_radius), order=3)
    return β1


def gvd(omega, pressure=100e-3, gas="Ar", core_radius=125e-6, plasma=(False, None), mode="he11", T=293.0):
    if type(omega) == float:
        Δω = 1e13
    else:
        Δω = abs(omega[1] - omega[0])/1e2
    β2 = derivative(beta, omega, dx=Δω, n=2, args=(pressure, gas, core_radius, plasma, mode, T), order=3)
    return β2


def β2(omega, pressure=100e-3, gas="Ar", core_radius=125e-6):
    if type(omega) == float:
        Δω = 1e13
    else:
        Δω = abs(omega[1] - omega[0])/1e2
    β2 = derivative(β, omega, dx=Δω, n=2, args=(pressure, gas, core_radius), order=3)
    return β2


def λzd(gvd_function, pressure, gas, core_radius):
    if gas == "Kr":
        wl_lims=(103e-9, 2000e-9)
    elif gas == "He_JCT":
        wl_lims=(80e-9, 2000e-9)
    else:
        wl_lims=(90e-9, 2000e-9)

    ωmin = nlfo_math.chrng(wl_lims[1])
    ωmax = nlfo_math.chrng(wl_lims[0])

    ωzd = sp.optimize.brentq(gvd_function, ωmin, ωmax, args=(pressure, gas, core_radius))

    λzd = nlfo_math.chrng(ωzd)
    return λzd


def given_λzd_find_p(gvd_function, zdw, gas, core_radius=125e-6):
    # pressure limits for zdw in the range (300, 1400) nm
    if gas == "He" or gas == "He_JCT":
        plims = (130e-3, 87.4)
    elif gas == "Ne":
        plims = (70e-3, 44.1)
    elif gas == "Ar":
        plims = (5e-3, 4.51)
    elif gas == "Kr":
        plims = (2e-3, 2.22)
    else:
        print("gas is not recognised")

    def find_λzd(pressure):
        return λzd(gvd_function, pressure, gas, core_radius) - zdw
    
    pmin = plims[0]
    pmax = plims[1]
    pressure = sp.optimize.brentq(find_λzd, pmin, pmax)

    return pressure


def tau_0(tau_fwhm):
    return tau_fwhm/(2*np.log(1 + np.sqrt(2)))


def L_dispersion(pressure=100e-3, gas="Ar", core_radius=125e-6, tau_fwhm=10e-15, pump_wl=800e-9, T=293.0):
    τ0 = tau_0(tau_fwhm)
    ωpump = nlfo_math.chrng(pump_wl)
    beta2 = β2(ωpump, pressure, gas, core_radius)
    return τ0**2/abs(beta2)


def peak_power(energy, tau_fwhm, pulse_shape="sech**2"):
    if pulse_shape == "sech**2":
        return 0.88*energy/tau_fwhm
    elif pulse_shape == "gauss":
        return 0.94*energy/tau_fwhm
    else:
        print("only implemented for analytical sech**2 and gauss pulses")


def gamma(pressure, gas, core_radius=125e-6, pump_wl=800e-9, T=293.0):
    ωpump = nlfo_math.chrng(pump_wl)
    χ3 = chi3(pressure, gas, T)
    n = n_gas(pump_wl, pressure, gas, T)
    return (3/4)*(ωpump*mu_0*χ3)/(0.48*pi*core_radius**2*n**2)


def n2(pressure, gas, pump_wl=800e-9, T=293.0):
    χ3 = chi3(pressure, gas, T)
    n = n_gas(pump_wl, pressure, gas, T)
    return (3/4)*(c*mu_0*χ3)/(n**2)


def L_nonlinear(energy, pressure, gas, core_radius=125e-6, tau_fwhm=10e-15, pulse_shape="sech**2", pump_wl=800e-9, T=293.0):
    P_peak = peak_power(energy, tau_fwhm, pulse_shape)
    γ = gamma(pressure, gas, core_radius, pump_wl, T)
    return 1/(γ*P_peak)


def Nsol(energy, pressure, gas, tau_fwhm=10e-15, pulse_shape="sech**2", pump_wl=800e-9, core_radius=125e-6, T=293.0):
    Ldisp = L_dispersion(pressure=pressure, gas=gas, core_radius=core_radius, tau_fwhm=tau_fwhm, pump_wl=pump_wl, T=T)
    Lnl = L_nonlinear(energy, pressure, gas, core_radius=core_radius, tau_fwhm=tau_fwhm, pulse_shape=pulse_shape, pump_wl=pump_wl, T=T)
    return np.sqrt(Ldisp/Lnl)


def given_Nsol_find_E(Nsol, pressure, gas, tau_fwhm=10e-15, pulse_shape="sech**2", pump_wl=800e-9, core_radius=125e-6, T=293.0):
    Ldisp = L_dispersion(pressure, gas, core_radius, tau_fwhm, pump_wl)
    if pulse_shape == "sech**2":
        return (tau_fwhm*Nsol**2)/(0.88*gamma(pressure, gas, core_radius, pump_wl)*Ldisp)
    elif pulse_shape == "gauss":
        return (tau_fwhm*Nsol**2)/(0.94*gamma(pressure, gas, core_radius, pump_wl)*Ldisp)
    else:
        print("only implemented for analytical sech**2 and gauss pulses")


def find_Nsol_slice(energies, data, Nsol, pressure, gas, tau_fwhm=10e-15, pulse_shape="sech**2", pump_wl=800e-9, core_radius=125e-6, T=293.0):
    energy = given_Nsol_find_E(Nsol, pressure, gas, tau_fwhm, pulse_shape, pump_wl, core_radius, T)
    idx_Nsol = np.abs(energies - energy).argmin()
    return data[idx_Nsol, :]


def fλ(λ, pressure, gas):
    if type(λ) == float or type(λ) == np.float64:
        Δλ = 1e-11
    else:
        Δλ = abs(λ[1] - λ[0])/1e2
    return derivative(χ_gas, λ, dx=Δλ, n=2, args=(pressure, gas), order=3)


def δ(λ, λzd, pressure, gas):
    return (((special.jn_zeros(0, 1)[-1]**2)*λ**3)/(8*(pi**3)*c**2))*(fλ(λ, pressure, gas)/fλ(λzd, pressure, gas) - 1)


def N_max_sf(gas, pressure, λ_pump=800e-9, τ_pump=10e-15, safety_factor=10, core_radius=125e-6):
    omega = np.linspace(100e-9, 3000e-9)
    zdw = λzd(β2, pressure, gas, core_radius)
    delta = δ(λ_pump, zdw, pressure, gas)
    return np.sqrt((λ_pump*tau_0(τ_pump)**2)/(safety_factor*abs(delta)))


def κ2Ith(gas):
    # copied from John's code; what is κ2 - the unscaled nonlinearity?
    if gas == 'He':
        Ip = 24.58741
        κ2 = 3.56e-25
    elif gas == 'Ne':
        Ip = 21.5646
        κ2 = 6.42e-25
    elif gas == 'Ar':
        Ip = 15.7596
        κ2 = 8.38e-24
    elif gas == 'Kr':
        Ip = 13.99961
        κ2 = 2.28e-23
    elif gas == 'Xe':
        Ip = 12.1298
        κ2 = 6.71e-23
    else:
        raise ValueError("unknown gas")
    Ith = 4e9*Ip**4*1e4
    return κ2, Ith


def barrier_suppression_intensity(gas, ion_level=1):
    if gas == 'He':
        Ip = 24.58741
    elif gas == 'Ne':
        Ip = 21.5646
    elif gas == 'Ar':
        Ip = 15.7596
    elif gas == 'Kr':
        Ip = 13.99961
    elif gas == 'Xe':
        Ip = 12.1298
    else:
        raise ValueError("unknown gas")
    Ibsi = 4e9*(Ip**4)*(ion_level**(-2))
    return Ibsi


def N_max_ion(gas, pressure, λ_pump=800e-9, τ_pump=10e-15, safety_factor=10, core_radius=125e-6):
    omega = np.linspace(100e-9, 3000e-9)
    zdw = λzd(β2, pressure, gas, core_radius)
    delta = δ(λ_pump, zdw, pressure, gas)
    κ2, Ith = κ2Ith(gas)
    return np.sqrt(((κ2*special.jn_zeros(0, 1)[-1]**2)*Ith*tau_0(τ_pump)**2)/(safety_factor*pi*λ_pump*abs(delta)*fλ(zdw, pressure, gas)))


def compression_ratio(Nsol):
    return 4.6*Nsol


def selfcompressed_peak_power(Nsol, pressure, gas, tau_fwhm=10e-15, pulse_shape="sech**2", pump_wl=800e-9, core_radius=125e-6):
    energy = given_Nsol_find_E(Nsol, pressure, gas, tau_fwhm=tau_fwhm, pulse_shape=pulse_shape, pump_wl=pump_wl, core_radius=core_radius)
    return peak_power(energy, tau_fwhm, pulse_shape=pulse_shape)*compression_ratio(Nsol)


def βsol(omega, Nsol, pressure, gas, tau_fwhm=10e-15, pump_wl=800e-9, core_radius=125e-6, pulse_shape="sech**2"):
    omega_sol = nlfo_math.chrng(pump_wl)
    beta = β(omega_sol, pressure=pressure, gas=gas, core_radius=core_radius)
    gdd = β1(omega_sol, pressure=pressure, gas=gas, core_radius=core_radius)
    γ = gamma(pressure, gas, core_radius=core_radius, pump_wl=pump_wl)
    Fc = selfcompressed_peak_power(Nsol, pressure, gas, tau_fwhm=tau_fwhm, pulse_shape=pulse_shape, pump_wl=pump_wl, core_radius=core_radius)
    return beta + gdd*(omega - omega_sol) + 0.5*γ*Fc


def dw_phasematching(omega, pump_wl, Nsol, pressure, gas, tau_fwhm=10e-15, core_radius=125e-6, pulse_shape="sech**2"):
    return β(omega, pressure=pressure, gas=gas, core_radius=core_radius) - βsol(omega, Nsol, pressure, gas, tau_fwhm=tau_fwhm, pump_wl=pump_wl, core_radius=core_radius, pulse_shape=pulse_shape)


def λdw(pump_wl, Nsol, pressure, gas, tau_fwhm=10e-15, core_radius=125e-6, pulse_shape="sech**2"):
    if gas == "Kr":
        wl_lims=(103e-9, 2000e-9)
    elif gas == "He_JCT":
        wl_lims=(80e-9, 2000e-9)
    else:
        wl_lims=(90e-9, 2000e-9)

    ωmin = nlfo_math.chrng(wl_lims[1])
    ωmax = nlfo_math.chrng(wl_lims[0])
    ωdw = sp.optimize.brentq(dw_phasematching, ωmin, ωmax, args=(pump_wl, Nsol, pressure, gas, tau_fwhm, core_radius, pulse_shape))
    return nlfo_math.chrng(ωdw)


def givenλzd_findλdw(gvd_function, zdw, pump_wl, Nsol, gas, tau_fwhm=10e-15, core_radius=125e-6, pulse_shape="sech**2"):
    if gas == "Kr":
        wl_lims=(103e-9, 2000e-9)
    elif gas == "He_JCT":
        wl_lims=(80e-9, 2000e-9)
    else:
        wl_lims=(90e-9, 2000e-9)

    ωmin = nlfo_math.chrng(wl_lims[1])
    ωmax = nlfo_math.chrng(wl_lims[0])
    pressure = given_λzd_find_p(gvd_function, zdw, gas, core_radius=125e-6)
    ωdw = sp.optimize.brentq(dw_phasematching, ωmin, ωmax, args=(pump_wl, Nsol, pressure, gas, tau_fwhm, core_radius, pulse_shape))
    return nlfo_math.chrng(ωdw)


def filter_Et(Et, t, λ0, Δλ, filter_fun=("gauss", 10)):
    def gauss_filter(x, x0, σ, supergauss_power):
        return sp.exp(-0.5*((x-x0)/σ)**supergauss_power)
    def chΔλ(Δλ, λ0):
        return (c/λ0**2)*Δλ
    Nt = Et.shape[0]
    Δω = chΔλ(Δλ, λ0)
    filt = gauss_filter(t, λ0, Δλ, filter_fun[1])

# TODO: function to calculate the intensity in a gaussian spot or a mode given a certain energy

def peak_intensity(energy, tau_fwhm, w0=80e-6):
    A = pi*(w0**2)
    Ppeak = peak_power(energy, tau_fwhm, pulse_shape="sech**2")
    return Ppeak/A

# TODO: a filtering function!!!!
# def plot_filtered2(filename, λ0IR, ΔλIR, λ0VIS, ΔλVIS, supergauss_power, plot_filtered=True):
#     def gauss_filter(x, x0, σ, supergauss_power):
#         return sp.exp(-0.5*((x-x0)/σ)**supergauss_power)
#     def chΔλ(Δλ, λ0):
#         return (c/λ0**2)*Δλ
#     z, ω, t, Eω, Et, _, _ = load_sim_prop(filename)
#     # stats: "pressure", "output energy", "input energy", "free electron density", "z", "neutral gas density"
#     # # wl = chrng(ω)*1e9
#     # # wl[0] = 6e5
#     Nt = len(t)
#     T = Nt//2-1
#     Nz = len(z)
#     Nw = len(ω)
#     gaussIR = np.zeros(Nt)
#     gaussIR[T:] = gauss_filter(ω, chrng(λ0IR), chΔλ(ΔλIR, λ0IR), supergauss_power)
#     gaussVIS = np.zeros(Nt)
#     gaussVIS[T:] = gauss_filter(ω, chrng(λ0VIS), chΔλ(ΔλVIS, λ0VIS), supergauss_power)
#     filteredEωIR = np.zeros((Nz, Nt), dtype=np.cdouble)
#     filteredEωIRprop = np.zeros((Nz, Nw))
#     filteredEtIR = np.zeros((Nz, Nt))
#     ###### here this array if of real values only, because otherwise the printing complains; if I need phase I need to change it
#     filteredEωVIS = np.zeros((Nz, Nt), dtype=np.cdouble)
#     filteredEωVISprop = np.zeros((Nz, Nw))
#     filteredEtVIS = np.zeros((Nz, Nt))
#     for step in range(0, len(z)):
#         filteredEωIR[step, T:] = gaussIR[T:]*Eω[step, 0, :]
#         filteredEωIRprop[step, :] = abs(filteredEωIR[step, T:]/filteredEωIR[0, T:].max())**2
#         filteredEtIR[step, :] = abs(2.0*sp.ifft(filteredEωIR[step, :]))**2
#         filteredEωVIS[step, T:] = gaussVIS[T:]*Eω[step, 0, :]
#         filteredEωVISprop[step, :] = abs(filteredEωVIS[step, T:]/filteredEωVIS[0, T:].max())**2
#         filteredEtVIS[step, :] = abs(2.0*sp.ifft(filteredEωVIS[step, :]))**2
#     colour = cm.viridis(0)
#     colourIR = cm.viridis(100)
#     colourVIS = cm.viridis(200)
#     colourUV = cm.viridis(300)
#     if plot_filtered == True:
#         filteredEωIR = gaussIR[T:]*(Eω[-1, 0, :]/Eω[-1, 0, :].max())
#         filteredEωVIS = gaussVIS[T:]*(Eω[-1, 0, :]/Eω[-1, 0, :].max())
#         plt.figure(dpi=200)
#         plt.plot(ω*1e-15, abs((Eω[-1, 0, :]))**2/(abs(Eω[-1, 0, :])**2).max(), color=colour, label="Full time")
#         #plt.plot(ω*1e-15, abs(filteredEωIR)**2, color=colourIR, linestyle="--", label="Filtered IR")
#         #plt.plot(ω*1e-15, abs(filteredEωVIS)**2, color=colourVIS, linestyle="--", label="Filtered VIS")
#         plt.xlabel("Ang. frequency (rad/fs)")
#         plt.ylabel("SPD (a.u.)")
#         plt.axvline(x=2.35, color="black", linestyle="--")
#         plt.xlim(0.1, 7)
#     else:
#         plt.figure(dpi=200)
#         plt.plot(t*1e15, 1e-9*abs(Et[-1, 0, :])**2, color=colour, label="Full time")
#         plt.plot(t*1e15, filteredEtIR[-1, :]*1e-9, color=colourIR, label="Filtered IR")
#         plt.plot(t*1e15, filteredEtVIS[-1, :]*1e-9, color=colourVIS, label="Filtered VIS")
#         plt.xlim(-30, 50)
#         plt.xlabel("Time (fs)")
#         plt.ylabel("Power (GW)")
#         plt.legend()
#     plt.show()

if __name__ == "__main__":
    
    print(Nsol(110e-6, 0.25, "Ar", tau_fwhm=10e-15, pulse_shape="sech**2", pump_wl=1030e-9, core_radius=100e-6))

    print(λzd(β2, 0.25, "Ar", 100e-6)*1e9)
