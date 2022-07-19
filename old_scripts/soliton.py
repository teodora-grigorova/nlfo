import numpy as np
import scipy as sp
from scipy import optimize
from scipy import fftpack
from scipy.constants import pi, c, mu_0

import imageio

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import scipy as sp
from scipy import optimize
from scipy.special import erf, erfc
from scipy.constants import pi, c, mu_0

rcParams.update({'figure.autolayout': True})
rcParams.update({"font.size": 9})

parser = argparse.ArgumentParser()
parser.add_argument("location")
# parser.add_argument("left")
# parser.add_argument("right")
parser.add_argument("sim_params")
parser.add_argument("disp")
args = parser.parse_args()

core_radius = 125e-6
T = 293.0

try:
    sim_location = os.path.join("C:\\Users\\Setaou\\Documents\\PhD\\Experiments\\HISOL\\SPM to soliton regime\\simulation\\spectrogram", args.location + "\\ionONshockON")
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