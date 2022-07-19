import matplotlib
import matplotlib as mpl
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams['pdf.fonttype'] = 42
matplotlib.rc('font', size=14)
matplotlib.rc('lines', linewidth=1.5)
matplotlib.rc('axes', linewidth=0.75)
matplotlib.rc('savefig', dpi=600)

import pylab as plt
import h5py
import sys
import scipy as sp
from scipy import pi
from scipy.constants import c
from scipy.interpolate import InterpolatedUnivariateSpline as ius

cdict = {'blue': ((0, 1, 1),
         (0.05, 0.5, 0.5),
         (0.11, 1, 1),
         (0.34, 1, 1),
         (0.65, 0, 0),
         (1, 0, 0)),
'green': ((0, 1, 1),
          (0.05, 0, 0),
          (0.125, 0, 0),
          (0.375, 1, 1),
          (0.64, 1, 1),
          (0.91, 0, 0),
          (1, 0, 0)),
'red': ((0, 1, 1),
        (0.05, 0, 0),
        (0.35, 0, 0),
        (0.66, 1, 1),
        (0.89, 1, 1),
        (1, 0.5, 0.5))}
jet_white= mpl.colors.LinearSegmentedColormap('jet_white',cdict,2048)

import argparse
parser = argparse.ArgumentParser(description="make pretty plots of fnfep output")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-l", "--wavelength", action="store_true", help="plot spectrum on wavelength scale")
group.add_argument("-f", "--frequency", action="store_true", help="plot spectrum on frequency scale")
parser.add_argument("--flo", type=float, default=0.0, help="the lower frequency spectrum axis limit [PHz]")
parser.add_argument("--fhi", type=float, default=1.5, help="the higher frequency spectrum axis limit [PHz]")
parser.add_argument("--llo", type=float, default=200.0, help="the lower wavelength spectrum axis limit [nm]")
parser.add_argument("--lhi", type=float, default=1200.0, help="the higher wavelength spectrum axis limit [nm]")
parser.add_argument("--tlo", type=float, default=-100.0, help="the lower time axis limit [fs]")
parser.add_argument("--thi", type=float, default=100.0, help="the higher time axis limit [fs]")
parser.add_argument("--zlo", type=float, default=0.0, help="the lower position axis limit [cm]")
parser.add_argument("--zhi", type=float, default=1000.0, help="the higher position axis limit [cm]")
parser.add_argument("--dB", type=float, default=40.0, help="the dynamic range of the spectrum dB scale")
parser.add_argument("--dBt", type=float, default=0.0, help="the dynamic range of the temporal dB scale")
parser.add_argument("filename", type=str, help="the filename to process")
args = parser.parse_args()

def get_env(ef):
    ew = sp.fft(ef)
    ew[ew.size//2+1:] = 0.0
    ep = 2.0*sp.ifft(ew)
    return ep

def get_grid(tw, n):
    dT = tw/n
    T = sp.arange(-n/2,n/2)*dT
    dW = 2.0*pi/tw
    F = sp.arange(n/2 + 1)/tw
    L = c/F[1:]
    return T/1e-15, F/1e15, L/1e-9

basename = ".".join(args.filename.split('/')[-1].split('.')[:-1])
f = h5py.File(args.filename, 'r')
T, F, L = get_grid(f['grid_temporal_width'].value, int(f['grid_n'].value))
Z = f['Z'][:]*100.0

iiz = (Z >= args.zlo) & (Z <= args.zhi)
iif = (F >= args.flo) & (F <= args.fhi)
iil = (L >= args.llo) & (L <= args.lhi)
iit = (T >= args.tlo) & (T <= args.thi)

titletxt = ""
#if f['input_mode_indices'].shape[0] == 1:
#    idx = f['input_mode_indices'][0]
#    titletxt += ('%f fs, %f $\mu$J, %f nm pump in mode ' % (f['input_fwhms'][0]/1e-15,
#                                       f['input_energies'][0]/1e-6, f['input_wavelengths'][0]/1e-9)) 
#    titletxt += f['mode_types'][idx] + "%i%i\n" % (f['mode_ns'][idx], f['mode_ms'][idx])
#titletxt += ("%f" % (f['gas_pressure'].value)) + " bar " + str(f['gas'].value) + " in " + ("%f" % (f['fibre_radius'].value/1e-6)) + " $\mu$m radius kagome\n"
titletxt = "title"

ss = []
for j in range(f['E'].shape[1]):
    lspecs = [sp.real(abs(sp.fft(f['E'][i][j][:])[:f['E'].shape[-1]//2+1])**2) for i in range(f['E'].shape[0])]
    if args.frequency:
        specs = [10.0*sp.log10(lspec) for lspec in lspecs]
    else:
        wls = sp.linspace(args.llo, args.lhi, L[iil].size)
        iil2 = sp.argsort(L[iil])
        specs = [sp.real(10.0*sp.log10(ius(L[iil][iil2], lspec[1:][iil][iil2])(wls)/wls**2)) for lspec in lspecs]
    specs = sp.array(specs)
    ss.append(specs)
mm = max([specs.max() for specs in ss])
if args.dB > 0:
    vmax = 0.0
    vmin = -args.dB
else:
    vmin = 0.0
    vmax = 10.0**(mm/10.0)
for i,specs in enumerate(ss):
    if args.dB > 0:
        specs = specs - mm
    else:
        specs = 10.0**(specs/10.0)
    plt.figure()
    #zf = f['mode_dispersion_zeros'][i]
    if args.frequency:
        #plt.pcolormesh(F[iif], Z[iiz], specs[:,iif][iiz,:])
        plt.imshow(sp.flipud(specs[:,iif][iiz,:]),
                extent=(F[iif].min(), F[iif].max(), Z[iiz].min(), Z[iiz].max()),
                vmin=vmin, vmax=vmax, cmap=jet_white, rasterized=True,
                interpolation='bicubic')
        plt.plot(f['zdf'][iiz,i]/1e15, Z[iiz], lw=3, color='k', ls='--')
        #plt.axvline(zf/1e15, color='w', lw=3, ls='--')
        plt.xlabel("Frequency (PHz)")
    else:
        #plt.pcolormesh(wls, Z[iiz], specs[iiz,:])
        plt.imshow(sp.flipud(specs[iiz,:]),
                extent=(wls.min(), wls.max(), Z[iiz].min(), Z[iiz].max()),
                vmin=vmin, vmax=vmax, cmap=jet_white, rasterized=True,
                interpolation='bicubic')
        plt.plot(c/f['zdf'][iiz,i]/1e-9, Z[iiz], lw=3, color='k', ls='--')
        #plt.axvline(c/zf/1e-9, color='w', lw=2, ls='--')
        plt.xlabel("Wavelength (nm)")      
    plt.axis('tight')
    plt.colorbar()
    #plt.clim(-args.dB, 0.0)
    plt.ylabel("Position (cm)")
    plt.title(titletxt + "Output spectrum in mode " + str(f['mode_types'][i]) + "%i%i\n" % (f['mode_ns'][i], f['mode_ms'][i]))
    plt.subplots_adjust(top=0.78, right=0.99)
    plt.savefig(basename + "_spec_%02i.png" % i)

#lspecs = []
#for i in range(f['E'].shape[0]):
#    Er = sp.zeros(f['E'].shape[-1])
#    for j in range(f['E'].shape[1]):
#        Er += f['E'][i][j][:]
#    lspecs.append(sp.real(abs(sp.fft(Er)[:f['E'].shape[-1]/2+1])**2))
#if args.frequency:
#    specs = [10.0*sp.log10(lspec) for lspec in lspecs]
#else:
#    wls = sp.linspace(args.llo, args.lhi, L[iil].size)
#    iil2 = sp.argsort(L[iil])
#    specs = [sp.real(10.0*sp.log10(ius(L[iil][iil2], lspec[iil][iil2])(wls)/wls**2)) for lspec in lspecs]
#msum = sp.array(specs)
ss = 10.0**(sp.array(ss)/10.0)
#print ss.shape
msum = sp.sum(ss, axis=0)
#print msum.shape
msum = 10.0*sp.log10(msum)
if args.frequency:
    msum = msum[:,iif][iiz,:]
else:
    msum = msum[iiz,:]
if args.dB > 0:
    vmax = 0.0
    vmin = -args.dB
    msum = msum - msum.max()
else:
    vmin = 0.0
    msum = 10.0**(msum/10.0)
    vmax = msum.max()
plt.figure()
if args.frequency:
    plt.imshow(sp.flipud(msum),
               extent=(F[iif].min(), F[iif].max(), Z[iiz].min(), Z[iiz].max()),
               vmin=vmin, vmax=vmax, cmap=jet_white, rasterized=True,
                interpolation='bicubic')
    plt.xlabel("Frequency (PHz)")
else:
    plt.imshow(sp.flipud(msum),
               extent=(wls.min(), wls.max(), Z[iiz].min(), Z[iiz].max()),
               vmin=vmin, vmax=vmax, cmap=jet_white, rasterized=True,
                interpolation='bicubic')
    plt.xlabel("Wavelength (nm)")      
plt.axis('tight')
plt.colorbar()
plt.ylabel("Position (cm)")
plt.title(titletxt + "Output spectrum in all modes")
plt.subplots_adjust(top=0.78, right=0.99)
plt.savefig(basename + "_spec_all.png")


for j in range(f['E'].shape[1]):
    ints = [abs(get_env(f['E'][i][j][:])[iit])**2 for i in range(f['E'].shape[0])]
    ints = sp.array(ints)
    if args.dBt > 0.0:
        ints = 10.0*sp.log10(ints)
    plt.figure()
    if args.dBt > 0.0:
        vmin = ints[iiz,:].max() - args.dBt
    else:
        vmin = ints[iiz,:].min()
    vmax = ints.max()
    plt.imshow(sp.flipud(ints[iiz,:]),
                extent=(T[iit].min(), T[iit].max(), Z[iiz].min(), Z[iiz].max()),
                vmin=vmin, vmax=vmax, cmap=jet_white, rasterized=True,
                interpolation='bicubic')
    plt.axis('tight')
    plt.colorbar()
    plt.xlabel("Delay (fs)")
    plt.ylabel("Position (cm)")
    plt.title(titletxt + "Output intensity in mode: " + str(f['mode_types'][j]) + "%i%i\n" % (f['mode_ns'][j], f['mode_ms'][j]))
    plt.subplots_adjust(top=0.78, right=0.99)
    plt.savefig(basename + "_temp_%02i.png" % j)
    
plt.figure(figsize=(7,12))
plt.subplot(411)
plt.title(titletxt)
plt.semilogy(f['stats_z'][:]*100.0, f['stats_peak_I'][:]/1e4)
plt.ylabel('Intensity (W/cm$^2$)')
plt.subplot(412)
for i in range(f['stats_fwhm_t_m'].shape[1]):
    plt.plot(f['stats_z'][:]*100.0, f['stats_fwhm_t_m'][:,i]/1e-15)
plt.ylim(0, 2*f['stats_fwhm_t_m'][0,0]/1e-15)
plt.ylabel('Pulse FWHM (fs)')
plt.subplot(413)
plt.plot(f['stats_z'][:]*100.0, f['stats_fwhm_r'][:]/1e-6)
plt.ylabel('Beam Diameter ($\mu$m)')
plt.subplot(414)
plt.plot(f['stats_z'][:]*100.0, f['stats_energy'][:]/f['stats_energy'][0], lw=3.0)
for i in range(f['stats_energy_m'].shape[1]):
    plt.semilogy(f['stats_z'][:]*100.0, f['stats_energy_m'][:,i]/f['stats_energy'][0])
plt.xlabel('Position in fibre (cm)')
plt.ylabel('Relative energy')
#plt.subplots_adjust(bottom=0.06, left=0.15, right=0.94, top=0.9, hspace=0.2)
plt.savefig(basename + "_stats.png")

if f["ionization_enabled"].value == "true":
    plt.figure(figsize=(7,9))
    plt.subplot(211)
    plt.title(titletxt)
    fed = f['stats_peak_Ne'][1:]/1e6;
    if sp.all(fed > 0.0):
        plt.semilogy(f['stats_z'][1:]*100.0, fed)
    plt.ylabel('Free electron\ndensity (1/cm$^3$)')
    plt.subplot(212)
    il = f["ion_levels"][:]
    for i in range(0, il.shape[1]):
        if Z[1:].shape == il[:,i,-1].shape:
            plt.semilogy(Z[1:],il[:,i,-1]/1e6, label="+%i" % i)
    plt.ylabel("Ion density (cm$^{-3}$)")
    plt.legend()
    plt.xlabel('Position in fibre (cm)')
    plt.subplots_adjust(left=0.15)
    plt.savefig(basename + "_ion_stats.png")

if f["raman_enabled"].value == "true":
    plt.figure(figsize=(7,5))
    plt.title(titletxt)
    plt.plot(f['stats_z'][1:]*100.0,
            f['stats_raman_max_w'][1:])
    plt.ylabel('Max Raman\npopulation')
    plt.xlabel('Position in fibre (cm)')
    #plt.subplots_adjust(bottom=0.06, left=0.15, right=0.94, top=0.94, hspace=0.2)
    plt.savefig(basename + "_raman_stats.png")

plt.figure(figsize=(7,9))
plt.subplot(211)
plt.title(titletxt)
plt.plot(f['stats_z'][:]*100.0, f['stats_pressure'][:], lw=3.0)
plt.ylabel('Gas pressure [bar]')
plt.subplot(212)
for j in range(f['E'].shape[1]):
    plt.plot(Z, c/f['zdf'][:,j]/1e-9)
plt.ylabel('Zero dispersion wavelength')
plt.xlabel('Position in fibre (cm)')
plt.savefig(basename + "_fibre_stats.png")

betas = f['betas'][:]
W = 2*pi*F*1e15
for i in range(0,): #betas.shape[1]):
    #fig = plt.figure(figsize=(7,9))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7,9))
    plt.sca(ax1)
    plt.title(titletxt + "Dispersion of mode " + f['mode_types'][i] + "%i%i\n" % (f['mode_ns'][i], f['mode_ms'][i]))
    #ax1 = fig.add_subplot(211)
    ax1.set_ylabel("Group velocity ($c$)")
    #ax2 = fig.add_subplot(212)
    ax2.set_ylabel("Dispersion (fs$^2$/cm)")
    if args.frequency:
        ax2.set_xlabel("Frequency (PHz)")
    else:
        ax2.set_xlabel("Wavelength (nm)")
    for j in range(betas.shape[0]):
        spl = ius(W, betas[j,i,:])
        b1s = sp.array([spl.derivatives(w)[1] for w in W])
        spl1 = ius(W, b1s)
        b2s = sp.array([spl1.derivatives(w)[1] for w in W])
        if args.frequency:
            ax1.plot(F[iif], 1.0/b1s[iif]/c)
            ax2.plot(F[iif], b2s[iif]/100.0/1e-30)
        else:
            ax1.plot(L[iil], 1.0/b1s[1:][iil]/c)
            ax2.plot(L[iil], b2s[1:][iil]/100.0/1e-30)
    #ax1.set_axes('tight')
    #ax2.set_axes('tight')
    plt.savefig(basename + "_fibre_disp_%03i.png" % i)



plt.show()

