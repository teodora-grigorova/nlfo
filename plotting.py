import h5py
import os
import argparse

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker

import numpy as np
import scipy as sp
from scipy import optimize
from scipy.constants import pi, c, mu_0, epsilon_0, e, m_e
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.fftpack as fft
import scipy.special as special

import nlfo_functions
import nlfo_math
import imports


viridis = cm.viridis                         
viridis_moded = [viridis(i) for i in range(0, viridis.N, 32)]
viridis_moded[0] = (1.0, 1.0, 1.0, 1.0)
white_viridis = LinearSegmentedColormap.from_list("white_viridis", viridis_moded, 8*viridis.N)

figsize_default = (3, (1/1.618)*3)

def updatercParams(font_size=8, axes_labelsize=10, legend_fontsize=7, axes_tickwidth=1, axes_ticksize=3.5, dpi=300, autolayout=True):
    rcParams.update({"figure.autolayout": True})
    rcParams.update({"font.size": font_size})
    rcParams.update({"xtick.major.size": axes_ticksize})
    rcParams.update({"xtick.major.width": axes_tickwidth})
    rcParams.update({"ytick.major.size": axes_ticksize})
    rcParams.update({"ytick.major.width": axes_tickwidth})
    rcParams.update({"figure.dpi": dpi})
    rcParams.update({"legend.fontsize": legend_fontsize})
    rcParams.update({"axes.labelsize": axes_labelsize})
    rcParams.update({"mathtext.default": "regular"})
    rcParams.update({"font.family": "Myriad Pro"})


def calib(wls, spectrum, calibration="vis"):

    if calibration == "vis":
        calib_vis = np.loadtxt(os.path.join("C:\\Users\\Setaou\\Documents\\PhD\\code\\nlfo\\calib_files", "calib.dat"))
        calibration_curve = InterpolatedUnivariateSpline(calib_vis[:, 0], calib_vis[:, 1])
        if len(spectrum) == len(wls):
            calibrated_spectrum = spectrum*calibration_curve(wls)
        else:
            calibrated_spectrum = spectrum[11:]*calibration_curve(wls)

    elif calibration == "nir":
        calib_ir = np.loadtxt(os.path.join("C:\\Users\\Setaou\\Documents\\PhD\\code\\nlfo\\calib_files", "nircalib.dat"))
        dark_ir = np.loadtxt(os.path.join("C:\\Users\\Setaou\\Documents\\PhD\\code\\nlfo\\calib_files", "dark_500ms_nir"))
        calibration_curve = InterpolatedUnivariateSpline(calib_ir[:, 0], calib_ir[:, 1])
        dark = InterpolatedUnivariateSpline(dark_ir[:, 0], dark_ir[:, 1])
        calibrated_spectrum = (spectrum-dark(wls))*calibration_curve(wls)

    elif calibration == "vuv":
        # vuv MgF grating
        calib_grating = np.loadtxt(os.path.join("C:\\Users\\Setaou\\Documents\\PhD\\code\\nlfo\\calib_files", "VUV_grating_efficiency_MgF.dat"))
        calib_detector = np.loadtxt(os.path.join("C:\\Users\\Setaou\\Documents\\PhD\\code\\nlfo\\calib_files", "VUV_detector_efficiency_AX.dat"))
        calibration_curve_grating = InterpolatedUnivariateSpline(calib_grating[:, 0], calib_grating[:, 1])
        calibration_curve_detector = InterpolatedUnivariateSpline(calib_detector[:, 0], calib_detector[:, 1])
        calibrated_spectrum = spectrum*calibration_curve_detector(wls)*calibration_curve_grating(wls)

    else:
        print("available calibration: 'vis', 'nir', and 'vuv_MgF'")

    return calibrated_spectrum


def plot_1D(x, y, scale="lin", style=("black", "solid", 2), fill=False, ext_plot=None, xaxis=None, yaxis=None, xticks=None, yticks=None, label=None, title=None, figsize=figsize_default, doubleyaxis=(False, (None, None, None, None, None, None)), savesvg=(False, None)):
    if plot_multiple1D.called or plot_doublesided1D.called:
        ax = ext_plot.add_subplot()
    else:
        ax = plt.figure(figsize=figsize)

    def Nsol_forward(energy):
        Ldisp = nlfo_functions.L_dispersion(pressure, gas, core_radius=core_radius, tau_fwhm=τfwhm, pump_wl=λpump)
        # using energy*1e-6, because the energy for the plot is passed in uJ, i.e. multiplies by 1e6
        return np.sqrt(Ldisp*nlfo_functions.gamma(pressure, gas, core_radius=core_radius, pump_wl=λpump)*0.88*(energy*1e-6)/τfwhm)

    def Nsol_inverse(Nsol):
        Ldisp = nlfo_functions.L_dispersion(pressure, gas, core_radius=core_radius, tau_fwhm=τfwhm, pump_wl=λpump)
        # the result is multiplied by 1e6, because the energies for the plot are given in uJ
        return ((τfwhm*Nsol**2)/(0.88*nlfo_functions.gamma(pressure, gas, core_radius=core_radius, pump_wl=λpump)*Ldisp))*1e6

    if scale == "lin":
        if style[1] == "o":
            ax.plot(x, y, linestyle="", marker=style[1], markersize=style[2], color=style[0], label=label)
        elif fill:
            ax.plot(x, y, linestyle=style[1], linewidth=style[2], color="black")
        else:
            ax.plot(x, y, linestyle=style[1], linewidth=style[2], color=style[0], label=label)

    elif scale == "lin_normed":
        y = nlfo_math.norm_by_max(y)
        if style[1] == "o":
            ax.plot(x, y, linestyle="", marker=style[1], markersize=style[2], color=style[0], label=label)
        elif fill:
            ax.plot(x, y, linestyle=style[1], linewidth=style[2], color="black")
        else:
            ax.plot(x, y, linestyle=style[1], linewidth=style[2], color=style[0], label=label)

    elif scale == "log_normed":
        y = nlfo_math.log10_normed(y)
        if style[1] == "o":
            ax.plot(x, y, linestyle="", marker=style[1], markersize=style[2], color=style[0], label=label)
        elif fill:
            ax.plot(x, y, linestyle=style[1], linewidth=style[2], color="black")
        else:
            ax.plot(x, y, linestyle=style[1], linewidth=style[2], color=style[0], label=label)

    else:
        print("scale has to be 'lin', 'lin_normed' or 'log_normed'")

    if doubleyaxis[0]:
        if doubleyaxis[2][0] == "N":
            pressure = doubleyaxis[2][1]
            gas = doubleyaxis[2][2]
            τfwhm = doubleyaxis[2][3]
            λpump = doubleyaxis[2][4]
            core_radius = doubleyaxis[2][5]            
            Nmax = nlfo_functions.Nsol(y.max()*1e-6, pressure, gas, tau_fwhm=τfwhm, pump_wl=λpump, core_radius=core_radius)
            axi = ax.secondary_yaxis("right", functions=(Nsol_forward, Nsol_inverse))
            if doubleyaxis[1]:
                axi.set_ylabel("Soliton order")
                axi.set_yticks([idx for idx in range(1, int(Nmax)+1)])
            else:
                axi.set_yticks([idx for idx in range(1, int(Nmax)+1)])
                axi.tick_params(labelright=False)

    if xaxis is not None:
        ax.set_xlabel(xaxis[0])
        ax.set_xlim(xaxis[1])

    if yaxis is not None:
        ax.set_ylabel(yaxis[0])
        ax.set_ylim(yaxis[1])

    if xticks is not None:
        ax.set_xticks(xticks[1])
        if xticks[0] == False:
            ax.tick_params(labelbottom=False)

    if yticks is not None:
        ax.set_yticks(yticks[1])
        if yticks[0] == False:
            ax.tick_params(labelleft=False) 

    if title is not None:
        ax.set_title(title)

    if savesvg[0] is True:
            plt.savefig(savesvg[1])

    if fill:
        kwargs={"color": style[0], "label": label, "zorder":1}
        ax.fill_between(x, y, y2=0, **kwargs)

    if plot_multiple1D.called:
        pass
    else:
        plt.legend(loc="center")
        plt.show()


def plot_multiple1D(xs, ys, scale="lin", xaxis=None, yaxis=None, xticks=None, yticks=None, legend=None, title=None, figsize=figsize_default, styles=None, fill=False, doubleyaxis=(False, (None, None, None, None, None, None)), cross_section=(None, None, None), savesvg=(False, None)):
    plot_multiple1D.called = True

    ax = plt.figure(figsize=figsize)

    if styles:
        num_viridis_styles = [style[0] for style in styles].count("viridis")
        viridis = cm.viridis
        colours = np.linspace(0, viridis.N-100, num_viridis_styles, dtype=np.int64)
        linestyles = []
        idx_viridis = 0
        for style in styles:
            if style[0] == "viridis":
                viridis_style = (viridis(colours[idx_viridis]), style[1], style[2])
                linestyles.append(viridis_style)
                idx_viridis += 1
            else:
                linestyles.append(style)

    for idx in range(0, len(ys)):
        if legend and fill:
            plot_1D(xs[idx], ys[idx], fill=True, ext_plot=ax, scale=scale, style=linestyles[idx], xaxis=xaxis, yaxis=yaxis, xticks=xticks, yticks=yticks, label=legend[idx], doubleyaxis=doubleyaxis, title=title)
        elif legend:
            plot_1D(xs[idx], ys[idx], fill=False, ext_plot=ax, scale=scale, style=linestyles[idx], xaxis=xaxis, yaxis=yaxis, xticks=xticks, yticks=yticks, label=legend[idx], doubleyaxis=doubleyaxis, title=title)
        else:
            plot_1D(xs[idx], ys[idx], ext_plot=ax, scale=scale, style=linestyles[idx], xaxis=xaxis, yaxis=yaxis, xticks=xticks, yticks=yticks, doubleyaxis=doubleyaxis, title=title)

    if cross_section[0]:
        if cross_section[1] == "vertical":
            plt.axvline(x=cross_section[2], color="black", linewidth=1, alpha=0.7, linestyle="--")
        if cross_section[1] == "horizontal":
            plt.axhline(y=cross_section[2], color="black", linewidth=1, alpha=0.7, linestyle="--")


    if legend:
        ax.legend(loc="center")

    if savesvg[0] is True:
        plt.savefig(savesvg[1])

    plt.show()

plot_multiple1D.called = False

def plot_doublesided1D(xs, ys, scale="lin", xaxis=None, yaxis=None, secondyaxis=None, xticks=None, yticks=None, legend=None, title=None, figsize=figsize_default, styles=None, cross_section=(None, None, None), savesvg=(False, None)):
    plot_doublesided1D.called = True

plot_doublesided1D.called = False

def plot_2D(x, y, z, scale="lin", cbar=False, xaxis=None, yaxis=None, vmin=-30, xticks=None, yticks=None, title=None, figsize=figsize_default, cmap=white_viridis, savesvg=(False, None), zdw_line=(False, None, None, None), cross_section=(False, (None, None, None, None, None, None)), doubleyaxis=(False, (None, None, None, None, None, None))):
    # for the current version of matplotlib (3.3.4), there is a problem with using shading="nearest" for non-uniformly spaced grid (such as here), that's why the shape of z has to be (shape(x)-1, shape(y)-1), hence z[-1, :-1]. it should be fixed in next versions.

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    if scale == "lin":
        img = ax.pcolormesh(x, y, z[:-1, :-1], shading='flat', cmap=cmap, rasterized=True)

    elif scale == "lin_normed":
        z = nlfo_math.norm_by_max(z)
        img = ax.pcolormesh(x, y, z[:-1, :-1], shading='flat', cmap=cmap, vmax=1.0, rasterized=True)

    elif scale == "log_normed":
        z = nlfo_math.log10_normed(z)
        img = ax.pcolormesh(x, y, z[:-1, :-1], shading='flat', cmap=cmap, vmin=vmin, rasterized=True)

    else:
        print("scale has to be 'lin', 'lin_normed' or 'log_normed'")

    def Nsol_forward(energy):
        Ldisp = nlfo_functions.L_dispersion(pressure, gas, core_radius=core_radius, tau_fwhm=τfwhm, pump_wl=λpump)
        # using energy*1e-6, because the energy for the plot is passed in uJ, i.e. multiplies by 1e6
        return np.sqrt(Ldisp*nlfo_functions.gamma(pressure, gas, core_radius=core_radius, pump_wl=λpump)*0.88*(energy*1e-6)/τfwhm)

    def Nsol_inverse(Nsol):
        Ldisp = nlfo_functions.L_dispersion(pressure, gas, core_radius=core_radius, tau_fwhm=τfwhm, pump_wl=λpump)
        # the result is multiplied by 1e6, because the energies for the plot are given in uJ
        return ((τfwhm*Nsol**2)/(0.88*nlfo_functions.gamma(pressure, gas, core_radius=core_radius, pump_wl=λpump)*Ldisp))*1e6

    if zdw_line[0]:
        pressure = zdw_line[1]
        gas = zdw_line[2]
        core_radius = zdw_line[3]
        if zdw_line[4] == "v":
            ax.axvline(x=nlfo_functions.λzd(nlfo_functions.β2, pressure=pressure, gas=gas, core_radius=core_radius)*1e9, color="black", linewidth=1, alpha=0.7, linestyle="--")
        if zdw_line[4] == "h":
            ax.axhline(y=nlfo_functions.λzd(nlfo_functions.β2, pressure=pressure, gas=gas, core_radius=core_radius)*1e9, color="black", linewidth=1, alpha=0.7, linestyle="--")


    if cross_section[0]:
        if cross_section[1][0] == "N":
            soliton_order = cross_section[1][1]
            pressure = cross_section[1][2]
            gas = cross_section[1][3]
            τfwhm = cross_section[1][4]
            λpump = cross_section[1][5]
            core_radius = cross_section[1][6]
        elif zdw_line[1][0] == "v":
            ax.axvline(x=cross_section[1][1], color="black", linewidth=1, alpha=0.7, linestyle="--")
        elif zdw_line[1][0] == "h":
            ax.axhline(y=cross_section[1][1], color="black", linewidth=1, alpha=0.7, linestyle="--")

    if doubleyaxis[0]:
        if doubleyaxis[2][0] == "N":
            pressure = doubleyaxis[2][1]
            gas = doubleyaxis[2][2]
            τfwhm = doubleyaxis[2][3]
            λpump = doubleyaxis[2][4]
            core_radius = doubleyaxis[2][5]            
            Nmax = nlfo_functions.Nsol(y.max()*1e-6, pressure, gas, tau_fwhm=τfwhm, pump_wl=λpump, core_radius=core_radius)
            axi = ax.secondary_yaxis("right", functions=(Nsol_forward, Nsol_inverse))
            if doubleyaxis[1]:
                axi.set_ylabel("Soliton order")
                axi.set_yticks([idx for idx in range(1, int(Nmax)+1)])
            else:
                axi.set_yticks([idx for idx in range(1, int(Nmax)+1)])
                axi.tick_params(labelright=False)

    if xaxis is not None:
        ax.set_xlabel(xaxis[0])
        ax.set_xlim(xaxis[1])

    if yaxis is not None:
        ax.set_ylabel(yaxis[0])
        ax.set_ylim(yaxis[1])

    if xticks is not None:
        ax.set_xticks(xticks[1])
        if xticks[0] == False:
            ax.tick_params(labelbottom=False)

    if yticks is not None:
        ax.set_yticks(yticks[1])
        if yticks[0] == False:
            ax.tick_params(labelleft=False) 

    if title is not None:
        ax.set_title(title)

    if cbar:
        kwargs={"shrink":0.8}
        if scale == "log_normed":
            fig.colorbar(img, ax=ax, ticks=[0, -10, -20, -30], **kwargs)
        else:
            fig.colorbar(img, ax=ax, **kwargs)


    if savesvg[0] is True:
        plt.savefig(savesvg[1])


    plt.show()


def plot_2D_baseline_subtraction(x, y, z, scale="lin", cbar=False, xaxis=None, yaxis=None, zaxis=None, xticks=None, yticks=None, title=None, figsize=figsize_default, cmap=white_viridis, savesvg=(False, None), zdw_line=(False, None, None, None), cross_section=(False, (None, None, None, None, None, None)), doubleyaxis=(False, (None, None, None, None, None, None))):
    # for the current version of matplotlib (3.3.4), there is a problem with using shading="nearest" for non-uniformly spaced grid (such as here), that's why the shape of z has to be (shape(x)-1, shape(y)-1), hence z[-1, :-1]. it should be fixed in next versions.

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    if scale == "lin":
        z[np.isnan(z)] = 1e-20
        img = ax.pcolormesh(x, y, z[1:, 1:], shading='flat', cmap=cmap, rasterized=True)

    elif scale == "lin_normed":
        z[np.isnan(z)] = 1e-20
        z = nlfo_math.norm_by_max(z)
        img = ax.pcolormesh(x, y, z[1:, 1:], shading='flat', cmap=cmap, rasterized=True)

    elif scale == "log":
        z[np.isnan(z)] = 1e-20
        z = nlfo_math.log_10(z)
        img = ax.pcolormesh(x, y, z[1:, 1:], shading='flat', cmap=cmap, vmin=-30, rasterized=True)

    elif scale == "log_normed":
        z[np.isnan(z)] = 1e-20
        z = nlfo_math.log10_normed(z)
        img = ax.pcolormesh(x, y, z[1:, 1:], shading='flat', cmap=cmap, vmin=-30, rasterized=True)

    else:
        print("scale has to be 'lin', 'lin_normed' or 'log_normed'")

    def Nsol_forward(energy):
        Ldisp = nlfo_functions.L_dispersion(pressure, gas, core_radius=core_radius, tau_fwhm=τfwhm, pump_wl=λpump)
        # using energy*1e-6, because the energy for the plot is passed in uJ, i.e. multiplies by 1e6
        return np.sqrt(Ldisp*nlfo_functions.gamma(pressure, gas, core_radius=core_radius, pump_wl=λpump)*0.88*(energy*1e-6)/τfwhm)

    def Nsol_inverse(Nsol):
        Ldisp = nlfo_functions.L_dispersion(pressure, gas, core_radius=core_radius, tau_fwhm=τfwhm, pump_wl=λpump)
        # the result is multiplied by 1e6, because the energies for the plot are given in uJ
        return ((τfwhm*Nsol**2)/(0.88*nlfo_functions.gamma(pressure, gas, core_radius=core_radius, pump_wl=λpump)*Ldisp))*1e6

    if zdw_line[0]:
        pressure = zdw_line[1]
        gas = zdw_line[2]
        core_radius = zdw_line[3]
        if zdw_line[4] == "v":
            ax.axvline(x=nlfo_functions.λzd(nlfo_functions.β2, pressure=pressure, gas=gas, core_radius=core_radius)*1e9, color="black", linewidth=1, alpha=0.7, linestyle="--")
        if zdw_line[4] == "h":
            ax.axhline(y=nlfo_functions.λzd(nlfo_functions.β2, pressure=pressure, gas=gas, core_radius=core_radius)*1e9, color="black", linewidth=1, alpha=0.7, linestyle="--")


    if cross_section[0]:
        if cross_section[1][0] == "N":
            soliton_order = cross_section[1][1]
            pressure = cross_section[1][2]
            gas = cross_section[1][3]
            τfwhm = cross_section[1][4]
            λpump = cross_section[1][5]
            core_radius = cross_section[1][6]
        elif zdw_line[1][0] == "v":
            ax.axvline(x=cross_section[1][1], color="black", linewidth=1, alpha=0.7, linestyle="--")
        elif zdw_line[1][0] == "h":
            ax.axhline(y=cross_section[1][1], color="black", linewidth=1, alpha=0.7, linestyle="--")

    if doubleyaxis[0]:
        if doubleyaxis[2][0] == "N":
            pressure = doubleyaxis[2][1]
            gas = doubleyaxis[2][2]
            τfwhm = doubleyaxis[2][3]
            λpump = doubleyaxis[2][4]
            core_radius = doubleyaxis[2][5]            
            Nmax = nlfo_functions.Nsol(y.max()*1e-6, pressure, gas, tau_fwhm=τfwhm, pump_wl=λpump, core_radius=core_radius)
            axi = ax.secondary_yaxis("right", functions=(Nsol_forward, Nsol_inverse))
            if doubleyaxis[1]:
                axi.set_ylabel("Soliton order")
                axi.set_yticks([idx for idx in range(1, int(Nmax)+1)])
            else:
                axi.set_yticks([idx for idx in range(1, int(Nmax)+1)])
                axi.tick_params(labelright=False)

    if xaxis is not None:
        ax.set_xlabel(xaxis[0])
        ax.set_xlim(xaxis[1])

    if yaxis is not None:
        ax.set_ylabel(yaxis[0])
        ax.set_ylim(yaxis[1])

    if xticks is not None:
        ax.set_xticks(xticks[1])
        if xticks[0] == False:
            ax.tick_params(labelbottom=False)

    if yticks is not None:
        ax.set_yticks(yticks[1])
        if yticks[0] == False:
            ax.tick_params(labelleft=False) 

    if zaxis is not None:
        pass

    if title is not None:
        ax.set_title(title)

    if cbar:
        kwargs={"shrink":0.8}
        if scale == "log_normed":
            fig.colorbar(img, ax=ax, ticks=[0, -10, -20, -30], **kwargs)
        else:
            fig.colorbar(img, ax=ax, **kwargs)


    if savesvg[0] is True:
        plt.savefig(savesvg[1])


    plt.show()


if __name__ == "__main__":

    print("plotting.py ran as __main__")