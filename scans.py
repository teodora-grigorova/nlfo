
import numpy as np
from scipy import integrate

import imports

import peakutils
import plotting
import nlfo_math
import nlfo_functions


class ExpScan(object):

    def __init__(self, filename, scan_type, energy_calibration="march2021", stats={"pressure": 100e-3, "gas": "He", "τfwhm": 10e-15, "pulse_shape": "sech**2", "λpump": 800e-9, "core_radius": 125e-6}):

        self.type = scan_type

        if self.type == "vis":
            scan_data = imports.load_escan(filename, self.type)
        elif self.type == "vuv":
            scan_data = imports.load_escan(filename, self.type)
        else:
            raise AttributeError("The scan type should be given ('vis' or 'vuv').")

        self.λ = scan_data["λ"]
        self.angles = scan_data["angles"]
        self.scan = scan_data["Iλ"]
        self.stats = stats

        if energy_calibration == "march2021":
            # angle to energy calibration for data from 16 March 2021
            self.calib_p_fit = np.array([-1253.88647158, 1.99976171, -413.40024199, 1255.76066302])
            self.calib_angles = np.asarray([52.0, 54.0, 56.0, 58.0, 60.0, 62.0, 64.0, 66.0, 68.0, 70.0, 72.0, 74.0, 76.0, 78.0, 80.0, 82.0, 84.0, 86.0, 88.0, 90.0, 92.0, 94.0, 96.0, 98.0, 100.0, 102.0, 104.0, 106.0, 108.0])
            self.calib_in_energies = np.asarray([1e-6, 4.7e-6, 16.6e-6, 39.2e-6, 74.0e-6, 121.0e-6, 176.0e-6, 241.0e-6, 314.0e-6, 392.0e-6, 474.0e-6, 561.0e-6, 645.0e-6, 734.0e-6, 821.0e-6, 902.0e-6, 977.0e-6, 1050.0e-6, 1110.0e-6, 1160.0e-6, 1200.0e-6, 1230.0e-6, 1250.0e-6, 1230.0e-6, 1210.0e-6, 1160.0e-6, 1110.0e-6])
            self.calib_out_energies = np.asarray([1e-6, 1e-6, 4.9e-6, 12.4e-6, 22.9e-6, 37.3e-6, 55.7e-6, 73.5e-6, 97.3e-6, 122.0e-6, 148.0e-6, 173.0e-6, 200.0e-6, 232.0e-6, 251.0e-6, 276.0e-6, 306.0e-6, 331.0e-6, 348.0e-6, 355.0e-6, 372.0e-6, 378.0e-6, 389.0e-6, 395.0e-6, 393.0e-6, 380.0e-6, 373.0e-6, 362.0e-6, 351.0e-6])
            coupling = 0.3

            self.energies = coupling*(nlfo_math.deg_to_energy(self.calib_p_fit, scan_data["angles"]))

        elif energy_calibration == "july2020":
            # angle to energy calibration for data from 15 July 2020
            #p_fit = np.array([-1253.88647158, 1.99976171, -413.40024199, 1255.76066302])
            #angles = np.asarray([52.0, 54.0, 56.0, 58.0, 60.0, 62.0, 64.0, 66.0, 68.0, 70.0, 72.0, 74.0, 76.0, 78.0, 80.0, 82.0, 84.0, 86.0, 88.0, 90.0, 92.0, 94.0, 96.0, 98.0, 100.0, 102.0, 104.0, 106.0, 108.0])
            #in_energies = np.asarray([1e-6, 4.7e-6, 16.6e-6, 39.2e-6, 74.0e-6, 121.0e-6, 176.0e-6, 241.0e-6, 314.0e-6, 392.0e-6, 474.0e-6, 561.0e-6, 645.0e-6, 734.0e-6, 821.0e-6, 902.0e-6, 977.0e-6, 1050.0e-6, 1110.0e-6, 1160.0e-6, 1200.0e-6, 1230.0e-6, 1250.0e-6, 1230.0e-6, 1210.0e-6, 1160.0e-6, 1110.0e-6])
            #out_energies = np.asarray([1e-6, 1e-6, 4.9e-6, 12.4e-6, 22.9e-6, 37.3e-6, 55.7e-6, 73.5e-6, 97.3e-6, 122.0e-6, 148.0e-6, 173.0e-6, 200.0e-6, 232.0e-6, 251.0e-6, 276.0e-6, 306.0e-6, 331.0e-6, 348.0e-6, 355.0e-6, 372.0e-6, 378.0e-6, 389.0e-6, 395.0e-6, 393.0e-6, 380.0e-6, 373.0e-6, 362.0e-6, 351.0e-6])
            
            #self.energies = nlfo_math.deg_to_energy(p_fit, scan_data["angles"])
            print("add the data and calculate the Malus fit")

        self.soliton_order = np.asarray([nlfo_functions.Nsol(energy*1e-6, stats["pressure"], stats["gas"], stats["τfwhm"], stats["pulse_shape"], stats["λpump"], stats["core_radius"]) for energy in self.energies])

    def subtract_baseline(self):
        self.nobaseline_scan = np.zeros(self.scan.shape)
        for angle in range(1, self.scan.shape[1]):
            slice_nobaseline = peakutils.baseline(self.scan[:, angle], 1)
            self.nobaseline_scan[:, angle] = self.scan[:, angle] - slice_nobaseline

    def calibrate(self):
        if self.type == "vuv":
            self.calibrated_scan = np.asarray([plotting.calib(self.λ, self.scan[:, spec_idx], calibration=self.type) for spec_idx in range(0, self.scan.shape[1])])
        elif self.type == "vis":
            self.raw_λ = self.λ
            self.λ = self.λ[11:]
            self.calibrated_scan = np.asarray([plotting.calib(self.λ, self.scan[11:, spec_idx], calibration=self.type) for spec_idx in range(0, self.scan.shape[1])])

    def process_scan(self, subtract_baseline=True):
        # process experimentally measured scan by subtracting the background and calibrating it
        self.processed_scan = np.zeros(self.scan.shape)
        for angle in range(1, self.scan.shape[1]):
            if subtract_baseline:
                slice_nobaseline = peakutils.baseline(self.scan[:, angle], 1)
                self.processed_scan[:, angle] = np.asarray(plotting.calib(self.λ, (self.scan[:, angle] - slice_nobaseline), calibration=self.type))
            self.processed_scan[:, angle] = np.asarray(plotting.calib(self.λ, (self.scan[:, angle]), calibration=self.type))

    def find_peaks(self, angle_rdw=0):
        if self.processed_scan.any():
            pass
        else:
            self.process_scan()

        scan_peak_idx = np.zeros(self.scan.shape[1], dtype=int)
        self.scan_peaks = np.zeros(self.scan.shape[1])

        for angle in range(angle_rdw, self.scan.shape[1]):
            scan_peak_idx[angle] = peakutils.indexes(self.processed_scan[:, angle], thres=0.8, min_dist=30)
            self.scan_peaks[angle] = self.λ[scan_peak_idx[angle]]

    def integrate_scan(self):
        if self.processed_scan.any():
            pass
        else:
            self.process_scan()

        self.scan_integral = [integrate.simps(self.processed_scan[:, angle], self.λ) for angle in range(0, self.scan.shape[1])]


class SimScan(object):

    pass
