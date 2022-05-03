import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from os import listdir, mkdir
from os.path import join, isfile, isdir
from datetime import datetime
import pickle

from src import CalVert, Config

s_cal = CalVert()
cf = Config()

class SXanes:
    """Load and process a single S Xanes spectrum"""

    def __init__(self, sample_file, load_pickle=False):
        """startup processing"""
        self.sample_file = sample_file

        self.pickledir = join(cf.data_dir, "pickles")
        save_name = join(self.pickledir, self.sample_file)[:-4]
        self.save_name = save_name + '.pic'

        # normalization parameters
        self.xlead_upper = None
        self.xtail_lower = None
        self.fit_point = None

        if load_pickle:
            load_state = pickle.load(self.save_name)
            self.drop_channels = load_state.drop_channels
        else:
            self.drop_channels = None

        self.energy = None
        self.sum_spec = None
        self.norm_spec = None
        self.spectra = None
        self.ff_chans = None
        self.timestamp = None
        self.xaxis_correction = None

        self.load_dat_file()

    def drop_channel(self, channel_number):
        """drop a channel from sum"""
        if self.drop_channels is None:
            self.drop_channels = [channel_number]
        elif channel_number not in self.drop_channels:
            self.drop_channels += [channel_number]

    def normalize_spectrum(self, xlead_upper=2468., xtail_lower=2490.,
                           xtail_order=1, fit_point=None):
        """
        Fit polynomials to leading and tailing edge, normalize spectrum
        """

        leadi = self.energy <= xlead_upper
        taili = self.energy >= xtail_lower

        plead = np.polyfit(self.energy[leadi], self.sum_spec[leadi], 1)
        ptail = np.polyfit(self.energy[taili], self.sum_spec[taili], xtail_order)

        self.xlead_upper = xlead_upper
        self.xtail_lower = xtail_lower

        # return polynomial fits
        lead = np.polyval(plead, self.energy)
        tail = np.polyval(ptail, self.energy)

        if fit_point is None:
            # compute fit point as maximum derivative
            dx = np.diff(self.energy)
            diff_points = np.abs((dx - dx[0]) < 1e-5)
            y = savgol_filter(self.sum_spec[:-1][diff_points],
                              15, 2, deriv=1, delta=dx[0])
            self.fit_point = self.energy[:-1][diff_points][np.argmax(y)]
        else:
            self.fit_point = fit_point

        lead_val = np.polyval(plead, self.fit_point)
        tail_val = np.polyval(ptail, self.fit_point)
        val_diff = tail_val - lead_val

        self.norm_spec = self.sum_spec - lead

        norm_i = self.energy > self.fit_point
        self.norm_spec[norm_i] -= np.polyval(ptail, self.energy[norm_i])
        self.norm_spec[norm_i] += tail_val
        self.norm_spec /= val_diff

        return lead, tail


    def load_dat_file(self):
        """Load all relevant data from a calibration file"""
        if self.drop_channels is not None:
            drop_channels = np.array(self.drop_channels, ndmin=1)
            # use list operations to remove channels
            drop_channels = drop_channels.tolist()
            detector_chans = cf.detector_chans.tolist()
            [detector_chans.remove(dc) for dc in drop_channels]
            detector_chans = np.array(detector_chans, ndmin=1)
        else:
            detector_chans = cf.detector_chans

        all_data = np.loadtxt(join(cf.data_dir, self.sample_file), skiprows=56)

        # load timestamp
        with open(join(cf.data_dir, self.sample_file), 'r') as readfile:
            _ = readfile.readline()
            timestamp = readfile.readline()
        dt = datetime.strptime(timestamp[:-1], "%a %b %d %H:%M:%S %Y")

        self.timestamp = np.datetime64(dt)
        self.xaxis_correction = s_cal.correct_ev(self.timestamp)
        self.energy = all_data[:, 1] + self.xaxis_correction

        # normalize by I1 (column 3)
        nd = all_data[:, cf.norm_channel][None, :]

        spectra = np.array([all_data[:, i + cf.channel_offset] for i in detector_chans])
        self.spectra = spectra / nd
        self.sum_spec = self.spectra.sum(axis=0)
        self.ff_chans = detector_chans

        return


    def plot_sample(self):
        """Standard plot of processing products"""
        self.load_dat_file()
        lead, tail = self.normalize_spectrum()
        fig, axes = plt.subplots(2, 2, sharey=False, figsize=(6.5, 6))

        for chan, i in zip(self.spectra, self.ff_chans):
            axes[0, 0].plot(self.energy, chan, label=f'FF{i}')
        axes[0, 0].legend()

        axes[0, 0].set_xlabel('Energy (eV)')
        axes[0, 0].set_ylabel('Normalized counts (eV)')

        axes[0, 1].plot(self.energy, self.sum_spec, 'k')
        axes[0, 1].plot(self.energy, lead, 'C0')
        axes[0, 1].plot(self.energy, tail, 'C1')

        # plot fit point
        fy = self.sum_spec[np.argmin(np.abs(self.fit_point - self.energy))]
        axes[0, 1].plot(self.fit_point, fy, 'o', color='C2')

        axes[1, 0].plot(self.energy, self.norm_spec, 'k')


        axes[1, 1].plot(self.energy, self.norm_spec, 'k')
        axes[1, 1].set_xlim(2466.5, 2487)

        return fig, axes


    def to_pickle(self):
        """Pickle processing"""

        if not isdir(self.pickledir): mkdir(self.pickledir)
        with open(self.save_name, 'wb') as f:
            pickle.dump(self, f)
