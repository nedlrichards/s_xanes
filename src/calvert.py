import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from os import listdir
from os.path import join, isfile
from datetime import datetime

from src import Config

cf = Config()
data_dir = cf.data_dir

class CalVert:
    """standardize the energy axis"""

    def __init__(self):
        """Load all calvert data and create a correction interpolator"""
        self.s_cal_files = []
        for f in listdir(data_dir):
            if not isfile(join(data_dir, f)):
                continue
            # parse file name to find calibration files, only use .dat files
            if 's_cal' in f.lower() and f.split('.')[-1] == 'dat':
                self.s_cal_files.append(f)

        # bad data
        bad_data = 's_cal_2021_0715_evening_001_002.dat'
        self.s_cal_files.remove(bad_data)

        # standards use I2 / I0
        self.request_eV = 1
        self.i2 = 4
        self.i0 = 2

        cal_time = []
        measured_eV = []

        for f in self.s_cal_files:
            _, _, _, peak_eV, timestamp = self.load_s_cal(f)
            cal_time.append(timestamp)
            measured_eV.append(peak_eV)

        self.cal_time = np.array(cal_time)
        self.measured_eV = np.array(measured_eV)

        # sort data by timestamp
        sorti = np.argsort(self.cal_time)
        self.cal_time = self.cal_time[sorti]
        self.measured_eV = self.measured_eV[sorti]

        # Relate measured eV to theoretical value, requested + delta = measured
        self.theory_eV = 2472.02
        self.delta_eV = self.theory_eV - self.measured_eV

        # Epoch is required for requested energy correction interpolator
        self._epoch = np.datetime64(50, 'Y')
        time_axis = (self.cal_time - self._epoch) / np.timedelta64(1, 'h')

        self._ev_ier = interp1d(time_axis, self.delta_eV)

    def correct_ev(self, dt):
        """Return eV correction from calvert"""
        return self._ev_ier((dt - self._epoch) / np.timedelta64(1, 'h'))

    # load data file
    def load_s_cal(self, f):
        """Load all relevant data from a calibration file"""
        all_data = np.loadtxt(join(data_dir, f), skiprows=56)
        data_ier = interp1d(all_data[:, self.request_eV],
                            all_data[:, self.i2] / all_data[:, self.i0],
                            kind=3)



        # upsample data
        upsample = 10
        x_up = np.linspace(all_data[0, self.request_eV],
                           all_data[-1, self.request_eV],
                           all_data.shape[0] * upsample)
        data_up = data_ier(x_up)

        # pick the first peak
        peak = (find_peaks(data_up, height=0.05)[0])[0]
        measured_eV = x_up[peak]

        # load timestamp
        with open(join(data_dir, f), 'r') as readfile:
            _ = readfile.readline()
            timestamp = readfile.readline()
        dt = datetime.strptime(timestamp[:-1], "%a %b %d %H:%M:%S %Y")
        timestamp = np.datetime64(dt)

        return all_data, x_up, data_up, measured_eV, timestamp

if __name__ == '__main__':
    plt.ion()

    scal = CalVert()

    fig, ax = plt.subplots()
    for f in scal.s_cal_files:
        all_data, x_up, data_up, peak_eV, timestamp = scal.load_s_cal(f)
        l = ax.plot(all_data[:, 1], all_data[:, 4] / all_data[:, 2])
        ax.plot(x_up, data_up, color=l[0].get_color())
        peaki = np.argmin(np.abs(x_up - peak_eV))
        ax.plot(peak_eV, data_up[peaki], '.', color=l[0].get_color())

    fig, ax = plt.subplots()
    ax.plot(scal.cal_time, scal.measured_eV, color='C0')
    ax.plot(scal.cal_time, scal.measured_eV, '.', color='C0')
