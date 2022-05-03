import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from os import listdir
from os.path import join, isfile
from datetime import datetime

from src import Config, CalVert

plt.ion()
cf = Config()

s_cal_files = [f for f in listdir(cf.data_dir) if isfile(join(cf.data_dir, f))]
s_cal_files = [f for f in s_cal_files if 's_cal' in f.lower() and f.split('.')[-1] == 'dat']

# bad data
bad_data = 's_cal_2021_0715_evening_001_002.dat'

s_cal_files.remove(bad_data)

# standards use I2 / I0

cv = CalVert()

# load data file
def load_s_cal(f):
    """Load all relevant data from a calibration file"""
    all_data = np.loadtxt(join(cf.data_dir, f), skiprows=56)
    # upsample by 10
    data_ier = interp1d(all_data[:, 1], all_data[:, 4] / all_data[:, 2], kind=3)
    x_up = np.linspace(all_data[0, 1], all_data[-1, 1], all_data.shape[0] * 10)
    data_up = data_ier(x_up)

    peaks = find_peaks(data_up, height=0.05)


    # load timestamp
    with open(join(cf.data_dir, f), 'r') as readfile:
        _ = readfile.readline()
        timestamp = readfile.readline()
    dt = datetime.strptime(timestamp[:-1], "%a %b %d %H:%M:%S %Y")
    timestamp = np.datetime64(dt)

    return all_data, x_up, data_up, peaks[0], timestamp

fig, ax = plt.subplots()
for f in s_cal_files:
    all_data, x_up, data_up, peaks, timestamp = load_s_cal(f)
    l = ax.plot(all_data[:, 1], all_data[:, 4] / all_data[:, 2])
    ax.plot(x_up, data_up, color=l[0].get_color())
    ax.plot(x_up[peaks[0]], data_up[peaks[0]], '.', color=l[0].get_color())

fig, ax = plt.subplots()
for f in s_cal_files:
    all_data, x_up, data_up, peaks, timestamp = load_s_cal(f)
    xaxis_correction = cv.correct_ev(timestamp)
    l = ax.plot(all_data[:, 1]  + xaxis_correction, all_data[:, 4] / all_data[:, 2])

