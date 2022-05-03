import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from os import listdir, remove
from os.path import join, isfile
from datetime import datetime

from src import SXanes, Config

plt.ion()
cf = Config()

sample_files = [f for f in listdir(cf.data_dir) if isfile(join(cf.data_dir, f))]
sample_files = [f for f in sample_files if 's_cal' not in f.lower() and f.split('.')[-1] == 'dat']


def inspect_sample(file_name):
    sample = SXanes(file_name)

    dc = False
    while not dc:
        dc = drop_channel(sample)

    return sample

def drop_channel(sample):
    fig, ax = sample.plot_sample()
    print('Sample: {sample.sample_file}')
    print('Enter channel number to drop, or d to remove file:')
    x = input()
    if len(x) == 0:
        sample.to_pickle()
        return True
    elif x.isdigit():
        sample.drop_channel(int(x))
        plt.close(fig)
        return False
    elif x.isalpha() and str(x) == 'd':
        if os.isfile(sample.save_name):
            os.remove(sample.save_name)
        plt.close(fig)
        return True

    else:
        plt.close(fig)
        return False

for _ in range(10):
    s0 = inspect_sample(sample_files.pop(0))

1/0
sample.drop_channel(6)
fig, ax = sample.plot_sample()


Channels = ["Real Time Clock", "Requested Energy", "I0", "I1", "I2", "I3",
            "MPFB1", "MPFB2", "FF1", "FF2", "FF3", "FF4", "FF5", "FF6", "FF7"]

# load data file
def load_sample(f, drop_channels=None):
    """Load all relevant data from a calibration file"""
    detector_chans = np.arange(8, 15)

    if drop_channels is not None:
        drop_channels = np.array(drop_channels, ndmin=1)

        # use list operations to remove channels
        drop_channels = drop_channels.tolist()
        detector_chans = detector_chans.tolist()
        [detector_chans.pop(dc) for dc in drop_channels]
        detector_chans = np.array(detector_chans, ndmin=1)


    all_data = np.loadtxt(join(cf.data_dir, f), skiprows=56)

    # load timestamp
    with open(join(cf.data_dir, f), 'r') as readfile:
        _ = readfile.readline()
        timestamp = readfile.readline()
    dt = datetime.strptime(timestamp[:-1], "%a %b %d %H:%M:%S %Y")
    timestamp = np.datetime64(dt)

    energy = all_data[:, 1]
    # normalize by I1 (column 3)
    spectra = all_data[:, detector_chans] / all_data[:, 3:4]
    ff_chans = detector_chans - 7

    return energy, spectra, ff_chans, timestamp

f = sample_files[0]
for f in sample_files[:10]:
    #energy, spectra, channels, timestamp = load_sample(f, drop_channels=5)
    energy, spectra, channels, timestamp = load_sample(f)

    fig, ax = plt.subplots()

    for chan, i in zip(spectra.T, channels):
        ax.plot(energy, chan, label=f'FF{i}')
    ax.legend()
    xas = np.mean(spectra, axis=-1)
    ax.plot(energy, xas, 'k')

    ax.set_xlabel('Requested energy (eV)')
    ax.set_ylabel('Normalized counts (eV)')
