import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from os import listdir
from os.path import join, isfile

from src import SXanes, Config

plt.ion()
cf = Config()

sample_files = [f for f in listdir(cf.data_dir) if isfile(join(cf.data_dir, f))]
sample_files = [f for f in sample_files if 's_cal' not in f.lower() and f.split('.')[-1] == 'dat']

def gaussian(x, amp, center, sigma):
    """Standard gaussian bell"""
    norm = amp / (sigma * np.sqrt(2  *np.pi))
    bell = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    return norm * bell

def sigmoid(x, center, scale):
    """shifted unit amplitude"""
    x_shift = x - center
    return 1 / (1 + np.exp(-scale * x_shift))

def model_fit(x, a1, c1, s1, a2, c2, s2, a3, c3, s3, c4, s4):
    g1 = gaussian(x, a1, c1, s1)
    g2 = gaussian(x, a2, c2, s2)
    g3 = gaussian(x, a3, c3, s3)
    s = sigmoid(x, c4, s4)
    return g1 + g2 + g3 + s

sample = SXanes('J820_R3_Chip3_007_001.dat')
sample.normalize_spectrum()
#sample.plot_sample()
guess = [0.4, 2469.4, 2,
         1.5, 2476.8, 5,
         1.4, 2482.3, 2,
         2472, 1]

popt, pcov = curve_fit(model_fit, sample.energy, sample.norm_spec, p0=guess)

max_i = 2491
max_i = sample.energy.max()
win_i = sample.energy < max_i
best_fit = model_fit(sample.energy[win_i], *popt)

fig, ax = plt.subplots()
ax.plot(sample.energy[win_i], sample.norm_spec[win_i])
ax.plot(sample.energy[win_i], best_fit)

