import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import join, isfile

from src import SXanes, Config

plt.ion()
cf = Config()

sample_files = [f for f in listdir(cf.data_dir) if isfile(join(cf.data_dir, f))]
sample_files = [f for f in sample_files if 's_cal' not in f.lower() and f.split('.')[-1] == 'dat']

#sample = SXanes('MND_Troi_Line_001_001.dat')
#sample = SXanes('Bar_4757_008_001.dat')
#sample = SXanes('gyp4012_001_001.dat')
sample = SXanes('J820_R3_Chip3_007_001.dat')
sample.plot_sample()
