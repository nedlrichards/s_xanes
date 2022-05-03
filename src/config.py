import numpy as np

class Config():
    """Standard information"""
    def __init__(self):
        self.data_dir = './data/raw/July2021_SSRL/ssrl_data/bl143_jul21/xas'
        # channel numbers fixed by SSRL
        self.norm_channel = 3
        self.channel_offset = 7
        self.detector_chans = np.arange(1, 8)
