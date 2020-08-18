import os
import sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('files', type=str, nargs='+',
                    help='files to calculate entropy for')
args = parser.parse_args()

mean_syncParam = np.zeros(len(args.files))
std_syncParam = np.zeros(len(args.files))
mus = np.zeros(len(args.files))

for file_index, file in enumerate(args.files):
    mus[file_index] = float(file.split(os.path.sep)[-2][2:-7])
    with h5py.File(file) as d:
        mean_syncParam[file_index] = d['data']['syncParam'][()].mean()
        std_syncParam[file_index] = d['data']['syncParam'][()].std()
