import h5py
import os
import pims
import numpy as np
import argparse
from coarse_grain import *

parser = argparse.ArgumentParser()
parser.add_argument('--loadpath', '-load', type=str,
                    help='absolute path to hdf5 file with alignment data')

args = parser.parse_args()

with h5py.File(args.loadpath) as f:
    winsize = f['params']['winsize'][()]
    overlap = f['params']['overlap'][()]
    imgpath = os.path.join(f['params']['loadpath'][()].split('WS')[0], f['params']['imgfile'][()])

    imgs = pims.open(imgpath)[0]

    cgim = np.array([coarse_grain(img, winsize, overlap) for img in imgs])
    d = f['data'].create_dataset('cgim', data=cgim)
    d.attrs['description'] = 'coarse grained image from imgpath'
    d.attrs['imgpath'] = imgpath
