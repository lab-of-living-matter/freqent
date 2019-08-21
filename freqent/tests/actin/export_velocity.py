import numpy as np
import argparse
import h5py
from scipy.io import loadmat
import os


parser = argparse.ArgumentParser()
parser.add_argument('--loadpath', '-load', type=str,
                    help='Absolute path to .mat file containing data.')
parser.add_argument('--savepath', '-save', type=str,
                    help='Path to save extracted data in hd5f file')

args = parser.parse_args()
expname = args.loadpath.split('WS')[0].split(os.path.sep)[-2]

try:
    # try loadmat in case .mat version < v7.3.
    # f is a dict. idk why I have to dig so deep into the numpy arrays
    # before I get to actual data, was found using trial and error
    f = loadmat(args.loadpath)
    imgfile = str(f['expt']['filename'][0][0][0])  # name of image file analyzed
    winsize = int(f['alignspec']['winsize'][0][0][0][0])  # size of window
    overlap = f['alignspec']['overlap'][0][0][0][0]  # fraction overlap between windows
    dt = f['expt']['tscl'][0][0][0][0]  # seconds per frame
    dx = f['expt']['dscl'][0][0][0][0]  # microns per pixel
    ordermat = np.array([m for m in f['FFTAlignmentData']['ordermat'][0]], dtype=float)  # matrix of order parameter calculations over time
    anglemat = np.array([m for m in f['FFTAlignmentData']['anglemat'][0]], dtype=float)  # matrix of angle calculations over time

except NotImplementedError:
    with h5py.File(args.loadpath, 'r') as f:
        imgfile = str(''.join([chr(n) for n in f['expt']['filename']]))
        winsize = int(f['alignspec']['winsize'][0][0])
        overlap = f['alignspec']['overlap'][0][0]
        dt = f['expt']['tscl'][0][0]
        dx = f['expt']['dscl'][0][0]
        nt = len(f['FFTAlignmentData']['ordermat'])
        ordermat = np.array([f[f['FFTAlignmentData']['ordermat'][ii][0]] for ii in range(nt)])
        anglemat = np.array([f[f['FFTAlignmentData']['anglemat'][ii][0]] for ii in range(nt)])
else:
    ValueError('Could not open .mat file with loadmat or h5py')

dat = {'ordermat': ordermat,
       'anglemat': anglemat}
datattrs = {'ordermat': 'nematic order parameter over coarse-grained area',
            'anglemat': 'angle of nematic director wrt image x-axis'}

params = {'loadpath': args.loadpath,
          'imgfile': imgfile,
          'winsize': winsize,
          'overlap': overlap,
          'dt': dt,
          'dx': dx}
paramsattrs = {'loadpath': 'location of .mat file',
               'imgfile': 'image file used to get order and angle data',
               'winsize': 'size of window over which angle and order is calculated',
               'overlap': 'fraction overlap between windows',
               'dt': 'seconds per frames',
               'dx': 'microns per pixel'}

with h5py.File(os.path.join(args.savepath, expname + '.hdf5')) as f:
    datagrp = f.create_group('data')
    paramsgrp = f.create_group('params')

    for datname in dat.keys():
        d = datagrp.create_dataset(datname, data=dat[datname])
        d.attrs['description'] = datattrs[datname]

    for paramname in params.keys():
        p = paramsgrp.create_dataset(paramname, data=params[paramname])
        p.attrs['description'] = paramsattrs[paramname]
