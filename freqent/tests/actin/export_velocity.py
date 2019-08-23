import numpy as np
import argparse
import h5py
from scipy.io import loadmat
import os
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel

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
    vxt = np.moveaxis(f['vxt'], -1, 0)  # time measured in last dimension, move to first
    vyt = np.moveaxis(f['vyt'], -1, 0)  # time measured in last dimension, move to first

    # interpolate to replace nans in velocities
    if np.isnan(np.sum(vxt)):
        kernel = Gaussian2DKernel(x_stddev=1)
        vxt = np.array([interpolate_replace_nans(vx, kernel) for vx in vxt])
        vyt = np.array([interpolate_replace_nans(vy, kernel) for vy in vyt])

    vt = np.array([np.sqrt(vx**2 + vy**2) for vx, vy in zip(vxt, vyt)])
    imgfile = str(f['expt']['filename'][0][0][0])
    imdist = int(f['pivspec']['imdist'][0][0][0][0])
    lastim = int(f['pivspec']['lastim'][0][0][0][0])
    winsize = int(f['pivspec']['winsize'][0][0][0][0])
    overlap = float(f['pivspec']['overlap'][0][0][0][0])
    method = str(f['pivspec']['method'][0][0][0])

except NotImplementedError:
    print('try h5py')
    # with h5py.File(args.loadpath, 'r') as f:
    #     imgfile = str(''.join([chr(n) for n in f['expt']['filename']]))
    #     winsize = int(f['alignspec']['winsize'][0][0])
    #     overlap = f['alignspec']['overlap'][0][0]
    #     dt = f['expt']['tscl'][0][0]
    #     dx = f['expt']['dscl'][0][0]
    #     nt = len(f['FFTAlignmentData']['ordermat'])
    #     ordermat = np.array([f[f['FFTAlignmentData']['ordermat'][ii][0]] for ii in range(nt)])
    #     anglemat = np.array([f[f['FFTAlignmentData']['anglemat'][ii][0]] for ii in range(nt)])
else:
    ValueError('Could not open .mat file with loadmat or h5py')

dat = {'vxt': vxt,
       'vyt': vyt,
       'vt': vt}
datattrs = {'vxt': 'velocity in x direction over time',
            'vyt': 'velocity in y direction over time',
            'vt': 'speed over time'}

params = {'pivloadpath': args.loadpath,
          'pivimgfile': imgfile,
          'pivimdist': imdist,
          'pivlastim': lastim,
          'pivwinsize': winsize,
          'pivoverlap': overlap,
          'pivmethod': method}
paramsattrs = {'pivloadpath': 'Absolute path to piv data loaded',
               'pivimgfile': 'File PIV was calculated for',
               'pivimdist': 'number of frames between which velocity is calculated',
               'pivlastim': 'last frame PIV is calculated for',
               'pivwinsize': 'size of window to calculate PIV over',
               'pivoverlap': 'overlap fraction between windows',
               'pivmethod': 'method used in FFTAlignment_Stack parameter'}

with h5py.File(args.savepath) as file:

    for datname in dat.keys():
        if '/data/{d}'.format(d=datname) not in file:
            d = file['data'].create_dataset(datname, data=dat[datname])
            d.attrs['description'] = datattrs[datname]
        else:
            dset = file['data'][datname]
            dset[...] = dat[datname]

    for paramname in params.keys():
        if '/params/{p}'.format(p=paramname) not in file:
            p = file['params'].create_dataset(paramname, data=params[paramname])
            p.attrs['description'] = paramsattrs[paramname]
        else:
            dset = file['params'][paramname]
            dset[...] = params[paramname]
