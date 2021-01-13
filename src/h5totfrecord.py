#!/usr/bin/python3

import numpy as np
import h5py
import sys
import re
import tensorflow as tf
from tensorflow import keras

def phase2id(phase):
    # careful, playing tricks with the 8 bit overflow in order to wrap the phase into 0..2pi and discritize into 0..15
    p = np.uint8(float(phase)/2./np.pi*256)
    return np.uint8(int(p)*16//256)

def energy2id(energy,e0,ewin):
    return np.uint8(float(energy-e0)*256/ewin)

def main():
    if len(sys.argv)<2:
        print('syntax: %s <h5 file to convert> <optional nshards/h5>'%sys.argv[0])
    h5filenames = [name for name in sys.argv[1:]]
    nshardsPfile = 4

    print(h5filenames)

    for fname in h5filenames:
        f = h5py.File(fname,'r')
        imkeys = [k for k in list(f.keys()) if re.match('^img\d+',k)]
        for k in imkeys:
            pulses = [p for p in list(f[k].keys()) if re.match('^pulse\d+',p)]
            carrier = f[k].attrs['carrier']
            for p in pulses:
                i = phase2id( carrier - f[k][p].attrs['phase'] )
                e = energy2id(f[k][p].attrs['energy'],450,128)
                x = f[k][p]['hist'][()]
                y = [i , e]

    return

if __name__=='__main__':
    main()
