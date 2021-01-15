#!/usr/bin/python3

import numpy as np
import sys
import h5py
import re

def phase2id(phase):
    # careful, playing tricks with the 8 bit overflow in order to wrap the phase into 0..2pi and discritize into 0..15
    p = np.uint8(float(phase)/2./np.pi*256)
    return np.uint8(int(p)*16//256)

def extract_XY(fname):
    f = h5py.File(fname,'r')
    X = []
    Y = []
    imkeys = [i for i in list(f.keys()) if re.match('^img\d+',i)]
    for imkey in imkeys:
        h = np.zeros((1,1),dtype=np.uint8)
        phases = []
        carrier = f[imkey].attrs['carrier']
        pulsekeys = [p for p in list(f[imkey].keys()) if re.match('^pulse\d+',p)]
        for pulsekey in pulsekeys:
            if h.shape[0]==1:
                h = np.zeros(f[imkey][pulsekey]['hist'][()].shape,dtype=np.uint8)
            h += f[imkey][pulsekey]['hist'][()]
            phases += [phase2id(carrier + f[imkey][pulsekey].attrs['phase'])]
        X += [h]
        Y += [phases]
    f.close()
    return X,Y


def main():
    imnums = [0]
    if len(sys.argv)<2:
        print('syntax:\t%s <h5files>'%sys.argv[0])
        return
    fnames = [n for n in sys.argv[1:]]

    for fname in fnames:
        X,Y = extract_XY(fname)
        print(fname)

        _ = [print(a.shape) for a in X]
        _ = [print(y) for y in Y]

    return

if __name__ == '__main__':
    main()
