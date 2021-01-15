#!/usr/bin/python3

import numpy as np
import sys
import h5py
import re


def main():
    imnums = [0]
    if len(sys.argv)<2:
        print('syntax:\t%s <h5file> <optional image numbers (space separated)>'%sys.argv[0])
        return
    if len(sys.argv)>2:
        imnums = [int(i) for i in sys.argv[2:]]
    fname = sys.argv[1]

    f = h5py.File(fname,'r')
    for i in imnums:
        h = np.zeros((1,1),dtype=int)
        pulses = [p for p in list(f['img%05i'%i].keys()) if re.match('^pulse\d+',p)]
        for pkey in pulses:
            if h.shape[0]==1:
                h = np.zeros(f['img%05i'%i][pkey]['hist'][()].shape,dtype=int)
            h += f['img%05i'%i][pkey]['hist'][()]
        np.savetxt('%s.img%05i'%(fname,i),h,fmt='%i')
    f.close()

    return

if __name__ == '__main__':
    main()
