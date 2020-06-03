#!/usr/bin/python3

import numpy as np
import h5py
import sys

def main():
    if len(sys.argv) < 2:
        print("syntax: %s <datafile.h5>"%(sys.argv[0]) )
        return
    for fname in sys.argv[1:]:
        f = h5py.File(fname,'r') #  IMORTANT NOTE: it looks like some h5py builds have a resouce lock that prevents easy parallelization... tread lightly and load files sequentially for now.
        print(list(f['sim_0']['pulse_0']['chan_0'].keys() ))
        X_t = []
        X_r = []
        Y_a = []
        Y_e = []
        for sim in list(f.keys()):
            print(list(f[sim].keys()))
            for pulse in list(f[sim].keys()):
                for chan in list(f[sim][pulse].keys()):
                    X_t += [float(v) for i,v in enumerate(list(f[sim][pulse][chan]['times']))]
                    X_r += [float(v) for i,v in enumerate(list(f[sim][pulse][chan]['r_detector']))]
                    Y_a += [float(v) for i,v in enumerate(list(f[sim][pulse][chan]['angle']))]
                    Y_e += [float(v) for i,v in enumerate(list(f[sim][pulse][chan]['energy']))]
        np.savetxt("%s.ascii"%(fname),np.column_stack((X_t,X_r,Y_a,Y_e)),fmt='%.3e')
    return

if __name__ == '__main__':
    main()

