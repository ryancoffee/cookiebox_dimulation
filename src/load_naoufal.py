#!/usr/bin/python3

import numpy as np
import h5py
import sys
import random
import math

def main():
    if len(sys.argv) < 2:
        print("syntax: %s <datafile.h5>"%(sys.argv[0]) )
        return
    for fname in sys.argv[1:]:
        f = h5py.File(fname,'r') #  IMORTANT NOTE: it looks like some h5py builds have a resouce lock that prevents easy parallelization... tread lightly and load files sequentially for now.
        print(list(f['sim_0']['pulse_0']['chan_0'].keys() ))
        X_t = []
        X_r = []
        X_q = []
        Y_a = []
        Y_e = []
        for sim in list(f.keys()):
            print(list(f[sim].keys()))
            for pulse in list(f[sim].keys()):
                for chan in list(f[sim][pulse].keys()):
                    tlist = list(f[sim][pulse][chan]['times'])
                    rlist = list(f[sim][pulse][chan]['r_detector'])
                    alist = list(f[sim][pulse][chan]['angle'])
                    elist = list(f[sim][pulse][chan]['energy'])
                    X_t += tlist
                    X_r += rlist
                    X_q += [2.*math.pi*random.random() for i in range(len(tlist))]
                    Y_a += alist
                    Y_e += elist
                    '''
                    X_t += [float(v) for i,v in enumerate(list(f[sim][pulse][chan]['times']))]
                    X_r += [float(v) for i,v in enumerate(list(f[sim][pulse][chan]['r_detector']))]
                    X_q += [2*np.pi*random.random() for i in ]
                    Y_a += [float(v) for i,v in enumerate(list(f[sim][pulse][chan]['angle']))]
                    Y_e += [float(v) for i,v in enumerate(list(f[sim][pulse][chan]['energy']))]
                    '''

        X_x = X_r*np.cos(X_q)
        X_y = X_r*np.sin(X_q)
        np.savetxt("%s.ascii"%(fname),np.column_stack((X_t,X_r,X_q,X_x,X_y,Y_a,Y_e)),fmt='%.3e')
    return

if __name__ == '__main__':
    main()

