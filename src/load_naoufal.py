#!/usr/bin/python3

import numpy as np
import h5py
import sys
import random
import math

def NaoufalToLorenzo(r):
    '''
    Lorenzo, e.g. Tixel detector, has 48x48 pixels per tile, each pixel is 100 microns square, r is in meters from Ave/Naoufal
    '''
    q = [2.*math.pi*random.random() for i in range(len(r))]
    return (np.array(q),1e4*np.array(r)*np.cos(q),1e4*np.array(r)*np.sin(q))

def main():
    if len(sys.argv) < 2:
        print("syntax: %s <datafile.h5>"%(sys.argv[0]) )
        return
    for fname in sys.argv[1:]:
        f = h5py.File(fname,'r') #  IMORTANT NOTE: it looks like some h5py builds have a resouce lock that prevents easy parallelization... tread lightly and load files sequentially for now.
        print('list(f[\'sim_0\'][\'chan_0\'][\'pulse_0\'].keys()',list(f['sim_0']['chan_0']['pulse_0'].keys() ))
        X_t = []
        X_r = []
        X_q = []
        Y_a = []
        Y_e = []
        for sim in list(f.keys()):  #[:10]:
            for chan in list(f[sim].keys()):
                for pulse in list(f[sim][chan].keys()):
                    tlist = list(f[sim][chan][pulse]['times'])
                    rlist = list(f[sim][chan][pulse]['r_detector'])
                    alist = list(f[sim][chan][pulse]['angle'])
                    elist = list(f[sim][chan][pulse]['energy'])
                    X_t += tlist
                    X_r += rlist
                    Y_a += alist
                    Y_e += elist
        X_q,X_x,X_y = NaoufalToLorenzo(X_r)
        np.savetxt("%s.ascii"%(fname),np.column_stack((X_t,X_r,X_q,X_x,X_y,Y_a,Y_e)),fmt='%.3e')
    return

if __name__ == '__main__':
    main()

