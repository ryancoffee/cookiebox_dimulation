#!/usr/bin/python3.5

import numpy as np
import sys

def gauss(x,c,w):
    return np.exp(-np.power((x-c)/w,int(2)))

def sig(x,c,w,b):
    g = gauss(x,c,w)
    dg = -2*(x-c)/w*gauss(x,c,w)
    return g -b*dg

def main():
    nhits = 10
    if 1<len(sys.argv):
        nhits = int(sys.argv[1])
    x = np.arange(667,dtype=int)
    n = np.random.sample(len(x)+1)
    dn = n[1:]-n[0:-1]
    f = np.fft.fftfreq(x.shape[0])
    F = gauss(f,0,1./5.)
    inds = np.random.choice(x,nhits)
    y = np.zeros(x.shape,dtype=float)
    y[inds] = [np.random.sample(len(inds))]
    s = sig(x,20,3,-2)
    S = np.fft.fft(s)
    DS = 1j*f*S
    SDS = S+DS
    Y = np.fft.fft(y)
    nscale=1e-9
    yg = np.fft.ifft(Y*(SDS))+nscale*n[0:-1]
    YG = np.fft.fft(yg.real+nscale*1j*dn)
    yd = np.fft.ifft(YG/(SDS))

    np.savetxt('data_fs/processed/deconvolve.out',np.column_stack((x,y,yg.real,yd.real,yg.imag,yd.imag)),fmt='%.6f')
    return

if __name__ == '__main__':
    main()
