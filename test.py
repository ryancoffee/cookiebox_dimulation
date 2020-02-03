#!/usr/bin/python3

import numpy as np
import sys

def main():
    if len(sys.argv)<3:
        print('I need a t0 and rate')
        return
    t0 = float(sys.argv[1])
    t = np.arange(0,2e3,.1)
    y = np.sin(2*np.pi*(t-t0)/1e3)
    d = np.sin(2*np.pi*(t-t0)/1e3)>0
    D = np.fft.fft(d)
    nu = np.fft.fftfreq(d.shape[0],.1)
    rate = float(sys.argv[2])
    ramp = np.fft.ifft(D / (1 + rate*1j*nu)).real
    q = ramp*np.sign(-y)

    
    fname = 'temp.dat'
    np.savetxt(fname,np.column_stack((t,y,ramp,q)),fmt='%.4f')
    return

if __name__ == '__main__':
    main()
