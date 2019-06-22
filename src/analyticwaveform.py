#!/usr/bin/python3.5

import numpy as np
import sys

from deconvolve_test import gauss

def main():
    x=np.arange(2000)
    g=gauss(x,20,10)
    
    datafile = 'data_fs/ave1/C1--HighPulse-in-100-out1700-an2100--00000.dat'
    d=np.loadtxt(datafile,usecols=(1,))
    d_orig = np.copy(d)
    D=np.fft.fft(d)
    out = np.power(np.abs(D),int(2))
    outwave = d_orig
    print(d)
    for i in range(1,300):
        datafile = 'data_fs/ave1/C1--HighPulse-in-100-out1700-an2100--%05i.dat' % i
        d += np.loadtxt(datafile,usecols=(1,))/float(i)
        outwave = np.column_stack((outwave,d))
        D=np.fft.fft(np.copy(d))
        out=np.column_stack((out,np.power(np.abs(D),int(2))))
    f = np.fft.fftfreq(len(D))
    df = f[1]-f[0]
    Dfilt= D*gauss(f,0,250*df)
    fftfilename = './data_fs/processed/powerspectrum.dat'
    np.savetxt(fftfilename,np.column_stack((out,np.power(np.abs(Dfilt),int(2)))),fmt='%.4f')
    backfilename = './data_fs/processed/signal.dat'
    np.savetxt(backfilename,np.column_stack((outwave,np.fft.ifft(Dfilt).real)),fmt='%.4f')

    filename = './data_fs/processed/analyticwaveform.dat'
    np.savetxt(filename,g,fmt='%.6f')
    return

if __name__ == '__main__':
    main()
