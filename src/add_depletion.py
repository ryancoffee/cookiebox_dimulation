#!/usr/bin/python3.5

import numpy as np
import sys
import re


def main():
    filename = 'data_fs/raw/CookieBox_waveforms.4pulses.image0920.dat'
    outfilename = filename + '.integral.out'
    sumfilename = filename + '.sum.out'
    if len(sys.argv)>1:
        filename = sys.argv[1]
    m = re.search('^(.*CookieBox_waveforms\.()pulses\.image(\d+))\.dat',filename)
    if m:
        print('working image {} with {} pulses'.format(m.group(3),int(m.group(2))))
        outfilename = m.group(1) + '.integral.out'
        sumfilename = m.group(1) + '.sum.out'
    data = np.loadtxt(filename,dtype=float)
    dataZeros = np.zeros((data.shape[0],data.shape[1]*4),dtype=float)
    dataZeros[:,:data.shape[1]]=data
    print('{} = data.shape\t{} = dataZeros.shape'.format(data.shape,dataZeros.shape))
    a = -2e-4
    #integral = [a/f for f in np.fft.fftfreq(data.shape[1]) if abs(f)>0 else 0.]
    integkernel = [a/(1.e-6+np.fft.fftfreq(dataZeros.shape[1]))]*data.shape[0]
    IdataFT = np.fft.fft(dataZeros,axis=1) * 1j * integkernel
    Idata = np.fft.ifft(IdataFT,axis=1).real
    for i in range(Idata.shape[0]):
        Idata[i,:] -= max(Idata[i,:])

    IdataOut = Idata[:,:data.shape[1]]
    gain = np.ones(data.shape,dtype=float) + IdataOut    

    np.savetxt(outfilename,IdataOut,fmt='%.4f')

    np.savetxt(sumfilename,IdataOut+data*gain,fmt='%.4f')

    return

if __name__ == '__main__':
    main()
