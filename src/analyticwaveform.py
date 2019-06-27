#!/usr/bin/python3.5

import numpy as np
import sys

from deconvolve_test import gauss

def altconv(f,y,ir,bwd=3.2e9):
    Y = np.fft.fft(np.copy(y))
    YFILT = Y * gauss(f,0,bwd) 
    yfilt = np.fft.ifft(YFILT).real
    ir_roll = np.copy(ir)
    i = np.argmin(ir_roll)
    ir_roll = np.roll(ir_roll,-i)
    FILT = np.fft.fft(ir_roll) * gauss(f,0,bwd) * np.power(1j*f,int(4))
    return yfilt * np.fft.ifft(YFILT * FILT).real #* 1.5e-37
    ## this is taking the derivative of both y and impulse response (ir) and doint the convolution via Fourier Y*IR*(-1j*f)**2
    #return y * np.fft.ifft(Y * IR*np.power(gauss(f,0,3.2e9),int(2))*(np.power(1j*f,int(4)))).real*1.5e-37
    #* 1.5e-37

def derivconv(f,y,ir):
    Y = np.fft.fft(np.copy(y))
    ir_roll = np.copy(ir)
    i = np.argmin(ir_roll)
    ir_roll = np.roll(ir_roll,-i)
    IR = np.fft.fft(ir_roll)
    ## this is taking the derivative of both y and impulse response (ir) and doint the convolution via Fourier Y*IR*(-1j*f)**2
    return np.fft.ifft(Y*gauss(f,0,3.2e9)*np.power(1j*f,int(2))).real * np.fft.ifft(Y*gauss(f,0,3.2e9)*IR*(np.power(f,int(2)))).real*2.1e-37

def deconv(f,y,ir):
    desire = gauss(f,0,6.4e9)
    Y = np.fft.fft(np.copy(y))
    filt = desire/np.fft.fft(ir)
    return np.fft.ifft(Y*filt).real

def main():
    x=np.arange(2000)
    g=gauss(x,20,10)

    datafile = 'data_fs/ave1/C1--HighPulse-in-100-out1700-an2100--00000.dat'
    t=np.loadtxt(datafile,usecols=(0,))
    f = np.fft.fftfreq(len(t),t[1]-t[0])
    d=np.loadtxt(datafile,usecols=(1,))
    d_orig = np.copy(d)
    D=np.fft.fft(d)
    out = np.power(np.abs(D),int(2))
    outwave = d_orig
    Dfilt= D*gauss(f,0,3.2e9)
    out = np.column_stack((out,np.power(np.abs(Dfilt),int(2))))
    outwave = np.column_stack((outwave,np.fft.ifft(Dfilt).real))
    N=0.4
    print(d)
    for i in range(1,300):
        datafile = 'data_fs/ave1/C1--HighPulse-in-100-out1700-an2100--%05i.dat' % i
        d += np.loadtxt(datafile,usecols=(1,))/float(i)
        outwave = np.column_stack((outwave,d))
        D=np.fft.fft(np.copy(d))
        out=np.column_stack((out,np.power(np.abs(D),int(2))))
    df = f[1]-f[0]
    #Dfilt= D*gauss(f,0,250*df)
    Dfilt= D*gauss(f,0,3.2e9)
    dfilt = np.fft.ifft(Dfilt).real
    naivedeconv = derivconv(f,np.copy(d_orig),dfilt)
    alternateconv = altconv(f,np.copy(d_orig),dfilt)
    Dfiltdiff = np.copy(Dfilt)*1j*f
    deriv_conv = np.fft.ifft(np.fft.fft(np.copy(d_orig))*1j*f*Dfiltdiff)
    deriv_conv = np.roll(deriv_conv,len(deriv_conv)//2-15)*6e-20
    dconvname = './data_fs/processed/derivconv.dat'
    np.savetxt(dconvname,np.column_stack((t*1e9,outwave[:,0].real,deriv_conv.real,naivedeconv,alternateconv)),fmt='%.4f')
    headstring = 'timestep = {}, N = {}\n#f[GHz]\t[dB]\t[dB]...'.format(t[1]-t[0],N)
    fftfilename = './data_fs/processed/powerspectrum.dat'
    np.savetxt(fftfilename,np.column_stack((f*1e-9,10.*np.log10(out/N),10*np.log10(np.power(np.abs(Dfilt),int(2))/N))),fmt='%.4f',header = headstring)
    backfilename = './data_fs/processed/signal.dat'
    headstring = 'timestep = {}, N = {}'.format(t[1]-t[0],N)
    np.savetxt(backfilename,np.column_stack((t*1e9,outwave,np.fft.ifft(Dfilt).real)),fmt='%.4f')

    filename = './data_fs/processed/analyticwaveform.dat'
    np.savetxt(filename,g,fmt='%.6f')
    print('Made it here')

    image = 4976
    waveformsnames = 'data_fs/raw/CookieBox_waveforms.4pulses.image%04i.dat' % image 
    timesnames = 'data_fs/raw/CookieBox_waveforms.times.dat'
    times = np.loadtxt(timesnames)
    freqs = np.fft.fftfreq(len(times),times[1]-times[0])
    dfiltfull = np.zeros(len(times),dtype=float)
    dfiltfull[:len(dfilt)] = np.copy(dfilt)
    waveforms = np.loadtxt(waveformsnames)
    waveforms_deconv = np.zeros(waveforms.shape,dtype=float)
    for c in range(waveforms.shape[0]):
        waveforms_deconv[c,:] = altconv(freqs,waveforms[c,:],dfiltfull)
    outname = './data_fs/processed/CookieBox_waveforms.4pulses.image%4i_deconv.dat' % image
    np.savetxt(outname,waveforms_deconv[0,:],fmt='%.4e') 
    return

if __name__ == '__main__':
    main()
