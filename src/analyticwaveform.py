#!/usr/bin/python3

import numpy as np
import sys
import re   
import glob
from timeit import default_timer as timer

from utilities import gauss,sigmoid,highpass,lowpass

from deconvolve_test import gauss

def analogprocess_theory(invec,bwd=2.4e9,dt=1):
    f = np.fft.fftfreq(invec.shape[0],dt)/bwd
    s = np.copy(invec)
    S = np.fft.fft(s) 
    I = np.zeros(S.shape,dtype = complex)
    DS = np.zeros(S.shape,dtype = complex)
    IDS = np.zeros(S.shape,dtype = complex)
    DS = np.zeros(S.shape,dtype = complex)
    DDS = np.zeros(S.shape,dtype = complex)
    inds = np.where(np.abs(f) < 1.)
    c2 = np.zeros(S.shape,dtype = float)
    c2[inds] = np.power(np.cos(np.abs(f[inds])*np.pi/2.),int(2))
    fpow1 = np.zeros(S.shape,dtype = float)
    fpow2 = np.zeros(S.shape,dtype = float)
    fpowm1 = np.zeros(S.shape,dtype = float)
    fpow2 = np.power(f,int(2))
    fpowm1[1:] = np.power(np.abs(f[1:]),-1)
    FILT_I = np.zeros(S.shape,dtype = complex)
    FILT_IDS = np.zeros(S.shape,dtype = complex)
    FILT_DS = np.zeros(S.shape,dtype = complex)
    FILT_DDS = np.zeros(S.shape,dtype = complex)
    FILT_I[inds] = c2[inds]*fpowm1[inds] + 0j
    FILT_IDS[inds] = c2[inds] + 0j
    FILT_DS[inds] = 0. + 1j * f[inds]*c2[inds]
    FILT_DDS[inds] = -fpow2[inds] * c2[inds] 
    I[inds] = np.copy(S[inds]) * FILT_I[inds] 
    IDS[inds] = np.copy(S[inds]) * FILT_IDS[inds]
    DS[inds] = np.copy(S[inds]) * FILT_DS[inds] 
    DDS[inds] = np.copy(S[inds]) * FILT_DDS[inds]

    dds = np.fft.ifft(DDS).real
    ds = np.fft.ifft(DS).real
    ids = np.fft.ifft(IDS).real
    i = np.fft.ifft(I).real

    thresh = np.zeros(ids.shape,dtype = float)
    thresh = ids * dds
    t = np.arange(invec.shape[0])
    capacitor = np.exp(-t/(invec.shape[0]/5))
    sqrtitor = np.power((t+1)*5e-4,-0.5)
    #print(t.shape)
    #deltas = np.zeros(ids.shape,dtype = float)
    #inds = np.where(thresh < -5e-4)
    #deltas[inds] = np.abs(1./(ds[inds]))
    NUM = np.fft.fft(sqrtitor * thresh)*c2*gauss(f,0,.1)
    DENOM = np.fft.fft(thresh)*c2*gauss(f,0,.1)
    deltas = np.fft.ifft(NUM).real/np.fft.ifft(DENOM).real
    (h,bins) = np.histogram(deltas,2**8,range=(0.45,1.))

    filt_i = np.roll(np.fft.ifft(FILT_I),100)
    filt_ids = np.roll(np.fft.ifft(FILT_IDS),100)
    filt_ds = np.roll(np.fft.ifft(FILT_DS),100)
    filt_dds = np.roll(np.fft.ifft(FILT_DDS),100)

    return ( np.column_stack(( f , np.abs(S), np.abs(I), np.abs(IDS), np.abs(DS), np.abs(DDS) , invec, i, ids,ds,dds,thresh,deltas,filt_i.real,filt_ids.real,filt_ds.real,filt_dds.real,filt_i.imag,filt_ids.imag,filt_ds.imag,filt_dds.imag)) , h )

def analogprocess(invec,bwd=2.4e9,dt=1):
    f = np.fft.fftfreq(invec.shape[0],dt)/bwd
    s = np.copy(invec)
    S = np.fft.fft(s)
    I = np.zeros(S.shape,dtype = complex)
    DS = np.zeros(S.shape,dtype = complex)
    IDS = np.zeros(S.shape,dtype = complex)
    DS = np.zeros(S.shape,dtype = complex)
    DDS = np.zeros(S.shape,dtype = complex)
    inds = np.where(np.abs(f) < 1.)
    logscale = 5e3
    logf = np.zeros(S.shape,dtype = float)
    logf[1:] = np.log(np.abs(f[1:])*logscale)
    linds = np.where( logf < 0 ) 
    logf[linds] = 0.
    fpowm1p2 = np.zeros(S.shape,dtype = float)
    fpowm0p2 = np.zeros(S.shape,dtype = float)
    fpowm1p2[1:] = np.power(np.abs(f[1:]),-1.2).real
    fpowm0p2[1:] = np.power(np.abs(f[1:]),-0.2).real
    FILT_I = np.zeros(S.shape,dtype = complex)
    FILT_IDS = np.zeros(S.shape,dtype = complex)
    FILT_DS = np.zeros(S.shape,dtype = complex)
    FILT_DDS = np.zeros(S.shape,dtype = complex)
    FILT_I[inds] = np.cos(np.abs(f[inds])*np.pi/2.)*logf[inds]*fpowm1p2[inds] + 0.j
    FILT_IDS[inds] = np.cos(f[inds]*np.pi/2.)*logf[inds]*fpowm0p2[inds] + 0.j# already is leaving the 0th element zero via definition of logf and fpowm0p2
    FILT_DS[inds] = 0. + 1j * np.sin(f[inds]*np.pi) * np.cos(f[inds]*np.pi/2.) * logf[inds] * fpowm0p2[inds]
    FILT_DDS[inds] = -np.sin(f[inds]*np.pi) * np.cos(f[inds]*np.pi/2.) * logf[inds] * fpowm0p2[inds] * f[inds]  + 0.j
    I[inds] = np.copy(S[inds]) * FILT_I[inds] 
    IDS[inds] = np.copy(S[inds]) * FILT_IDS[inds]
    DS[inds] = np.copy(S[inds]) * FILT_DS[inds] 
    DDS[inds] = np.copy(S[inds]) * FILT_DDS[inds]
    dds = np.fft.ifft(DDS).real
    ds = np.fft.ifft(DS).real
    ids = np.fft.ifft(IDS).real
    i = np.fft.ifft(I).real

    filt_i = np.roll(np.fft.ifft(FILT_I),100)
    filt_ids = np.roll(np.fft.ifft(FILT_IDS),100)
    filt_ds = np.roll(np.fft.ifft(FILT_DS),100)
    filt_dds = np.roll(np.fft.ifft(FILT_DDS),100)

    thresh = np.zeros(ids.shape,dtype = float)
    thresh = ids * dds
    deltas = np.zeros(ids.shape,dtype = float)
    inds = np.where(thresh < -.1)
    deltas[inds] = np.abs(1./(ds[inds]))
    return np.column_stack(( f , np.abs(S), np.abs(I), np.abs(IDS), np.abs(DS), np.abs(DDS) , invec, i, ids,ds,dds,thresh,deltas,filt_i.real,filt_ids.real,filt_ds.real,filt_dds.real,filt_i.imag,filt_ids.imag,filt_ds.imag,filt_dds.imag))

def althomomorphic(invec,ir,bwd=2.4e9,dt=1.):
    f = np.fft.fftfreq(invec.shape[0],dt) 
    ir_roll = np.copy(ir)
    i = np.argmin(ir_roll)
    ir_roll = np.roll(ir_roll,-i)
    y = np.fft.ifft( np.fft.fft(np.copy(invec))*gauss(f,0,bwd) ).real
    ys = np.sign(y)
    #dy = np.fft.ifft( np.fft.fft(np.copy(y))*1j*f*gauss(f,0,bwd)  )
    #y = y.real + 1j*dy.real
    #ys = np.unwrap(np.angle(y))
    #ys[len(ys)//2:] = -(ys[len(ys)//2+1:1:-1])
    #ys_smooth = np.fft.ifft( np.fft.fft(ys) * gauss(f,0,bwd) )
    #return (ys,ys_smooth.real)
    ya = np.abs(y).astype(complex)
    lowlim = 1e-16
    inds = np.where(ya<lowlim)
    ya[inds] += 1j*1e-15
    yla = np.log(ya)
    #rs = np.unwrap(np.angle(ir_roll))
    #rs[len(rs)//2:] = -(rs[len(rs)//2+1:1:-1])
    #rs_smooth = np.fft.ifft( np.fft.fft(rs) * gauss(f,0,bwd) )
    rs = np.sign(ir_roll)
    ra = np.abs(ir_roll).astype(complex)
    inds = np.where(ra<lowlim)
    ra[inds] += 1j*1e-15
    rla = np.log(ra)
    Y = np.fft.fft(yla)
    Ylow = np.copy(Y)*gauss(f,0,bwd)
    Yhigh = np.copy(Y)*highpass(f,2*bwd,bwd/2)
    R = np.fft.fft(rla)
    Rlow = np.copy(R)*gauss(f,0,bwd)
    RES = (Ylow-Yhigh)*gauss(f,0,bwd)
    result = np.exp(np.fft.ifft(RES))*ys*rs
    return (result.real,result.imag)

def homomorphic(invec,ir,bwd=3.2e9,dt=1.):
    f = np.fft.fftfreq(invec.shape[0],dt) 
    ir_roll = np.copy(ir)
    i = np.argmin(ir_roll)
    ir_roll = np.roll(ir_roll,-i)
    y = np.copy(invec)
    Y = np.fft.fft(y)
    R = np.fft.fft(ir_roll)
    YA = np.abs(np.copy(Y))*gauss(f,0,bwd)
    YS = np.angle(Y)
    RA = np.abs(np.copy(R))*gauss(f,0,bwd)
    RS = np.angle(R)
    qya = np.fft.fft(YA)
    qra = np.fft.fft(RA)
    q = (qya - qra)
    Q = np.fft.ifft(q)
    result = np.fft.ifft(Q * np.exp(1j*(YS)))
    return (result.real,result.imag)

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

def main(runAve=False):
    filelist = sys.argv[1:]
    if runAve:
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
        for i in range(1,300):
            datafile = 'data_fs/ave1/C1--HighPulse-in-100-out1700-an2100--%05i.dat' % i
            d += np.loadtxt(datafile,usecols=(1,))
            outwave = np.column_stack((outwave,d))
            D=np.fft.fft(np.copy(d))
            out=np.column_stack((out,np.power(np.abs(D),int(2))))
        d /= 300.
        df = f[1]-f[0]
        #Dfilt= D*gauss(f,0,250*df)
        Dfilt= D*gauss(f,0,3.2e9)
        dfilt = np.fft.ifft(Dfilt).real
        naivedeconv = derivconv(f,np.copy(d_orig),dfilt)
        alternateconv = altconv(f,np.copy(d_orig),dfilt) * 1.5e-37
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

    timesnames = 'data_fs/raw/CookieBox_waveforms.times.dat'
    times = np.loadtxt(timesnames)*1e-9
    dt = times[1]-times[0]
    freqs = np.fft.fftfreq(len(times),dt)
    if runAve:
        dfiltfull = np.zeros(len(times),dtype=float)
        dfiltfull[:len(dfilt)] = np.copy(dfilt)

    for fname in filelist:
        m = re.match('(.+)raw/(CookieBox_waveforms.(\d+)pulses.image(\d+)).dat',fname)
        if m:
            npulses = int(m.group(3))
            image = int(m.group(4))
            waveformsnames = m.group(0) 
            waveforms = np.loadtxt(waveformsnames)
            WAVEFORMS = np.fft.fft(np.copy(waveforms),axis=1)
            waveforms_deconv = np.zeros(waveforms.shape,dtype=float)
            waveforms_homodeconv = np.zeros(waveforms.shape,dtype=float)
            waveforms_homodeconv_imag = np.zeros(waveforms.shape,dtype=float)
            c = 4
            outresult = analogprocess(waveforms[c,:],bwd=2.4e9,dt=dt)
            (theoryresult,h) = analogprocess_theory(waveforms[c,:],bwd=2.4e9,dt=dt)

            energiesfile = m.group(1) + 'raw/CookieBox_Energies.' + m.group(3) + 'pulses.image' + m.group(4) + '.dat'
            energies = np.loadtxt(energiesfile)
            (h2,bins) = np.histogram(np.sqrt(energies[c,:]),h.shape[0],range = (0,16))

# Now, build a histogram of the energies file CookieBox_Energies.4pulses.image101.dat with also 2**10 bins, then plot the two histograms against each other
# Plot them as you used to with the coincidence method, e.g. <h1 h2> / <h1><h2>
            for c in range(waveforms.shape[0]):
                #wf_filt = np.fft.ifft(WAVEFORMS[c,:] * gauss(freqs,0,3.2e9)).real
                if runAve:
                    waveforms_deconv[c,:] = altconv(freqs,waveforms[c,:],dfiltfull)*1e-36
                    (waveforms_homodeconv[c,:],waveforms_homodeconv_imag[c,:]) = homomorphic(waveforms[c,:],dfiltfull,2.4e9,dt)
            if runAve:
                outname = m.group(1)+'processed/'+m.group(2)+'.deconv.out'
                np.savetxt(outname,waveforms_deconv,fmt='%.4e') 
                outname = m.group(1)+'processed/'+m.group(2)+'.homodeconv.real.out'
                np.savetxt(outname,waveforms_homodeconv,fmt='%.4e') 
                outname = m.group(1)+'processed/'+m.group(2)+'.homodeconv.imag.out'
                np.savetxt(outname,waveforms_homodeconv_imag,fmt='%.4e') 
            outname = m.group(1)+'processed/'+m.group(2)+'.analogtheory.out'
            np.savetxt(outname,theoryresult,fmt='%.4e') 
            outname = m.group(1)+'processed/'+m.group(2)+'.analogtheory.hist'
            np.savetxt(outname,np.column_stack((h,h2)),fmt='%i') 
            #out = np.tile(h,(h2.shape[0],1))* np.tile(h2,(h.shape[0],1)).T
            #print(out)
            #out -= np.outer(h,h2)
            #print(out.shape)
            #outname = m.group(1)+'processed/'+m.group(2)+'.analogtheory.cormat'
            #np.savetxt(outname,out,fmt='%.3e') 
            outname = m.group(1)+'processed/'+m.group(2)+'.analogprocess.out'
            np.savetxt(outname,outresult,fmt='%.4e') 
            print('printed {}'.format(outname))
    return

if __name__ == '__main__':
    runAve = False
    start = timer()
    main(runAve)
    end = timer()
    print('Elapsed {} s'.format(end-start))
