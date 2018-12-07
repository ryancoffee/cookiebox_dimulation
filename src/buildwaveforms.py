#!/usr/bin/python3

import glob
import re
import numpy as np
import numpy.random as rand
from numpy.fft import fft as FFT
from numpy.fft import ifft as IFFT
from numpy.fft import fftfreq as FREQ

from cmath import rect
nprect = np.vectorize(rect)

def Weiner(f,s,n,cut,p):
    w=np.zeros(f.shape[0])
    #print(w.shape)
    p = int(4)
    inds = [i for i,nu in enumerate(f) if abs(nu)<cut]
    w[inds] = s*np.power(np.cos(np.pi/2. * f[inds] / cut) , p)
    return w/(w+n)

def fourier_delay(f,v,dt):
    ## int f(t) exp(i*w*t) dt
    ## int f(t+tau) exp(i*w*t) dt --> int f(t)exp(i*w*t)exp(-i*w*tau) dt
    ## IFFT{ F(w) exp(-i*w*tau) }
    phase = nprect(np.ones(v.shape).T,-f*2.*np.pi*dt).T
    return v * phase


def main():
    #filepath = '/data/projects/slac/hamamatsu/dec2018/ave1/'
    filepath = '../data_fs/ave1/'
    filematch = filepath + 'C1--LowPulseHighRes-in-100-out1700-an2100--*.txt'
    filelist = glob.glob(filematch)

    for f in filelist[:10]:
        m = re.search('(.+).txt$',f)
        fi = open(f, "r")
        outname_spect = m.group(1) + '.spect.dat'
        outname_time = m.group(1) + '.time.dat'
        for i in range(6):
            headline = '# ' + fi.readline()
        (t,v) = fi.readline().split()
        v_vec=np.array(float(v),dtype=float)
        t_vec=np.array(float(t)*1.e9,dtype=float)
        for line in fi:
            (t,v) = line.split()
            v_vec = np.row_stack((v_vec,float(v)))
            t_vec = np.row_stack((t_vec,float(t)*1.e9))
        fi.close()
        #Get the mean time-step for sake of frequencies
        dt = np.mean(np.diff(t_vec,n=1,axis=0))
        #FFT the vector
        v_vec_ft = FFT(v_vec,axis=0)
        f = FREQ(v_vec_ft.shape[0],dt)
        # deep copy for the noise extimation 
        n_vec_ft = np.copy(v_vec_ft)
        # find indices where there is only noise in the power, and indices with predominantly signal
        # replace the signal elements in the noise vector with a random sampling from the noise portion
        chooseinds = np.array([i for i,nu in enumerate(f) if (abs(nu)> 6.5 and abs(nu)<(20))])
        replaceinds = np.array([i for i,nu in enumerate(f) if abs(nu)< 6.5])
        values = rand.choice(n_vec_ft[chooseinds,0],len(replaceinds))
        n_vec_ft[replaceinds,0] = values


        noiseamp = np.power(np.mean(np.abs(values)),int(2))
        sigamp = np.power(np.mean(np.array([i for i,nu in enumerate(f) if abs(nu)< 3.5])),int(2))
        vout = np.copy(v_vec_ft)
        vout[:,0] *= Weiner(f,sigamp,noiseamp,cut = 5,p = 4)

        spect = np.abs(v_vec_ft)
        out = np.column_stack((f,spect,np.abs(n_vec_ft),np.abs(vout)))
        np.savetxt(outname_spect,out,fmt='%.4f')
        v_copy = np.copy(vout)
        v_copy = fourier_delay(f,v_copy,10)
        v_back = np.real(IFFT(v_vec_ft,axis=0))
        v_filter_back = np.real(IFFT(vout,axis=0))
        v_filter_copy_back = np.real(IFFT(v_copy+n_vec_ft,axis=0))
        out = np.column_stack((t_vec,v_back,v_filter_back,v_filter_copy_back))
        np.savetxt(outname_time,out,fmt='%.4f')

    return 0

if __name__ == '__main__':
    main()
