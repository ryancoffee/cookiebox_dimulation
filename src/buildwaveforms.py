#!/usr/bin/python3

import glob
import re
import numpy as np
from numpy.random import choice, shuffle
from numpy.fft import fft as FFT
from numpy.fft import ifft as IFFT
from numpy.fft import fftfreq as FREQ
from scipy.constants import c

from generate_distribution import fillcollection

from cmath import rect
nprect = np.vectorize(rect)

def energy2time(e,r=0,d1=5,d2=5,d3=30):
    #distances are in centimiters and energies are in eV and times are in ns
    C_cmPns = c*100.*1e-9
    mc2 = float(0.511e6)
    t = 1.e3 + np.zeros(e.shape,dtype=float);
    if r==0:
        t = (d1+d2+d3)/C_cmPns * np.sqrt(mc2/(2.*e))
        return t
    inds = np.argwhere(e>r)
    t[inds] = d1/C_cmPns * np.sqrt(mc2/(2.*e[inds]))
    t[inds] += d3/C_cmPns * np.sqrt(mc2/(2.*(e[inds]-r)))
    t[inds] += d2/C_cmPns * np.sqrt(2)*(mc2/r)*(np.sqrt(e[inds]/mc2) - np.sqrt((e[inds]-r)/mc2))
    return t

def Weiner(f,s,n,cut,p):
    w=np.zeros(f.shape[0])
    #print(w.shape)
    p = int(4)
    inds = [i for i,nu in enumerate(f) if np.abs(nu)<cut]
    w[inds] = s*np.power(np.cos(np.pi/2. * f[inds] / cut) , p)
    return w/(w+n)

def fourier_delay(f,dt):
    #print('f.shape = ',f.shape)
    ## int f(t) exp(i*w*t) dt
    ## int f(t+tau) exp(i*w*t) dt --> int f(t)exp(i*w*t)exp(-i*w*tau) dt
    ## IFFT{ F(w) exp(-i*w*tau) }
    return nprect(np.ones(f.shape),-f*2.*np.pi*dt)

def main():
    #filepath = '/data/projects/slac/hamamatsu/dec2018/ave1/'
    filepath = '../data_fs/ave1/'
    filematch = filepath + 'C1--LowPulseHighRes-in-100-out1700-an2100--*.txt'
    filelist = glob.glob(filematch)

    collection = np.array((0,1,2),dtype=float)
    for f in filelist[:10]:

        ## processing images 
        m = re.search('(.+).txt$',f)
        fi = open(f, "r")
        outname_spect = m.group(1) + '.spect.dat'
        outname_time = m.group(1) + '.time.dat'
        outname_simTOF = m.group(1) + '.simTOF.dat'
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
        m_extend = 10
        f_extend = FREQ(v_vec_ft.shape[0]*m_extend,dt)
        t_extend = np.arange(0,((t_vec[-1]-t_vec[0])+dt)*m_extend,dt)
        # deep copy for the noise extimation 
        n_vec_ft = np.copy(v_vec_ft)
        # find indices where there is only noise in the power, and indices with predominantly signal
        # replace the signal elements in the noise vector with a random sampling from the noise portion
        chooseinds = np.array([i for i,nu in enumerate(f) if (np.abs(nu)> 6.5 and np.abs(nu)<(20))])
        replaceinds = np.array([i for i,nu in enumerate(f) if np.abs(nu)< 6.5])
        values = choice(n_vec_ft[chooseinds,0],len(replaceinds))
        n_vec_ft[replaceinds,0] = values

        noiseamp = np.power(np.mean(np.abs(values)),int(2))
        sigamp = np.power(np.mean(np.array([i for i,nu in enumerate(f) if np.abs(nu)< 3.5])),int(2))
        s_vec_ft = np.copy(v_vec_ft)
        s_vec_ft[:,0] *= Weiner(f,sigamp,noiseamp,cut = 5,p = 4)

        spect = np.abs(v_vec_ft)
        out = np.column_stack((f,spect,np.abs(v_vec_ft),np.abs(n_vec_ft),np.abs(s_vec_ft)))
        np.savetxt(outname_spect,out,fmt='%.4f')

        ## generate a noise waveform from the spectral data, then tile() it to be as long as 1 musec
        ## Same for the filtered waveform, but for that one, just add zeros to be 1 musec long

        ## setting up to do synthetic waveforms 
        nelectrons = int(12)
        e_retardation = 531
        s_vec_ft[:,0] *= fourier_delay(f,-40) ## dial back by 40 ns
        s_vec = np.real(IFFT(s_vec_ft,axis=0))
        s_vec_extend = np.zeros((f_extend.shape[0],1),dtype=float) 
        s_vec_extend[:s_vec.shape[0],0] = s_vec[:,0]
        s_vec_extend_ft = FFT(s_vec_extend,axis=0)

        # sort inds for f and use for interp to extend noise in fourier domain
        inds = np.argsort(f)
        n_vec_extend_ft_r = np.interp(f_extend,f[inds],np.abs(n_vec_ft[inds,0]))
        n_vec_extend_ft_phi = choice(np.angle(n_vec_ft[:,0]),f_extend.shape[0])
        n_vec_extend_ft = nprect(n_vec_extend_ft_r,n_vec_extend_ft_phi)
        n_vec_extend_ft.shape = (n_vec_extend_ft.shape[0],1)

        v_sim_ft = np.tile(s_vec_extend_ft,nelectrons)
        #print('v_sim_ft.shape = ', v_sim_ft.shape)
        # first sum all the Weiner filtered and foureir_delay() signals, then add the single noise vector back

        nphotos = nelectrons//3
        npistars = nelectrons//3
        nsigstars = nelectrons//3
        evec = fillcollection(e_photon = 700,e_ret = 0,nphotos=nphotos,npistars=npistars,nsigstars=nsigstars)
        sim_times = energy2time(evec,r=e_retardation)
        print(sim_times.shape)
        sim_times = np.row_stack((0,sim_times))
        print('(shortest, longest) times [ns]\t(%i, %i)'%(np.min(sim_times),np.max(sim_times)))

        v_simsum_ft = np.zeros(s_vec_extend_ft.shape,dtype=complex)
        for i,t in enumerate(sim_times):
            #delayvec = fourier_delay(f_extend,float(i**2)*30.)
            v_simsum_ft[:,0] += s_vec_extend_ft[:,0] * fourier_delay(f_extend,t) #* fourier_delay(f_extend,10)#float(i**2)*10.)
        v_simsum_ft += n_vec_extend_ft * nelectrons
        v_simsum = np.real(IFFT(v_simsum_ft,axis=0))
        if collection.shape[0] < v_simsum.shape[0]:
            collection = t_extend
        collection = np.column_stack((collection,v_simsum))
        out = np.column_stack((t_extend,v_simsum))
        np.savetxt(outname_simTOF,out,fmt='%4f')
    collection_name = '../data_fs/extern/waveforms.dat'
    np.savetxt(collection_name,collection,fmt='%4f')
    return 0

if __name__ == '__main__':
    main()
