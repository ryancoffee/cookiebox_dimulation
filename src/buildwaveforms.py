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
    t = 1.e6;
    if r==0:
        t = (d1+d2+d3)/C_cmPns * np.sqrt(mc2/(2.*e))
    if r>0:
        t = d1/C_cmPns * np.sqrt(mc2/(2.*e))
        t += d3/C_cmPns * np.sqrt(mc2/(2.*(e-r)))
        t += d2/C_cmPns * np.sqrt(2)*(mc2/r)*(np.sqrt(e/mc2) - np.sqrt((e-r)/mc2))
    return t

def Weiner(f,s,n,cut,p):
    w=np.zeros(f.shape[0])
    #print(w.shape)
    p = int(4)
    inds = [i for i,nu in enumerate(f) if np.abs(nu)<cut]
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
        m_extend = 20
        f_extend = FREQ(v_vec_ft.shape[0]*m_extend,dt)
        print(f_extend.shape)
        t_extend = np.arange(t_vec[0]*m_extend,(t_vec[-1]+dt)*m_extend,dt)
        print(t_extend.shape)
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
        vout = np.copy(v_vec_ft)
        vout[:,0] *= Weiner(f,sigamp,noiseamp,cut = 5,p = 4)

        spect = np.abs(v_vec_ft)
        out = np.column_stack((f,spect,np.abs(n_vec_ft),np.abs(vout)))
        np.savetxt(outname_spect,out,fmt='%.4f')




        ## setting up to do synthetic waveforms 
        nelectrons = int(10)
        e_retardation = 500
        v_vec_ft[:,0] *= Weiner(f,sigamp,noiseamp,cut = 5,p = 4)
        v_vec_ft_r = np.abs(v_vec_ft)
        v_vec_ft_phi = np.angle(v_vec_ft)
        # sort inds for f and use for interp to extend in fourier domain
        inds = np.argsort(f)
        
        v_extend_ft_r = np.interp(f_extend,f[inds],v_vec_ft_r[inds,0])
        v_extend_ft_phi = np.interp(f_extend,f[inds],np.unwrap(v_vec_ft_phi[inds,0]))
        v_extend_vec_ft = nprect(v_extend_ft_r,v_extend_ft_phi)
        v_extend_vec_ft = fourier_delay(f_extend,v_extend_vec_ft,-800) ## pre-advancing the t0 up by 40ns to register it early in the trace

        n_extend_ft_r = np.interp(f_extend,f[inds],np.abs(n_vec_ft[inds,0]))
        n_extend_ft_phi = choice(np.angle(n_vec_ft[:,0]),f_extend.shape[0])
        n_extend_vec_ft = nprect(n_extend_ft_r,n_extend_ft_phi)
        n_extend_vec_ft.shape = (n_extend_vec_ft.shape[0],1)

        v_extend_vec_ft.shape = (v_extend_vec_ft.shape[0],1)
        v_sim_ft = np.tile(v_extend_vec_ft,nelectrons)
        print('v_sim_ft.shape = ', v_sim_ft.shape)
        # first sum all the Weiner filtered and foureir_delay() signals, then add the single noise vector back

        nphotos = nelectrons//3
        npistars = nelectrons//3
        nsigstars = nelectrons//3
        evec = fillcollection(e_photon = 700,e_ret = 0,nphotos=nphotos,npistars=npistars,nsigstars=nsigstars)
        sim_times = energy2time(evec,r=e_retardation)

        for i,t in enumerate(sim_times):
            v_sim_ft[:,i] = fourier_delay(f_extend,v_sim_ft[:,i],t)
        v_simsum_ft = np.sum(v_sim_ft,axis=1) + n_extend_vec_ft[:,0]
        print('v_simsum_ft.shape = ', v_simsum_ft.shape)
        v_simsum = np.real(IFFT(v_simsum_ft,axis=0))
        out = np.column_stack((t_extend,v_simsum))
        np.savetxt(outname_simTOF,out,fmt='%4f')


        v_copy = np.copy(vout)
        v_copy = fourier_delay(f,v_copy,10)
        v_back = np.real(IFFT(v_vec_ft,axis=0))
        v_filter_back = np.real(IFFT(vout,axis=0))
        v_filter_copy_back = np.real(IFFT(v_copy+n_vec_ft,axis=0))
        out = np.column_stack((t_vec,v_back,v_filter_back,v_filter_copy_back))
        np.savetxt(outname_time,out,fmt='%.4f')

        ## OK, now I have my energy to time
        """
        ens = np.array([5,10,25,50,100])
        print(energy2time(ens))
        ens += 500
        print(energy2time(ens,r=500))
        """

    ## OK, and now I have my fillcollection() method for getting energies
    e_retardation = 520
    totalelectrons = int(1e3)
    nphotos = totalelectrons//3
    npistars = totalelectrons//3
    nsigstars = totalelectrons//3
    v = fillcollection(e_photon = 600,e_ret = 0,nphotos=nphotos,npistars=npistars,nsigstars=nsigstars)
    np.savetxt('../data_fs/extern/timedistribution.dat',energy2time(v,r=e_retardation),fmt='%4f')

    return 0

if __name__ == '__main__':
    main()
