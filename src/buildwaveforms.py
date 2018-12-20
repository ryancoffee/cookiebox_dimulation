#!/usr/bin/python3

import glob
import re
import numpy as np
from numpy import array as nparray
from numpy import power as nppower
from numpy import save as npsave
from numpy import abs as npabs
from numpy import angle as npangle
from numpy import sqrt as npsqrt
from numpy import copy as npcopy
from numpy import sum as npsum
from numpy import column_stack, row_stack, mean, diff, savetxt, append, vectorize, pi, cos, ones, zeros, arange, argsort, interp, real, imag
from numpy.random import choice, shuffle
from numpy.fft import fft as FFT
from numpy.fft import ifft as IFFT
from numpy.fft import fftfreq as FREQ
from scipy.constants import c

from generate_distribution import fillcollection

from cmath import rect
nprect = vectorize(rect)

def energy2time(e,r=0,d1=2.5,d2=5,d3=35):
    #distances are in centimiters and energies are in eV and times are in ns
    C_cmPns = c*100.*1e-9
    mc2 = float(0.511e6)
    t = 1.e3 + zeros(e.shape,dtype=float);
    if r==0:
        return nparray([ (d1+d2+d3)/C_cmPns * npsqrt(mc2/(2.*en)) for en in e if en > 0])
    return nparray([d1/C_cmPns * npsqrt(mc2/(2.*en)) + d3/C_cmPns * npsqrt(mc2/(2.*(en-r))) + d2/C_cmPns * npsqrt(2)*(mc2/r)*(npsqrt(en/mc2) - npsqrt((en-r)/mc2)) for en in e if en>r])

def Weiner(f,s,n,cut,p):
    w=zeros(f.shape[0])
    #print(w.shape)
    p = int(4)
    inds = [i for i,nu in enumerate(f) if npabs(nu)<cut]
    w[inds] = s*nppower(cos(pi/2. * f[inds] / cut) , p)
    return w/(w+n)

def fourier_delay(f,dt):
    #print('f.shape = ',f.shape)
    ## int f(t) exp(i*w*t) dt
    ## int f(t+tau) exp(i*w*t) dt --> int f(t)exp(i*w*t)exp(-i*w*tau) dt
    ## IFFT{ F(w) exp(-i*w*tau) }
    return nprect(ones(f.shape),-f*2.*pi*dt)

def fillimpulseresponses(s_collection_ft,n_collection_ft,printfiles = True,samplefiles = False):
    filepath = '../data_fs/ave1/'
    filematch = filepath + 'C1--LowPulseHighRes-in-100-out1700-an2100--*.txt'
    filelist = glob.glob(filematch)


    print('filling impulse response files\n\tnum files = %i' % len(filelist))

    for i,f in enumerate(filelist):

        ## processing images 
        ## samplefiles = False
        m = re.search('(.+).txt$',f)
        if (i%10 == 0 and samplefiles):
            outname_spect = m.group(1) + '.spect.dat'
            outname_time = m.group(1) + '.time.dat'
            outname_simTOF = m.group(1) + '.simTOF.dat'

        fi = open(f, "r")
        for passline in range(6):
            headline = '# ' + fi.readline()
        (t,v) = fi.readline().split()
        v_vec=nparray(float(v),dtype=float)
        t_vec=nparray(float(t)*1.e9,dtype=float)
        for line in fi:
            (t,v) = line.split()
            v_vec = row_stack((v_vec,float(v)))
            t_vec = row_stack((t_vec,float(t)*1.e9))
        fi.close()
        #Get the mean time-step for sake of frequencies
        dt = mean(diff(t_vec,n=1,axis=0))
        #FFT the vector
        v_vec_ft = FFT(v_vec,axis=0)
        f = FREQ(v_vec_ft.shape[0],dt)
        m_extend = 10
        f_extend = FREQ(v_vec_ft.shape[0]*m_extend,dt)
        t_extend = arange(0,((t_vec[-1]-t_vec[0])+dt)*m_extend,dt)
        # deep copy for the noise extimation 
        n_vec_ft = npcopy(v_vec_ft)
        # find indices where there is only noise in the power, and indices with predominantly signal
        # replace the signal elements in the noise vector with a random sampling from the noise portion
        chooseinds = nparray([i for i,nu in enumerate(f) if (npabs(nu)> 6.5 and npabs(nu)<(20))])
        replaceinds = nparray([i for i,nu in enumerate(f) if npabs(nu)< 6.5])
        values = choice(n_vec_ft[chooseinds,0],len(replaceinds))
        n_vec_ft[replaceinds,0] = values

        ## build noise vector and add to n_collection_ft
        # sort inds for f and use for interp to extend noise in fourier domain
        inds = argsort(f)
        n_vec_extend_ft_r = interp(f_extend,f[inds],npabs(n_vec_ft[inds,0]))
        n_vec_extend_ft_phi = choice(npangle(n_vec_ft[:,0]),f_extend.shape[0])
        n_vec_extend_ft = nprect(n_vec_extend_ft_r,n_vec_extend_ft_phi)
        n_vec_extend_ft.shape = (n_vec_extend_ft.shape[0],1)
        
        if n_collection_ft.shape[0] < n_vec_extend_ft.shape[0]:
            n_collection_ft = npcopy(n_vec_extend_ft)
           # s_collection_ft.shape = (s_collection_ft.shape[0],1)
        else:
            n_collection_ft = column_stack((n_collection_ft,n_vec_extend_ft))

        ## build signal vector and add to n_collection_ft
        noiseamp = nppower(mean(npabs(values)),int(2))
        sigamp = nppower(mean(nparray([i for i,nu in enumerate(f) if npabs(nu)< 1.0])),int(2))
        s_vec_ft = npcopy(v_vec_ft)
        s_vec_ft[:,0] *= Weiner(f,sigamp,noiseamp,cut = 5,p = 4) * fourier_delay(f,-40) ## Weiner filter and dial back by 40 ns

        if samplefiles:
            out = column_stack((f,npabs(v_vec_ft),npabs(n_vec_ft),npabs(s_vec_ft)))
            savetxt(outname_spect,out,fmt='%.4f')

        s_vec = real(IFFT(s_vec_ft,axis=0))
        s_vec_extend = zeros((f_extend.shape[0],1),dtype=float) 
        s_vec_extend[:s_vec.shape[0],0] = s_vec[:,0]
        s_vec_extend_ft = FFT(s_vec_extend,axis=0)

        if s_collection_ft.shape[0] < s_vec_extend_ft.shape[0]:
            s_collection_ft = npcopy(s_vec_extend_ft)
           # s_collection_ft.shape = (s_collection_ft.shape[0],1)
        else:
            s_collection_ft = column_stack((s_collection_ft,s_vec_extend_ft))

        # first sum all the Weiner filtered and foureir_delay() signals, then add the single noise vector back
    if printfiles:
        outpath = '../data_fs/extern/'
        filename = outpath + 'signal_collection_ft'
        npsave(filename,s_collection_ft)
        filename = outpath + 'noise_collection_ft'
        npsave(filename,n_collection_ft)
        filename = outpath + 'frequencies_collection'
        npsave(filename,f_extend)
        filename = outpath + 'times_collection'
        npsave(filename,t_extend)

    return (s_collection_ft,n_collection_ft,f_extend,t_extend)


def simulate_tof(nwaveforms=16,nelectrons=12,e_retardation=530,e_photon=600):
    collection = nparray([0,1,2],dtype=float)
    s_collection_ft = nparray([0,1,2],dtype=complex)
    n_collection_ft = nparray([0,1,2],dtype=complex)
    (s_collection_ft,n_collection_ft,f_extend,t_extend) = fillimpulseresponses(s_collection_ft,n_collection_ft,printfiles=True)
    print(s_collection_ft.shape)
    dt = t_extend[1]-t_extend[0]

    #nwaveforms=10 # now a method input
    for i in range(nwaveforms):
        # this is for the incremental output as the collection is building
        #nelectrons = int(16)


        #e_retardation = 530 ## now a method input
        nphotos = nelectrons//3
        npistars = nelectrons//3
        nsigstars = nelectrons//3
        # d1-3 based on CookieBoxLayout_v2.3.dxf
        d1 = 7.6/2.
        d2 = 17.6/2.
        d3 = 58.4/2. 
        d3 -= d2
        d2 -= d1
        evec = fillcollection(e_photon = e_photon,nphotos=nphotos,npistars=npistars,nsigstars=nsigstars)
        sim_times = energy2time(evec,r=15.,d1=d1,d2=d2,d3=d3)
        sim_times = append(sim_times,0.) # adds a prompt

        s_collection_colinds = choice(s_collection_ft.shape[1],sim_times.shape[0]) 
        n_collection_colinds = choice(n_collection_ft.shape[1],sim_times.shape[0]) 

        v_simsum_ft = zeros(s_collection_ft.shape[0],dtype=complex)
        
        for i,t in enumerate(sim_times):
            #samplestring = 'enumerate sim_times returns\t%i\t%f' % (i,t)
            #print(samplestring)
            v_simsum_ft += s_collection_ft[:,s_collection_colinds[i]] * fourier_delay(f_extend,t) 
            v_simsum_ft += n_collection_ft[:,n_collection_colinds[i]] 

        v_simsum = np.real(IFFT(v_simsum_ft,axis=0))
        if collection.shape[0] < v_simsum.shape[0]:
            collection = t_extend
        collection = column_stack((collection,v_simsum))


    return collection

def main():

    collection = simulate_tof(nwaveforms=16,nelectrons=12,e_retardation=530,e_photon=600)

    ### Writing output files ###
    collection_name = '../data_fs/extern/CookieBox_waveforms.randomsources.dat'
    savetxt(collection_name,collection,fmt='%4f')

    collection_name = '../data_fs/extern/CookieBox_waveforms.randomsources'
    npsave(collection_name,collection)

    integration_name = '../data_fs/extern/integration.randomsources.dat'
    out = column_stack((collection[:,0],npsum(collection[:,1:],axis=1)))
    savetxt(integration_name,out,fmt='%4f')

if __name__ == '__main__':
    main()
