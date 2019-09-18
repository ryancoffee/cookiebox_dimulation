#!/usr/bin/python3

from os import getpid
from hashlib import sha1,sha256
from multiprocessing import Process,cpu_count,Pool
from ctypes import Structure,c_double,c_int,c_uint,c_wchar_p
from multiprocessing.sharedctypes import Value,Array
from time import time
from timeit import timeit
from timeit import default_timer as timer
import sys
import glob
import re as regexp
import numpy as np
from numpy import array as nparray
from numpy import concatenate as npconcatenate
from numpy import power as nppower
from numpy import save as npsave
from numpy import load as npload
from numpy import abs as npabs
from numpy import angle as npangle
from numpy import sqrt as npsqrt
from numpy import copy as npcopy
from numpy import sum as npsum
from numpy import max as npmax
from numpy import column_stack, row_stack, mean, diff, savetxt, append, vectorize, pi, cos, sin, ones, zeros, arange, argsort, interp, real, imag
from numpy.random import choice, shuffle, gamma, randn,rand,random_integers,permutation
from random import randrange, randint
from numpy.fft import fft as FFT
from numpy.fft import ifft as IFFT
from numpy.fft import fftfreq as FREQ
from scipy.constants import c
from scipy.constants import physical_constants as pc
from scipy.stats import gengamma
from scipy.sparse import coo_matrix,csr_matrix,find

#from tensorflow import python_io as tf_python_io
import tensorflow as tf

(e_mc2,unit,err) = pc["electron mass energy equivalent in MeV"]
e_mc2 *= 1e6 # eV now
(re,unit,err) = pc["classical electron radius"]
re *= 1e2 # in centimeters now


from generate_distribution import fillcollection

from cmath import rect
nprect = vectorize(rect)

def energy2time(e,r=0,d1=3.75,d2=5,d3=35):
    #distances are in centimiters and energies are in eV and times are in ns
    C_cmPns = c*100.*1e-9
    t = 1.e3 + zeros(e.shape,dtype=float);
    if r==0:
        return nparray([ (d1+d2+d3)/C_cmPns * npsqrt(e_mc2/(2.*en)) for en in e if en > 0])
    return nparray([d1/C_cmPns * npsqrt(e_mc2/(2.*en)) + d3/C_cmPns * npsqrt(e_mc2/(2.*(en-r))) + d2/C_cmPns * npsqrt(2)*(e_mc2/r)*(npsqrt(en/e_mc2) - npsqrt((en-r)/e_mc2)) for en in e if en>r])

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

def fourier_delay_matrix(f,t):
    fmat = np.tile(f,(t.shape[0],1)).T
    tmat = np.tile(t,(f.shape[0],1))
    return nprect(ones((f.shape[0],t.shape[0])),-2.*pi*fmat*tmat)

def fillimpulseresponses(printfiles = True,samplefiles = False):
    (s_collection_ft,n_collection_ft) = (nparray([0,0,0],dtype=complex),nparray([0,0,0],dtype=complex))
    filepath = './data_fs/ave1/'
    filematch = filepath + 'C1--LowPulseHighRes-in-100-out1700-an2100--*.txt'
    filelist = glob.glob(filematch)


    print('filling impulse response files\n\tnum files = %i' % len(filelist))

    for i,f in enumerate(filelist):

        ## processing images 
        ## samplefiles = False
        m = regexp.search('(.+).txt$',f)
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
        outpath = './data_fs/extern/'
        filename = outpath + 'signal_collection_ft'
        npsave(filename,s_collection_ft)
        filename = outpath + 'noise_collection_ft'
        npsave(filename,n_collection_ft)
        filename = outpath + 'frequencies_collection'
        npsave(filename,f_extend)
        filename = outpath + 'times_collection'
        npsave(filename,t_extend)

    return (s_collection_ft,n_collection_ft,f_extend,t_extend)

def readimpulseresponses(filepath='./data_fs/extern/'):
    name = filepath + 'signal_collection_ft.npy'
    s = npload(name)
    name = filepath + 'noise_collection_ft.npy'
    n = npload(name)
    name = filepath + 'times_collection.npy'
    t = npload(name)
    name = filepath + 'frequencies_collection.npy'
    f = npload(name)
    return (s,n,f,t)

def simulate_timeenergy(timeenergy,nchannels=16,e_retardation=0,energywin=(590,610),max_streak=20,printfiles = False):
    # d1-3 based on CookieBoxLayout_v2.3.dxf
    d1 = 7.6/2.
    d2 = 17.6/2.
    d3 = 58.4/2. 
    d3 -= d2
    d2 -= d1

    s_collection_ft = nparray([0],dtype=complex)
    n_collection_ft = nparray([0],dtype=complex)
    (tinds,einds,nelectrons)=find(timeenergy)
    if printfiles:
        (s_collection_ft,n_collection_ft,f_extend,t_extend) = fillimpulseresponses(printfiles=printfiles)
    else:
        infilepath = './data_fs/extern/'
        (s_collection_ft,n_collection_ft,f_extend,t_extend) = readimpulseresponses(infilepath)

    dt = t_extend[1]-t_extend[0]
    carrier_phase = 2.*pi*rand()
    sim_times = nparray([0],dtype=float)
    waveforms=np.zeros((nchannels,len(t_extend)),dtype=float)
    ToFs=nparray([],dtype=float)
    Ens=nparray([],dtype=float)
    for pulse in range(tinds.shape[0]):
        #loop on pulse
        carrier_phase += 2.*pi*(tinds[pulse]/timeenergy.shape[0])
        photon_energy = energywin[0] + (energywin[1]-energywin[0])*einds[pulse]/timeenergy.shape[1]
        tdata=nparray([],dtype=float)
        edata=nparray([],dtype=float)
        trowinds=nparray([],dtype=int)
        erowinds=nparray([],dtype=int)
        tindptr=nparray([0],int)
        eindptr=nparray([0],int)
        for chan in range(nchannels):
            #loop on channels
            angle = 2.*pi*chan/nchannels
            nphotos = int(1.+ float(nelectrons[pulse]) * nppower(sin(angle+0.0625),int(2)))
            evec = fillcollection(e_photon = photon_energy,nphotos=nphotos,npistars=0,nsigstars=0,nvalence=0,angle = carrier_phase + angle,max_streak = max_streak)
            sim_times = energy2time(evec,r=15.,d1=d1,d2=d2,d3=d3)
            edata = npconcatenate((edata,evec),axis=None)
            tdata = npconcatenate((tdata,sim_times),axis=None)
            trowinds = npconcatenate((trowinds,np.arange(sim_times.shape[0])),axis=None)
            erowinds = npconcatenate((erowinds,np.arange(evec.shape[0])),axis=None)
            tindptr = npconcatenate((tindptr,trowinds.shape[0]),axis=None)
            eindptr = npconcatenate((eindptr,erowinds.shape[0]),axis=None)

            #sim_times = append(sim_times,0.) # adds a prompt
            s_collection_colinds = choice(s_collection_ft.shape[1],sim_times.shape[0]) 
            n_collection_colinds = choice(n_collection_ft.shape[1],sim_times.shape[0]) 

            v_simsum_ft = zeros(s_collection_ft.shape[0],dtype=complex)
        
            for i,t in enumerate(sim_times):
                #samplestring = 'enumerate sim_times returns\t%i\t%f' % (i,t)
                #print(samplestring)
                v_simsum_ft += s_collection_ft[:,s_collection_colinds[i]] * fourier_delay(f_extend,t) 
                v_simsum_ft += n_collection_ft[:,n_collection_colinds[i]] 

            v_simsum = real(IFFT(v_simsum_ft,axis=0))
            waveforms[chan,:] += v_simsum
        if (ToFs.shape[0] < nchannels):
            ToFs = csr_matrix((tdata,trowinds,tindptr),shape=(nchannels,max(trowinds)+1)).toarray()
            Ens = csr_matrix((edata,erowinds,eindptr),shape=(nchannels,max(erowinds)+1)).toarray()
        else:
            ToFs = column_stack((ToFs,csr_matrix((tdata,trowinds,tindptr),shape=(nchannels,max(trowinds)+1)).toarray()))
            Ens = column_stack((Ens,csr_matrix((edata,erowinds,eindptr),shape=(nchannels,max(erowinds)+1)).toarray()))

    return (waveforms,ToFs,Ens)

def computeImages():
        nelectronsrange = (50,100)
        ntbins=8
        nebins=8
        npulses = randrange(1,5)
        tinds = [randrange(ntbins) for i in range(npulses)]
        einds = [randrange(nebins) for i in range(npulses)]
        nelectrons = [randrange(nelectronsrange[0]//npulses,nelectronsrange[1]//npulses) for i in range(npulses)]
        timeenergy = coo_matrix((nelectrons, (tinds,einds)),shape=(ntbins,nebins),dtype=int)
        (WaveForms,ToFs,Energies) = simulate_timeenergy(timeenergy,nchannels=nchannels,e_retardation=0,energywin=(600,610),max_streak=50,printfiles = False)
        return (nchannels,ntbins,nebins,npulses,WaveForms,ToFs,Energies,timeenergy.toarray())

#def spawnprocess(nchannels=16,nimages=2,nchunks=2,tfrecordpath = './data_fs/raw/tf_record_files/'):
def spawnprocess(t):
    for c in range(nchunks):
        hashstring = sha256(str.encode( '{}{}{}'.format(time(), getpid(), c) )).hexdigest()
        shardfilename = '{}tfrecord.{}'.format(tfrecordpath,hashstring)
        metafilename = '{}meta.{}'.format(tfrecordpath,hashstring)
        writer = tf.io.TFRecordWriter(shardfilename)
        strengtharray = np.zeros((nimages,),dtype=int)
        invpurityarray = np.zeros((nimages,),dtype=int)
        npulsesarray = np.zeros((nimages,),dtype=int)
        for i in range(nimages):
            print("processing image {} inside pid {}".format(i,getpid()))
            (nchannels,ntbins,nebins,npulses,WaveForms,ToFs,Energies,timeenergy) = computeImages()
            strengtharray[i] = npsum(timeenergy)
            invpurityarray[i] = npsum(timeenergy)*100//npmax(timeenergy)
            npulsesarray[i] = npulses
            waveforms_tf = WaveForms.tostring()
            ToFs_tf = ToFs.tostring()
            Energies_tf = Energies.tostring()
            simsample_tf = tf.train.Example(features = tf.train.Features(feature={
                'nangles': tf.train.Feature(int64_list=tf.train.Int64List(value = [nchannels])),
                'ntbins': tf.train.Feature(int64_list=tf.train.Int64List(value = [ntbins])),
                'nebins': tf.train.Feature(int64_list=tf.train.Int64List(value = [nebins])),
                'npulses': tf.train.Feature(int64_list=tf.train.Int64List(value = [npulses])),
                'waveforms': tf.train.Feature(bytes_list=tf.train.BytesList(value = [WaveForms.tostring()])),
                'tofs': tf.train.Feature(bytes_list=tf.train.BytesList(value = [ToFs.tostring()])),
                'energies': tf.train.Feature(bytes_list=tf.train.BytesList(value = [Energies.tostring()])),
                'timeenergy': tf.train.Feature(bytes_list=tf.train.BytesList(value = [timeenergy.tostring()]))
                }
                ))
            writer.write(simsample_tf.SerializeToString())
        writer.close()
        headstring = 'npulsesarray\tinverse purityarray*100\tstrengtharray'
        np.savetxt(metafilename,np.column_stack((npulsesarray,invpurityarray,strengtharray)),fmt='%i',header=headstring)

'''
        ############
        HERE IS MAIN
        ############

https://www.tensorflow.org/guide/datasets#preprocessing_data_with_datasetmap
'''


class MetaData(Structure):
    _fields_ = [('fname',c_wchar_p),('npulses',c_uint)]

	
def main():
    #argslist = [(nchannels,nimages,nchunks,tfrecordpath,) for i in range(nthreads)]
    argslist = [(i,) for i in range(nthreads)]
    start = timer()
    pool = Pool(nthreads)
    pool.starmap(spawnprocess, argslist)
    pool.close()
    stop = timer()
    print('### Whole loop of %i images took %.3f s' % (nimages*nchunks*nthreads,stop-start))
    return

if __name__ == '__main__':
    tfrecordpath = './data_fs/raw/tf_record_files/'
    nchannels = int(16)
    nthreads = cpu_count()*3//4
    nchunks = int(4)
    nimages = int(4)
    if len(sys.argv)>1:
        nimages = int(sys.argv[1])
        if len(sys.argv)>2:
            nchunks = int(sys.argv[2])
            if len(sys.argv)>3:
                nthreads = min(cpu_count(),int(sys.argv[3]))
                if len(sys.argv)>4:
                    nchannels = int(sys.argv[4])
    main()
