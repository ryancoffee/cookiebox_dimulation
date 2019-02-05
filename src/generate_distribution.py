#!/usr/bin/python3

from scipy.stats import gengamma as gamma
from numpy import row_stack
from numpy import concatenate as npconcatenate
from numpy import array as nparray
from numpy.random import shuffle
from numpy import savetxt
from numpy import save
from numpy import max as npmax


def samplegamma(a,c,loc,scale,n):
    #mean, var, skew, kurt = gamma.stats(a=a, c=c, loc = loc,scale=scale, moments='mvsk')
    #outstring = 'mean: %f\tvariance: %f' % (mean,var)
    #print(outstring)
    #v = gamma.rvs(a=a,c=c,loc=loc,scale=scale,size=n)
    return gamma.rvs(a=a,c=c,loc=loc,scale=scale,size=n)

def fillcollection(e_photon = 600., nphotos=10,nvalence=1,nsigstars=10,npistars=20):
    ph_a = 2.
    ph_scale = 1.
    ph_ip = 540.
    v_ip = 22.
    v_scale = 1.
    v_a = 2.
    sigstar_a = 5.
    sigstar_e=542.
    sigstar_scale = 0.5
    pistar_a = 2.
    pistar_e=532.
    pistar_scale = 0.5
    c , loc = 1. , 0.
    e = e_photon - ph_ip + ph_a - samplegamma(a=ph_a,c=c,loc=loc,scale=ph_scale,n=nphotos)
    v = nparray([val for val in e if val >0])
    e = e_photon - v_ip  + v_a - samplegamma(a = v_a,c=c,loc=0,scale=v_scale,n=nvalence)
    v = npconcatenate( (v, nparray([val for val in e if val >0])))
    #print(v.shape)
    e = sigstar_e + sigstar_a - samplegamma(a = sigstar_a,c=1.,loc=0,scale=sigstar_scale,n=nsigstars)
    v = npconcatenate( (v, nparray([val for val in e if val > 0])))
    #print(v.shape)
    e = pistar_e + pistar_a - samplegamma(a = pistar_a,c=1.,loc=0,scale=pistar_scale,n=npistars)
    v = npconcatenate( (v, nparray([val for val in e if val > 0])))
    #print(v.shape)
    shuffle(v)
    return v

def main():
    ## Treating energies as in eV
    nphotos = int(10)
    npistars = int(10)
    nsigstars = int(10)
    v = fillcollection(e_photon = 700,nphotos=nphotos,npistars=npistars,nsigstars=nsigstars)
    #print(v)
    shuffle(v)
    for p in v:
        stringout = '%.2f\t:|' % p
        stringout += ' '*int(p/10)+'|'
        print(stringout)
    savetxt('../data_fs/extern/electron_energy_collection.dat',v,fmt='%4f')
    save('../data_fs/extern/electron_energy_collection',v)
    return 0




if __name__ == '__main__':
    main()
