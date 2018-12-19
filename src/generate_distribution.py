#!/usr/bin/python3

from scipy.stats import gengamma as gamma
from numpy import row_stack
from numpy.random import shuffle
from numpy import savetxt


def samplegamma(a,c,loc,scale,n):
    mean, var, skew, kurt = gamma.stats(a=a, c=c, loc = loc,scale=scale, moments='mvsk')
    outstring = 'mean: %f\tvariance: %f' % (mean,var)
    #print(outstring)
    v = gamma.rvs(a=a,c=c,loc=loc,scale=scale,size=n)
    v.shape = (v.shape[0],1)
    return v

def fillcollection(e_photon = 600., e_ret = 500. , nphotos=10,nsigstars=10,npistars=20):
    ph_a = 2.
    ph_scale = 1.
    ph_ip = 540.
    v_ip = 20.
    v_scale = 1.
    v_a = 2.
    sigstar_a = 5.
    sigstar_e=542.
    sigstar_scale = 0.5
    pistar_a = 2.
    pistar_e=532.
    pistar_scale = 0.5
    c , loc = 1. , 0.
    e = e_photon - ph_ip - e_ret + ph_a
    v =  e - samplegamma(a=ph_a,c=c,loc=loc,scale=ph_scale,n=nphotos)
    #e = e_photon - v_ip - e_ret + sigstar_a
    #if e>0:
    #    v = row_stack( (v, e - samplegamma(a = v_a,c=c,loc=loc,scale=v_scale,n=nphotos//10)) )
    #print(v.shape)
    e = sigstar_e - e_ret + sigstar_a
    v = row_stack( (v, e - samplegamma(a = sigstar_a,c=1.,loc=0,scale=sigstar_scale,n=nsigstars)) )
    #print(v.shape)
    e = pistar_e - e_ret + pistar_a
    v = row_stack( (v, e - samplegamma(a = pistar_a,c=1.,loc=0,scale=pistar_scale,n=npistars)) )
    #print(v.shape)
    shuffle(v)
    return v

def main():
    ## Treating energies as in eV
    nphotos = int(80)*10
    npistars = int(80)*10
    nsigstars = int(40)*10
    v = fillcollection(e_photon = 560,e_ret = 500,nphotos=nphotos,npistars=npistars,nsigstars=nsigstars//10)
    #print(v)
    shuffle(v)
    for p in v[:50]:
        stringout = ' '*int(p)+'|'
        print(stringout)
    savetxt('../data_fs/extern/electron_collection.dat',v)
    return 0




if __name__ == '__main__':
    main()
