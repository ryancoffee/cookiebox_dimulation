from scipy.stats import gengamma as gamma
import numpy as np
from numpy.random import shuffle,rand

def fillcollection(e_photon = 600., nphotos=10,nvalence=1,nsigstars=10,npistars=20,angle = 0.,max_streak=0):
    #Function returns an array with the number of interactions at a certain time and energy
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
    e = e_photon - ph_ip + ph_a - gamma.rvs(a=ph_a,c=c,loc=loc,scale=ph_scale,size=nphotos) + max_streak * np.cos(angle)
    v = np.array([val for val in e if val >0])
    e = e_photon - v_ip  + v_a - gamma.rvs(a = v_a,c=c,loc=0,scale=v_scale,size=nvalence)
    v = np.concatenate( (v, np.array([val for val in e if val >0])))
    #print(v.shape)
    e = sigstar_e + sigstar_a - gamma.rvs(a = sigstar_a,c=1.,loc=0,scale=sigstar_scale,size=nsigstars)
    v = np.concatenate( (v, np.array([val for val in e if val > 0])))
    #print(v.shape)
    e = pistar_e + pistar_a - gamma.rvs(a = pistar_a,c=1.,loc=0,scale=pistar_scale,size=npistars)
    v = np.concatenate( (v, np.array([val for val in e if val > 0])))
    #print(v.shape)
    np.random.shuffle(v)
    return v


#TEST FUNCTIONS
def test_fillcollection():
    ## Treating energies as in eV
    nphotos = int(10)
    npistars = int(10)
    nsigstars = int(10)
    v = fillcollection(e_photon = 700,nphotos=nphotos,npistars=npistars,nsigstars=nsigstars)
    #print(v)
    np.random.shuffle(v)
    for p in v:
        stringout = '%.2f\t:|' % p
        stringout += ' '*int(p/10)+'|'
        print(stringout)
    np.savetxt('../data_fs/extern/test_electron_energy_collection.dat',v,fmt='%4f')
    np.save('../data_fs/extern/test_electron_energy_collection',v)

if __name__ == '__main__':
    test_fillcollection()