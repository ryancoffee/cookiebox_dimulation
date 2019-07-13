from scipy.stats import gengamma as gamma
import numpy as np
from numpy.random import shuffle,rand

#Global constants
from scipy.constants import c as SPEED_OF_LIGHT
from scipy.constants import physical_constants as pc
ELECTRON_MASS_ENERGY,unit, err = pc["electron mass energy equivalent in MeV"]
ELECTRON_MASS_ENERGY *= 1e6 # eV now
ELECTRON_RADIUS, unit, err = pc["classical electron radius"]
ELECTRON_RADIUS *= 1e2 # in centimeters now

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

def energy2time(e,r=0,d1=3.75,d2=5,d3=35):
    #distances are in centimiters and energies are in eV and times are in ns
    C_cm_per_ns = SPEED_OF_LIGHT*100.*1e-9
    t = 1.e3 + np.zeros(e.shape,dtype=float);
    if r==0:
        return np.array([ (d1+d2+d3)/C_cm_per_ns * np.sqrt(ELECTRON_MASS_ENERGY/(2.*en)) for en in e if en > 0])
    else :
        return np.array([d1/C_cm_per_ns * np.sqrt(ELECTRON_MASS_ENERGY/(2.*en)) + d3/C_cm_per_ns * 
                        np.sqrt(ELECTRON_MASS_ENERGY/(2.*(en-r))) + d2/C_cm_per_ns * np.sqrt(2)*
                        (C_cm_per_ns/r)*(np.sqrt(en/ELECTRON_MASS_ENERGY) - np.sqrt((en-r)/ELECTRON_MASS_ENERGY)) for en in e if en>r])


#TEST FUNCTIONS
def test_constants():
    print("List of physical constants used in this simulation")
    print("The electron mass energy equivalent in eV is {}".format(ELECTRON_MASS_ENERGY))
    print("The classical electron radius in cm is {}".format(ELECTRON_RADIUS))
    
def test_energy2time():
    print("Testing the energy to time conversion function.")
    nphotos = int(10)
    npistars = int(10)
    nsigstars = int(10)
    v = fillcollection(e_photon = 700,nphotos=nphotos,npistars=npistars,nsigstars=nsigstars)
    print(energy2time(v))
    
def test_fillcollection():
    ## Treating energies as in eV
    print("Creating 30 interactions at various times. 10 photons, 10 pistars, 10 sigstars")
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
    test_constants()
    test_fillcollection()
    test_energy2time()