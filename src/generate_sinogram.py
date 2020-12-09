#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import h5py


def main():
    if len(sys.argv)<3:
        print('syntax:\t%s <outputfilehead> <nimages> <streakamp optional> <nelectrons scale optional>'%(sys.argv[0]))
        return
    streakamp = 20.
    nimages = 10
    scale = 100
    if len(sys.argv)>2:
        nimages = int(sys.argv[2])
    if len(sys.argv)>3:
        streakamp = float(sys.argv[3])
    if len(sys.argv)>4:
        scale = int(sys.argv[4])

    outhead = sys.argv[1]
    nangles = 16 
    nenergies = 1024 
    emin = 0
    emax = 512

    etotalwidth = 10.
    ecentral = 535.
    angles = np.linspace(0,np.pi*2.,nangles+1)
    energies = np.linspace(emin,emax,nenergies+1)

    h5f = h5py.File('%s.simdata.h5'%(outhead),'w')
    
    for i in range(nimages):
        img = h5f.create_group('img%05i'%i)

        hist = np.zeros((nenergies,nangles),dtype=int)
        img.create_dataset('hits',maxshape = (None,nangles))
        img.attrs['npulses'] = int(np.random.uniform(1,10))
        img.attrs['esase'] = np.random.normal(ecentral,etotalwidth,(npulses,))
        img.attrs['ewidths'] = np.random.gamma(1.5,.125,(npulses,))+.5
        img.attrs['ephases'] = np.random.uniform(0.,2.*np.pi,(npulses,))
        # rather than this, let's eventually switch to using a dict for the Auger features and then for every ncounts photoelectron, we pick from this distribution an Auger electron.
        naugerfeatures = {365:1.5,369:1.5,372:1.5}
        caugerfeatures = {250.:3.,255.:2.5,260.:2.5}
        oaugerfeatures = {505:2.5,497:1.,492:1.}
        img.attrs['augers'] = {**naugerfeatures,**caugerfeatures,**oaugerfeatures}
        nitrogencenters = img.attrs['esase']-409.9
        carboncenters = img.attrs['esase']-284.2
        nvalencecenters = img.attrs['esase']-37.3
        ovalencecenters = img.attrs['esase']-41.6
        img.attrs['photos'] = {**carboncenters,**nitrogencenters}
        img.attrs['valencephotos'] = {**nvalencecenters,**ovalencecenters}
        img.attrs['carrier'] = np.random.uniform(0.,2.*np.pi)
        img.attrs['streakamp'] = streakamp
        img.attrs['legcoeffs'] = np.zeros((npulses,5),dtype=float) # only allowing for the 0,2,4 even coeffs for now

        for p in range(img.attrs['npulses']):
            img.attrs['legcoeffs'][p,:] = [1., 0., np.random.uniform(-1,1), 0., np.random.uniform(-(c0+c2),c0+c2)]
            poldist = np.polynomial.legendre.Legendre(img.attrs['legcoeffs'][p,:])(np.cos(angles[:-1]))
            for a in range(nangles):
                ncounts = int(poldist[a] * scale/3.)
                augercounts = int(scale)
                if ncounts > 0:
                    streak = img.attrs['streakamp']*np.cos(angles[a]-img.attrs['ephases'][i]+img.attrs['carrier'])
                    ens = [v for v in np.random.normal(img.attrs['esase'][p]+streak,img.attrs['ewidths'][p],(ncounts,)))]
                    ens += [v for v in np.random.normal(img.attrs['nitrogencenters'][p]+streak,img.attrs['ewidths'][p],(ncounts,)))]
                    ens += [v for v in np.random.normal(img.attrs['carboncenters'][p]+streak,img.attrs['ewidths'][p],(ncounts,)))]
                    ens += [v for v in np.random.normal(img.attrs['nvalencecenters'][p]+streak,img.attrs['ewidths'][p],(ncounts,)))]
                    ens += [v for v in np.random.normal(img.attrs['ovalencecenters'][p]+streak,img.attrs['ewidths'][p],(ncounts,)))]
                    augercenters = np.random.choice(list(img.attrs['augers'].keys()),(augercounts,))
                    ens += [np.random.normal(c,augerfeatures[c])+streak for c in augercenters]
                    img['hist'][:,a] += np.histogram(ens,energies)[0]
            HERE HERE HERE trying to set an accumulation oint for all the hits as list of energies.

        img.create_dataset('hist',data=hist)

        np.savetxt(outname,hist_ens,fmt='%i')
        np.savetxt('%s.%s'%(outname,'augers'),hist_ens_auger,fmt='%i')
        np.savetxt('%s.%s'%(outname,'full'),hist_ens_auger+hist_ens,fmt='%i')

    h5f.close()
    '''
    fig, ax = plt.subplots()#subplot_kw={"projection": "3d"})
    X,Y = np.meshgrid(angles,energies)
    Z = hist_ens.copy()
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    '''

    
    return


if __name__ == '__main__':
    main()
