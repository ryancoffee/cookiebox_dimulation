#!/usr/bin/python3

import sys
import numpy as np
import h5py

def main():
    if len(sys.argv)<2:
        print('syntax:\t%s <outputfilehead> <nimages> <streakamp optional> <nelectrons scale optional>'%(sys.argv[0]))
        return
    streakamp = 20.
    nimages = 10
    scale = 10
    if len(sys.argv)>2:
        nimages = int(sys.argv[2])
    if len(sys.argv)>3:
        streakamp = float(sys.argv[3])
    if len(sys.argv)>4:
        scale = int(sys.argv[4])

    outhead = sys.argv[1]
    nangles = 16 
    nenergies = 2048 
    emin = 0
    emax = 512

    etotalwidth = 10.
    ecentral = 535.
    angles = np.linspace(0,np.pi*2.,nangles+1)
    energies = np.linspace(emin,emax,nenergies+1)

    h5f = h5py.File('%s.simdata.h5'%(outhead),'w')
    
    for i in range(nimages):
        img = h5f.create_group('img%05i'%i)

        img.attrs['npulses'] = int(np.random.uniform(1,4))
        img.attrs['esase'] = np.random.normal(ecentral,etotalwidth,(img.attrs['npulses'],))
        img.attrs['ewidths'] = np.random.gamma(1.5,.125,(img.attrs['npulses'],))+.5
        img.attrs['ephases'] = np.random.uniform(0.,2.*np.pi,(img.attrs['npulses'],))
        # rather than this, let's eventually switch to using a dict for the Auger features and then for every ncounts photoelectron, we pick from this distribution an Auger electron.
        naugerfeatures = {365:1.5,369:1.5,372:1.5}
        caugerfeatures = {250.:3.,255.:2.5,260.:2.5}
        oaugerfeatures = {505:2.5,497:1.,492:1.}
        augerfeatures = {**naugerfeatures,**caugerfeatures,**oaugerfeatures}
        img.create_group('augers')
        for center in list(augerfeatures.keys()):
            img['augers'].attrs['%.2f'%center] = float(augerfeatures[center])
            #print('%.2f'%center)
            #print(img['augers'].attrs['%.2f'%center])
        for center in (img.attrs['esase']):
            nitrogencenters = {center-409.9 : 0.5}
            carboncenters = {center-284.2 : 0.5}
            nvalencecenters = {center-37.3 : 0.5}
            ovalencecenters = {center-41.6 : 0.5}

        photofeatures = {**carboncenters,**nitrogencenters}
        img.create_group('photos')
        print(list(photofeatures.keys()))
        for center in list(photofeatures.keys()):
            img['photos'].attrs['%.2f'%center] = float(photofeatures[center])

        valencefeatures = {**nvalencecenters,**ovalencecenters}
        img.create_group('valencephotos')
        for center in list(valencefeatures.keys()):
            img['valencephotos'].attrs['%.2f'%center] = float(valencefeatures[center])

        img.attrs['carrier'] = np.random.uniform(0.,2.*np.pi)
        img.attrs['streakamp'] = streakamp
        img.create_dataset('legcoeffs',shape=(img.attrs['npulses'],5),dtype=float) # only allowing for the 0,2,4 even coeffs for now

        ens = []
        for a in range(nangles):
            ens.append([])
        for p in range(img.attrs['npulses']):
            c0 = 1.
            c2 = np.random.uniform(-1,1)
            c4 = np.random.uniform(-(c0+c2),c0+c2)
            img['legcoeffs'][p,:] = [c0, 0., c2, 0., c4]
            poldist = np.polynomial.legendre.Legendre(img['legcoeffs'][p,:])(np.cos(angles[:-1]))
            for a in range(nangles):
                ncounts = int(poldist[a] * scale)
                augercounts = int(scale)
                if ncounts > 0:
                    streak = img.attrs['streakamp']*np.cos(angles[a]-img.attrs['ephases'][p]+img.attrs['carrier'])
                    centers = list(np.random.choice(list(img['photos'].attrs.keys()),ncounts))
                    ens[a] += [np.random.normal(float(c),float(img['photos'].attrs[c]),ncounts) for c in centers]
                    '''
                    ens = [v for v in np.random(center + streak, width,)]
                    ens = [v for v in np.random.normal(img.attrs['esase'][p]+streak,img.attrs['ewidths'][p],(ncounts,))]
                    ens += [v for v in np.random.normal(img['photos'].attrs['nitrogencenters'][p]+streak,img.attrs['ewidths'][p],(ncounts,))]
                    ens += [v for v in np.random.normal(img.attrs['carboncenters'][p]+streak,img.attrs['ewidths'][p],(ncounts,))]
                    ens += [v for v in np.random.normal(img.attrs['nvalencecenters'][p]+streak,img.attrs['ewidths'][p],(ncounts,))]
                    ens += [v for v in np.random.normal(img.attrs['ovalencecenters'][p]+streak,img.attrs['ewidths'][p],(ncounts,))]
                    augercenters = np.random.choice(list(img.attrs['augers'].keys()),(augercounts,))
                    ens += [np.random.normal(c,augerfeatures[c])+streak for c in augercenters]
                    nens = len(ens)
                    print(nens)
                    '''
        h = np.zeros((nenergies,nangles),dtype=int)
        for a in range(nangles):
            print(ens[a])
            h[:,a] = np.histogram(ens[a],energies)[0]
        img.create_dataset('hist',data=h)
        hits = img.create_group('hits')
        for a in range(nangles):
            hits.create_dataset('%i'%a,ens[a])
    h5f.close()

    return


if __name__ == '__main__':
    main()
