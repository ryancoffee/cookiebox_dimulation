#!/usr/bin/python3

import sys
import numpy as np
import h5py

def main():
    if len(sys.argv)<2:
        print('syntax:\t%s <outputfilehead> <nimages> <streakamp optional> <nelectrons scale optional>'%(sys.argv[0]))
        return
    streakamp = 50.
    nimages = 10
    scale = 10
    if len(sys.argv)>2:
        nimages = int(sys.argv[2])
    if len(sys.argv)>3:
        streakamp = float(sys.argv[3])
    if len(sys.argv)>4:
        scale = int(sys.argv[4])

    outhead = sys.argv[1]
    nangles = 256 
    nenergies = 512 
    emin = 0
    emax = 512 

    etotalwidth = 20.
    ecentral = 600.
    angles = np.linspace(0,np.pi*2.,nangles+1)
    energies = np.linspace(emin,emax,nenergies+1)

    h5f = h5py.File('%s.ImgSegSim.h5'%(outhead),'w')

    # using dictionary to hold the list of [width,crossection]
    oxygencenters = {-541.5 : [0.5,1.]}
    nitrogencenters = {-409.9 : [1.5,2.]}
    carboncenters = {-284.2 : [2.5,4.]}
    nvalencecenters = {-37.3 : [0.5,.2]}
    ovalencecenters = {-41.6 : [0.5,1.]}

    h5f.create_group('augers')
    naugerfeatures = {365:1.5,369:1.5,372:1.5}
    caugerfeatures = {250.:3.,255.:2.5,260.:2.5}
    oaugerfeatures = {505:2.5,497:1.,492:1.}
    augerfeatures = {**naugerfeatures,**caugerfeatures,**oaugerfeatures}
    for center in list(augerfeatures.keys()):
        h5f['augers'].attrs['%.2f'%center] = float(augerfeatures[center])
        #print('%.2f'%center)
        #print(img['augers'].attrs['%.2f'%center])

    photofeatures = {**carboncenters,**nitrogencenters,**oxygencenters}
    h5f.create_group('photos')
    for center in list(photofeatures.keys()):
        h5f['photos'].attrs['%.2f'%center] = [photofeatures[center][0],photofeatures[center][1]]

    valencefeatures = {**nvalencecenters,**ovalencecenters}
    h5f.create_group('valencephotos')
    for center in list(valencefeatures.keys()):
        h5f['valencephotos'].attrs['%.2f'%center] = valencefeatures[center]

    
    for i in range(nimages):
        img = h5f.create_group('img%05i'%i)

        img.attrs['npulses'] = int(np.random.uniform(2,4))
        # rather than this, let's eventually switch to using a dict for the Auger features and then for every ncounts photoelectron, we pick from this distribution an Auger electron.
        img.attrs['carrier'] = np.random.uniform(0.,2.*np.pi)
        img.attrs['streakamp'] = streakamp

        ens = []
        for p in range(img.attrs['npulses']):
            ens.append([])
            for a in range(nangles):
                ens[-1].append([])

        #h = np.zeros((nenergies,nangles),dtype=int)

        for p in range(img.attrs['npulses']):
            pulsegrp = img.create_group('pulse%02i'%p)
            pulsegrp.attrs['phase'] = np.random.normal(0.,np.pi/8)
            pulsegrp.attrs['esase'] = np.random.normal(ecentral,etotalwidth)
            pulsegrp.attrs['ewidth'] = np.random.gamma(1.5,.125)+.5
            c0 = 1.
            c2 = -1.0 #np.random.uniform(-1,1) 
            c4 = 0 #np.random.uniform(-(c0+c2),c0+c2)
            pulsegrp.create_dataset('legcoeffs',data=[c0, 0., c2, 0., c4])
            poldist = np.polynomial.legendre.Legendre(pulsegrp['legcoeffs'])(np.cos(angles[:-1]))
            pulsehist = np.zeros((nenergies,nangles),dtype=int)
            for a in range(nangles):
                ncounts = int(poldist[a] * scale)
                augercounts = int(scale)
                if ncounts > 0:
                    streak = img.attrs['streakamp']*np.cos(angles[a]-pulsegrp.attrs['phase']+img.attrs['carrier'])
                    centers = list(np.random.choice(list(h5f['photos'].attrs.keys()),int(np.sqrt(ncounts))))
                    for c in centers:
                        ens[p][a] = list(np.random.normal(pulsegrp.attrs['esase'] + float(c) + float(streak)
                            , np.sqrt( np.power(float(pulsegrp.attrs['ewidth']),int(2)) + np.power(float(h5f['photos'].attrs[c][0]),int(2)) )
                            , int(np.sqrt(ncounts)*h5f['photos'].attrs[c][1])))
                        '''
                    centers = list(np.random.choice(list(h5f['valencephotos'].attrs.keys()),int(np.sqrt(ncounts//10))))
                    for c in centers:
                        ens[p][a] += list(np.random.normal(img.attrs['esase'][p] + float(c) + float(streak),float(h5f['valencephotos'].attrs[c]),int(np.sqrt(ncounts//10))))
                    centers = list(np.random.choice(list(h5f['augers'].attrs.keys()),int(np.sqrt(augercounts))))
                    for c in centers:
                        ens[p][a] += list(np.random.normal(img.attrs['esase'][p] + float(c) + float(streak),float(h5f['augers'].attrs[c]),int(np.sqrt(augercounts))))
                        '''
                    #h[:,a] += np.histogram(ens[a],energies)[0]
                    pulsehist[:,a] = np.histogram(ens[p][a],energies)[0]
                pulsegrp.create_dataset('hits_ang%02i'%a,data=ens[p][a])
            pulsegrp.create_dataset('hist',data=pulsehist)

        #img.create_dataset('hist',data=h)
        #img.create_dataset('energies',data=energies[:-1])

    h5f.close()

    return


if __name__ == '__main__':
    main()
