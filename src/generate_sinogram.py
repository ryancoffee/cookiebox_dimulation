#!/usr/bin/python3

import sys
import numpy as np


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
    nangles = 64 
    nenergies = 1024 
    emin = 0
    emax = 512

    etotalwidth = 10.
    ecentral = 0.
    angles = np.linspace(0,np.pi*2.,nangles+1)
    energies = np.linspace(emin,emax,nenergies+1)
    
    for img in range(nimages):
        hist_ens_NNO = np.zeros((nenergies,nangles),dtype=int)
        hist_ens_OCO = np.zeros((nenergies,nangles),dtype=int)
        hist_ens_auger_NNO = np.zeros((nenergies,nangles),dtype=int)
        hist_ens_auger_OCO = np.zeros((nenergies,nangles),dtype=int)
        outname = '%s.%02i'%(outhead,img)
        npulses = int(np.random.uniform(3,8))
        ecenters = np.random.normal(ecentral,etotalwidth,(npulses,))
        ewidths = np.random.gamma(3.5,.125,(npulses,))+.5
        ephases= np.random.uniform(0.,2.*np.pi,(npulses,))
        # rather than this, let's eventually switch to using a dict for the AUger features and then for every ncounts photoelectron, we pick from this distribution an Auger electron.
        augerfeatures_NNO = {505:2.5,497:1.,492:1.,365:1.5,369:1.5,372:1.5}
        augerfeatures_OCO = {505:2.5,497:1.,492:1.,250.:3.,255.:2.5,260.:2.5}
        #augerfeatures = {505:1.5,492:1.,365:1.5,372:1.5,250.:1.,260.:1.5}
        nitrogencenters = ecenters+535.-410.
        carboncenters = ecenters+535.-285.

        for i in range(npulses):
            c0 = 1.
            c2 = np.random.uniform(-1,1)
            c4 = np.random.uniform(-(c0+c2),c0+c2)
            poldist = np.polynomial.legendre.Legendre([c0,0,c2,0,c4])(np.cos(angles[:-1]))
            for a in range(nangles):
                ncounts = int(poldist[a] * scale/3.)
                augercounts = int(scale)
                if ncounts > 0:
                    streak = streakamp*np.cos(angles[a]-ephases[i])
                    #hist_ens_NNO[:,a] += np.histogram(np.random.normal(ecenters[i]+streak,ewidths[i],(ncounts,)),energies)[0]
                    hist_ens_NNO[:,a] += np.histogram(np.random.normal(nitrogencenters[i]+streak,ewidths[i],(ncounts,)),energies)[0]
                    augercenters_NNO = np.random.choice(list(augerfeatures_NNO.keys()),(augercounts,))
                    ens_auger_NNO = [np.random.normal(c,augerfeatures_NNO[c])+streak for c in augercenters_NNO]
                    hist_ens_auger_NNO[:,a] += np.histogram(ens_auger_NNO,energies)[0]
                    #hist_ens_OCO[:,a] += np.histogram(np.random.normal(ecenters[i]+streak,ewidths[i],(ncounts,)),energies)[0]
                    hist_ens_OCO[:,a] += np.histogram(np.random.normal(carboncenters[i]+streak,ewidths[i],(ncounts,)),energies)[0]
                    augercenters_OCO = np.random.choice(list(augerfeatures_OCO.keys()),(augercounts,))
                    ens_auger_OCO = [np.random.normal(c,augerfeatures_OCO[c])+streak for c in augercenters_OCO]
                    hist_ens_auger_OCO[:,a] += np.histogram(ens_auger_OCO,energies)[0]

        #np.savetxt(outname,hist_ens,fmt='%i')
        #np.savetxt('%s.%s'%(outname,'augers'),hist_ens_auger,fmt='%i')
        np.savetxt('%s.%s'%(outname,'full_NNO'),hist_ens_auger_NNO+hist_ens_NNO,fmt='%i')
        np.savetxt('%s.%s'%(outname,'full_OCO'),hist_ens_auger_OCO+hist_ens_OCO,fmt='%i')
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
