#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def main():
    if len(sys.argv)<2:
        print('syntax:\t%s <outputfilename> <streakamp optional>'%(sys.argv[0]))
        return
    streakamp = 20.
    if len(sys.argv)>2:
        streakamp = float(sys.argv[2])

    outname = sys.argv[1]
    nangles = 128 
    nenergies = 128
    emin = 0
    emax = 100
    npulses = int(np.random.uniform(3,8))


    etotalwidth = 10.
    ecentral = 50.
    
    ecenters = np.random.normal(ecentral,etotalwidth,(npulses,))
    ewidths = np.random.gamma(5.,.25,(npulses,))+1.25
    ephases= np.random.uniform(0.,2.*np.pi,(npulses,))

    angles = np.linspace(0,np.pi*2.,nangles+1)
    energies = np.linspace(emin,emax,nenergies+1)
    scale = 100
    hist_ens = np.zeros((nenergies,nangles),dtype=int)
    for i in range(npulses):
        c0 = 1.
        c2 = np.random.uniform(-1,1)
        c4 = np.random.uniform(-(c0+c2),c0+c2)
        poldist = np.polynomial.legendre.Legendre([c0,0,c2,0,c4])(np.cos(angles[:-1]))
        for a in range(nangles):
            ncounts = int(poldist[a] * scale)
            streak = streakamp*np.cos(angles[a]-ephases[i])
            hist_ens[:,a] += np.histogram(np.random.normal(ecenters[i]+streak,ewidths[i],(ncounts,)),energies)[0]


    np.savetxt(outname,hist_ens,fmt='%i')
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
