#!/usr/bin/python3

import numpy as np
import h5py
import hashlib
import time

def cossq(x,w,c):
    inds = np.where(np.abs(x.astype(float)-c)<w)
    y = np.zeros(x.shape)
    y[inds] = 0.5*(1+np.cos(np.pi*(x[inds].astype(float)-c)/w))
    return y

def gauss(x,w,c):
    return np.exp(-((x.astype(float)-c)/w)**2)

def build_XY(nenergies=128,nangles=64):
    x = np.arange(nenergies,dtype=float)
    w = 5.
    amp = 30.
    rng = np.random.default_rng()
    ncenters = rng.poisson(3)
    phases = rng.normal(np.pi,2,ncenters)
    centers = rng.random(ncenters)*x.shape[0]
    ymat = np.zeros((x.shape[0],nangles),dtype=float)
    for i in range(len(centers)):
        for a in range(nangles):
            kick = amp*np.cos(a*2.*np.pi/nangles + phases[i])
            ymat[:,a] += cossq(x,w,centers[i]+kick) # this produces the 2D PDF
    cmat = np.cumsum(ymat,axis=0)
    # the number of draws for each angle should be proportional to the total sum of that angle
    drawscale = 10
    hits = []
    for a in range(nangles):
        cum = cmat[-1,a]
        draws = int(drawscale*cum)
        drawpoints = np.sort(rng.random(draws))
        if cum>0:
            hits.append(list(np.interp(drawpoints,cmat[:,a]/cum,x)))
        else:
            hits.append([])
    return hits,ymat

def main():
    tstring = '%.9f'%time.clock_gettime(time.CLOCK_REALTIME)
    keyhash = hashlib.sha256(bytearray(map(ord,tstring)))
    with h5py.File('simdata.h5','a') as f:
        for i in range(10):
            bs = bytearray(map(ord,'shot_%i_'%i))
            keyhash.update(bs)
            key = keyhash.hexdigest()
            grp = f.create_group(key)
            X,Y = build_XY(nenergies = 128, nangles = 64)
            grp.create_dataset('Ypdf',data=Y,dtype=np.float32)
            hitsvec = []
            nedges = [0]
            addresses = []
            for h in X:
                if len(h)==0:
                    nedges += [0]
                    addresses += [0]
                else:
                    nedges += [len(h)]
                    addresses += [len(hitsvec)]
                    hitsvec += h
            grp.create_dataset('Xhits',data=hitsvec,dtype=np.float32)
            grp.create_dataset('Xaddresses',data=addresses,dtype=int)
            grp.create_dataset('Xnedges',data=nedges,dtype=int)
    return

if __name__ == '__main__':
    main()
