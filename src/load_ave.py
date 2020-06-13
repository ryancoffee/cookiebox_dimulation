#!/usr/bin/python3

import numpy as np
import h5py
import sys
import random
import math

def ydetToLorenzo(y):
    '''
    Lorenzo, e.g. Tixel detector, has 48x48 pixels per tile, each pixel is 100 microns square, r is in meters from Ave/Naoufal
    '''
    q = [2.*math.pi*random.random() for i in range(len(y))]
    return (np.array(q),1e4*np.array(r)*np.cos(q),1e4*np.array(r)*np.sin(q))

def katiesplit(x,y):
    sz = len(x)
    inds = np.arange(len(x))
    np.random.shuffle(inds)
    x_train = [x[i] for i in inds[:sz//4]]
    x_test = [x[i] for i in inds[sz//4:2*sz//4]]
    x_validate = [x[i] for i in inds[2*sz//4:3*sz//4]]
    x_oob = [x[i] for i in inds[3*sz//4:]]
    y_train = [y[i] for i in inds[:sz//4]]
    y_test = [y[i] for i in inds[sz//4:2*sz//4]]
    y_validate = [y[i] for i in inds[2*sz//4:3*sz//4]]
    y_oob = [y[i] for i in inds[3*sz//4:]]
    return (x_train,x_test,x_validate,y_train,y_test,y_validate,x_oob,y_oob)


def main():
    if len(sys.argv) < 2:
        print("syntax: %s <datafile>"%(sys.argv[0]) )
        return
    for fname in sys.argv[1:]:
        f = h5py.File(fname,'r') #  IMORTANT NOTE: it looks like some h5py builds have a resouce lock that prevents easy parallelization... tread lightly and load files sequentially for now.
        for vsetting in list(f.keys()):
            elist = list(f[vsetting]['energy'])
            alist = list(f[vsetting]['angle'])
            amat = np.tile(alist,(len(elist),1)).flatten()
            emat = np.tile(elist,(len(alist),1)).T.flatten()
            tdata = f[vsetting]['t_offset'][()].flatten()
            ydata = f[vsetting]['y_detector'][()].flatten()
            xdata = f[vsetting]['x_detector'][()].flatten()
            xsplat = f[vsetting]['splat']['x'][()].flatten()
            vset = f[vsetting][ list(f[vsetting].keys())[0] ][-1][1] # eventually, extract the whole voltage vector as a feature vector for use in GP inference
            # now featurize HERE HERE HERE HERE
            
    '''
    features = []
    for v in x:
        features.append((int((v-center)*8) , np.power(int((v-center)*8),int(2))//100 ))
    return features
    '''
            
        X_t = []
        X_r = []
        X_q = []
        Y_a = []
        Y_e = []
        for sim in list(f.keys()):  #[:10]:
            for chan in list(f[sim].keys()):
                for pulse in list(f[sim][chan].keys()):
                    tlist = list(f[sim][chan][pulse]['times'])
                    rlist = list(f[sim][chan][pulse]['r_detector'])
                    alist = list(f[sim][chan][pulse]['angle'])
                    elist = list(f[sim][chan][pulse]['energy'])
                    X_t += tlist
                    X_r += rlist
                    Y_a += alist
                    Y_e += elist
        X_q,X_x,X_y = NaoufalToLorenzo(X_r)
        np.savetxt("%s.ascii"%(fname),np.column_stack((X_t,X_r,X_q,X_x,X_y,Y_a,Y_e)),fmt='%.3e')
    return

if __name__ == '__main__':
    main()

