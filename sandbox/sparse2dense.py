#!/usr/bin/python3

import numpy
import h5py
import sys
import re
import numpy as np

def sparse2dense(h5name,split=0.1):
    rng = np.random.default_rng()
    with h5py.File(h5name,'r') as f:
        Y_train = []
        X_train = []
        Y_test = []
        X_test = []
        for i,k in enumerate(f.keys()):
            grp = f[k]
            headstr = str(k)
            headstr += '\t%i_drawscale\t%i_nangles\t%i_nenergies'%(grp.attrs['drawscale'],grp.attrs['nangles'],grp.attrs['nenergies'])
            hbins = np.arange(grp.attrs['nenergies']+1)
            img = np.zeros((grp.attrs['nangles'],grp.attrs['nenergies']),dtype=np.uint16)

            for a in range(grp.attrs['nangles']):
                offset = grp['Xaddresses'][()][a]
                nhits = grp['Xnedges'][()][a]
                img[a,:] += np.histogram(grp['Xhits'][()][offset:offset+nhits],hbins)[0].astype(np.uint16)

            if rng.uniform()<split:
                X_test += [img]
                Y_test += [f[k]['Ypdf'][()].T]
            else:
                X_train += [img]
                Y_train += [f[k]['Ypdf'][()].T]

            if (i%100==0):
                ofname = '%s.shot_%i.dat'%(h5name,i)
                oYname = '%s.pdf_%i.dat'%(h5name,i)
                np.savetxt(ofname,img,fmt='%i',header=headstr)
                np.savetxt(oYname,f[k]['Ypdf'][()].T,fmt='%.3f',header=headstr)
    return X_train,Y_train,X_test,Y_test

def main():
    if len(sys.argv)<3:
        print('syntax:%s <infname.h5> <testsplit [0..1]> '%sys.argv[0])
        return
    infname = sys.argv[1]
    m = re.search('(^.*)\.h5',infname)
    if not m:
        print('failed file match')
        return
    densename = '%s_dense.h5'%m.group(1)
    split = float(sys.argv[2])
    X_train,Y_train,X_test,Y_test = sparse2dense(infname,split=split)
    with h5py.File(densename,'a') as f:
        f.create_dataset('Y_train',data=Y_train,dtype=np.float32)
        f.create_dataset('X_train',data=X_train,dtype=np.float32)
        f.create_dataset('Y_test',data=Y_test,dtype=np.float32)
        f.create_dataset('X_test',data=X_test,dtype=np.float32)
    return

if __name__ == '__main__':
    main()
