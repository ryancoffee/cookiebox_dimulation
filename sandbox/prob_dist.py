#!/usr/bin/python3

import numpy as np

def gauss(x,w,c):
    return np.exp(-((x.astype(float)-c)/w)**2)

def main():
    x = np.arange(2048,dtype=int)
    y = 1024*gauss(x,128,256) + 1024*gauss(x,256,1500)
    i = np.cumsum(y)
    '''
    OK, take the cumulative sum, i, and mirror it, discrete cosine transform, fill x4 with zeros, idct(), and now do the mapping but report val //4
    '''
    scale = 2047./i[-1]
    i *= scale

    ## Y = A X; think of X as a one-hot encoded for a mapping
    ## then 
    ## y = f(x) determined the one-hot y encoded reciever 'pixel'
    ##

    X = np.eye(len(x),dtype=int)
    Y = np.zeros(X.shape,dtype=int)
    for c in range(X.shape[0]):
        Y[int(i[c]),c] = 1
    np.savetxt('mask.dat',Y.T,fmt='%i')

    Yinv = Y.T

    ndraws = 512
    Xdraw = np.eye(2048)
    inds=np.random.uniform(0,2048,ndraws).astype(int)
    H = Yinv.dot(Xdraw[:,inds])
    h = np.sum(H,axis=1)
    for j in range(10):
        inds=np.random.uniform(0,2048,ndraws).astype(int)
        H = Yinv.dot(Xdraw[:,inds])
        h += np.sum(H,axis=1)
    out = np.column_stack((x,y,i,h))
    np.savetxt('temp.dat',out,fmt='%i')
    np.savetxt('trans.dat',H,fmt='%i')

        




    return

if __name__ == '__main__':
    main()
