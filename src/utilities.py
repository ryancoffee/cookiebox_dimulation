#!/usr/bin/python3.5

import numpy as np;


def gauss(x):
    return np.exp(-1.0*np.power(x,int(2)))

def gauss(x,c,w):
    return np.exp(-np.power((x-c)/w,int(2)))

def highpass(f,c,w):
    inds = np.where((abs(f)>c-w/2.)*(abs(f)<c+w/2.))
    y = np.zeros(f.shape,dtype=float)
    y[inds] += np.power(np.sin((np.abs(f[inds])-c-w)/w*np.pi/2.),int(2))
    inds = np.where(abs(f)>=c+w/2.)
    y[inds] = 1. 
    return y

def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

def lowpass(f,c,w):
    inds = np.where((abs(f)>c-w/2.)*(abs(f)<c+w/2.))
    y = np.zeros(f.shape,dtype=float)
    y[inds] += np.power(np.cos((np.abs(f[inds])-c-w)/w*np.pi/2.),int(2))
    inds = np.where(abs(f)<=c-w/2.)
    y[inds] = 1.
    return y

