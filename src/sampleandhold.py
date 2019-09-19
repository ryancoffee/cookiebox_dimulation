#!/usr/bin/python3

import numpy as np
import sys
import re

def holdvals(times,t0,rc=10):
    result = np.zeros(times.shape)
    inds = np.where(times>t0)
    result[inds] = np.exp(-(times[inds]-t0)/rc)
    return result

def main():
    if len(sys.argv) < 2:
        print('need a file of times to process')
        return
    times = np.loadtxt(sys.argv[1])
    result = holdvals(times,50,500)
    np.savetxt('temp.out',result,fmt='%e')
    return

if __name__ == '__main__':
    main()
