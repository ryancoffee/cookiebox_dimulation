#!/usr/bin/python3

import h5py
import numpy as np
import re
import sys

def fname(pathhead,i): 
    return '%s%i/analyzed_data.hdf5'%(pathhead,i)

def getpots(name): 
    d = [float(0)]
    f = h5py.File(name,'r')
    kstr = [k for k in f['data_run_3'].keys()][0]
    d += [v for v in f['data_run_3'][kstr][()][:,1]]
    f.close()
    m = re.search('^(.*) potential',kstr)
    if m:
        d += [float(m.group(1))]
    return d

def main():
    i = 39
    pathhead = './ind_25-plate_tune_grid_NM_log40Ret_'
    if len(sys.argv)>1:
        pathhead = sys.argv[1]
    if len(sys.argv)>2:
        i = int(sys.argv[2])
    _ = [print(p) for p in getpots(fname(pathhead,i))]
    return

if __name__ == '__main__':
    main()
