#!/usr/bin/python3

import numpy as np
import sys
import re

def main():
    if len(sys.argv) < 2:
        print('I need a chunk to expand into images')
        return
    for chunkname in sys.argv[1:]:
        f = open(chunkname,'rc')
        fout = open(chunkname,'wc')
        headstr = f.readline()
        headstr += f.readline()
        imgnum = 0
        for line in f.readline():
            chanstringarray = line.strip().split(':')
            nchannels = len(chanstringarray)
            thislen = 0
            imgwidth = 10
            data = np.zeros((nchannels,imgwidth),dtype=float)
            for i in range(len(chanstringarray)):
                vstrings = chstringarray[i].strip().split(' ')
                thislen = len(vstrings)
                if imgwidth < thislen:
                    data = np.column_stack((data,np.zeros(nchannels,thislen-imgwidth)),dtype=float)
                data[i,:thislen] = vstrings
            np.savetxt(chunkname + '.img{}'.format(imgnum),data,fmt='%.3f',header = headstr)
            imgnum += 1
        f.close()
    return

if __name__ = '__main__':
    main()
