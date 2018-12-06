#!/usr/bin/python3

import glob
import re

#filepath = '/data/projects/slac/hamamatsu/dec2018/ave1/'
filepath = '../data_fs/ave1/'
filematch = filepath + 'C1*.txt'

filelist = glob.glob(filematch)

for f in filelist:
    m = re.search('(.+).txt$',f)
    fi = open(f, "r")
    fo_name = m.group(1) + '.dat'
    fo = open(fo_name, "w")
    for i in range(5):
        headline = '# ' + fi.readline()
        fo.write(headline) 
    for line in fi:
        fo.write(line)
    fi.close()
    fo.close()

