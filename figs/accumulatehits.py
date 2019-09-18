#!/usr/bin/python3

import numpy as np
import sys
import re

global nsamples 
global dt

def getheader(infile):
    line = infile.readline()
    listvals = [x.strip() for x in line.split(',')]
    (scopename,scopemodel,arraytype) = (listvals[0],listvals[1],listvals[2]) 
    line = infile.readline()
    listvals = [x.strip() for x in line.split(',')]
    (nsegments,nsamples) = (int(listvals[1]),int(listvals[3]))
    #nsegments = int(nsegmentsStr)
    #nsamples = int(nsamplesStr)
    line = infile.readline()
    line = infile.readline()
    #(dummy1,dummy2,dummy3) = [x.strip for x in line.split(',')]
    #(dummy1,dummy2,dummy3) = [x.strip for x in line.split(',')]
    line = infile.readline()
    (xlabel,ylabel) = [x.strip() for x in line.split(',')]
    return (scopename,scopemodel,arraytype,nsegments,nsamples,xlabel,ylabel)

def processfiles(filelist,data):
    dirlist = []
    filecounter = int(0)
    for filename in filelist:
        m = re.search('(^.+/)(C\d--C\d.+).txt$',filename)
        if m:
            dirstr = m.group(1)
            fileheadstr = m.group(2)
            infile = open(filename,'rt')
            (scopename,scopemodel,arraytype,nsegments,nsamples_chk,xlabel,ylabel) = getheader(infile)
            if (nsamples_chk != data.shape[0]):
                print("failed, {} are unequal nsamples for file {}".format(data.shape[0],m.group(0)))
                continue
            data += np.loadtxt(infile,dtype=float,delimiter=',',usecols=1)
            infile.close()
            filecounter +=1
            if (len(dirlist) > 0):
                if dirlist[-1] != dirstr:
                    dirlist.append(dirstr)
            else:
                dirlist.append(dirstr)
    return (dirlist,data)


def main():
    if len(sys.argv)<2:
        print("I need a list of files to process")
        return

    checkfile = sys.argv[1]
    m = re.search('(^.+/)(C\d--C\d.+).txt$',checkfile)
    if m:
        infile = open(m.group(0),'rt')
        (scopename,scopemodel,arraytype,nsegments,nsamples,xlabel,ylabel) = getheader(infile)
        t = np.loadtxt(infile,dtype=float,delimiter=',',usecols=0)
        dt = t[1]-t[0]
        infile.close()
    print(nsamples,dt)
    t = np.arange(nsamples)*dt
    data = np.zeros((nsamples,),dtype=float)
    (dirlist,data) = processfiles(sys.argv[1:],data)
    headerStr = "\t".join(dirlist)
    outname = dirlist[0] + 'accum.dat'
    np.savetxt(outname,np.column_stack((t,data)),fmt="%.3e",header=headerStr)
    return

if __name__=="__main__":
    main()

