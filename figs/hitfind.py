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

def c2(freq,bwd):
    csqr = np.zeros(freq.shape[0],dtype=float) 
    inds = np.where(np.abs(freq)<bwd)
    csqr[inds] = np.power(np.cos(freq[inds]/bwd*np.pi/2.),int(2))
    return csqr

def sampleandhold(y,thresh,dead,e):
    result = np.zeros(y.shape,dtype=float)
    i = 0
    while i < e.shape[0]:
        ramp = e[i]
        result[i] = ramp
        if y[i] < thresh:
            while i<e.shape[0] and y[i]<0:
                if i<e.shape[0]:
                    ramp = e[i]
                    result[i] = ramp
                    hold = ramp
                    i += 1
            if i+int(dead+.5) < e.shape[0]:
                result[i:i+int(dead+.5)] = hold
                i += int(dead+.5)
        if i<e.shape[0]:
            ramp = e[i]
            result[i] = ramp
            i += 1
    return result

#def processfiles(filelist,c2,en,jac,b):
def processfiles(filelist,c2):
    dirlist = []
    filecounter = int(0)
    #nbins = jac.shape[0]
    nbins=1024
    hist = np.zeros(nbins,dtype=float)
    jac = np.zeros(nbins,dtype=float)
    en = np.zeros((0,),dtype=float)
    (en_low,en_high) = (0.25,11.25)
    for filename in filelist:
        m = re.search('(^.+/)(C\d--C\d.+).txt$',filename)
        if m:
            dirstr = m.group(1)
            fileheadstr = m.group(2)
            infile = open(filename,'rt')
            (scopename,scopemodel,arraytype,nsegments,nsamples_chk,xlabel,ylabel) = getheader(infile)
            if (nsamples_chk != c2.shape[0]):
                print("failed, {} is unequal nsamples for file {}".format(c2.shape[0],m.group(0)))
                continue
            d = np.loadtxt(infile,dtype=float,delimiter=',')
            t0=d[0,0]
            t=(d[:,0]-t0)*1e9
            y=d[:,1]
            dt = t[1]-t[0]
            infile.close()
            #f = np.fft.fftfreq(y.shape[0],dt)
            #BOX = 1.*(np.abs(f)<0.24)
            #Y = np.fft.fft(y)*c2
            #DY = 1j*f*Y*c2
            #outfile = dirstr + fileheadstr + '.fft'
            #np.savetxt(outfile,np.column_stack((f,np.abs(Y),np.abs(DY))),fmt='%.3e')
            #dy = np.fft.ifft(DY).real
            #num = np.fft.ifft(np.fft.fft(dy*t)*BOX).real
            #denom = np.fft.ifft(DY*BOX).real
            #inds = np.where(np.abs(denom)>0)
            #result = np.zeros(y.shape,dtype=float)
            #result[inds] = num[inds]/denom[inds]
            if en.shape[0] < t.shape[0]:
                en = np.zeros(t.shape[0],dtype=float)
                (ramp0,rampstart,rampstop) = (250,300,1300)
                inds = np.where((t>rampstart)*(t<rampstop))
                en[inds] = 2e4 * float(rampstart-ramp0)*np.power(t[inds]-t[ramp0],int(-2))
                (jac,_) = np.histogram(en,bins = nbins,range=(en_low,en_high))
            result_hold = sampleandhold(y,-.05,.5/dt,en)
            #outfile = dirstr + fileheadstr + '.expect'
            #np.savetxt(outfile,np.column_stack((t,y,result_hold)),fmt='%.6e')
            (h,b) = np.histogram(result_hold,bins = nbins,range=(en_low,en_high))
            h = (h-jac)
            inds = np.where(h>0)
            hist[inds] += 1
            #hist += (h-jac)

            
            filecounter +=1
            if (len(dirlist) > 0):
                if dirlist[-1] != dirstr:
                    dirlist.append(dirstr)
            else:
                dirlist.append(dirstr)

    return (dirlist,b,hist,jac)


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
    f = np.fft.fftfreq(nsamples,dt*1e9) #in GHz
    t = np.arange(nsamples)*dt
    bwd = 2.4
    cos2 = c2(f,bwd)
    data = np.zeros((nsamples,),dtype=float)
    nbins = 1024
    (dirlist,bins,hist,jac) = processfiles(sys.argv[1:],cos2)
    if m:
        headerStr = "\t".join(dirlist)
        outfilename = dirlist[-1] + 'hist.out'
        np.savetxt(outfilename,np.column_stack((bins[:-1],hist,jac)),header=headerStr)
    return

if __name__=="__main__":
    main()

