import numpy as np
import h5py
import sys
import random
import math
import re

from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_regression

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def prependOnesToX(x):
    # note, this may be a shallow bias add
    x_bias = x.copy()
    x_bias = np.column_stack((np.ones(x.shape[0],dtype=float),x_bias))
    return x_bias

def pseudoinversemethod(x,y):
    theta = np.linalg.pinv(prependOnesToX(x)).dot(y)
    return theta

def ydetToLorenzo(y):
    #Lorenzo, e.g. Tixel detector, has 48x48 pixels per tile, each pixel is 100 microns square, r is in meters from Ave/Naoufal
    q = [2.*math.pi*random.random() for i in range(len(y))]
    return (np.array(q),1e4*np.array(r)*np.cos(q),1e4*np.array(r)*np.sin(q))

def reservesplit(x,y,reserve = .2):
    sz = x.shape[0] 
    inds = np.arange(x.shape[0])
    np.random.shuffle(inds)
    splitind = int(sz*(1. - reserve))
    maininds = inds[:splitind]
    reserveinds = inds[splitind:]
    return x[maininds,:],x[reserveinds,:],y[maininds,:],y[reserveinds,:]

def katiesplit(x,y):
    sz = x.shape[0] 
    inds = np.arange(x.shape[0])
    np.random.shuffle(inds)
    traininds = inds[:sz//4]
    testinds = inds[sz//4:2*sz//4]
    validateinds = inds[2*sz//4:3*sz//4]
    oobinds = inds[3*sz//4:]
    x_train = x[traininds,:]
    y_train = y[traininds,:]
    x_test = x[testinds,:]
    y_test = y[testinds,:]
    x_validate = x[validateinds,:]
    y_validate = y[validateinds,:]
    x_oob = x[oobinds,:]
    y_oob = y[oobinds,:]
    return (x_train,x_test,x_validate,x_oob,y_train,y_test,y_validate,y_oob)


def loadT2Edata_tixel():
    x_all = []
    y_all = []
    if len(sys.argv) < 2:
        print("syntax: %s <datafile>"%(sys.argv[0]) )
        return x_all,y_all

    for fname in sys.argv[1:]:
        f = h5py.File(fname,'r') #  IMORTANT NOTE: it looks like some h5py builds have a resouce lock that prevents easy parallelization... tread lightly and load files sequentially for now.
        if True: #bypassing the for loop on vsetting
            print('bypassing the for loop on vsetting')
            vsetting = list(f.keys())[3]
        #for vsetting in list(f.keys()): # restricting to only the closest couple vsettings to optimal... correction, now only the central (optimal) one, but for each 'logos'
            elist = list(f[vsetting]['energy'])
            alist = list(f[vsetting]['angle'])
            amat = np.tile(alist,(len(elist),1)).flatten()
            emat = np.tile(elist,(len(alist),1)).T.flatten()
            tdata = f[vsetting]['t_offset'][()].flatten()
            ydata = f[vsetting]['y_detector'][()].flatten()
            xsplat = f[vsetting]['splat']['x'][()].flatten()
            vset = f[vsetting][ list(f[vsetting].keys())[0] ][-1][1] # eventually, extract the whole voltage vector as a feature vector for use in GP inference
            vsetvec = np.ones(xsplat.shape,dtype=float)*vset
            
            # range of good splat[x] 182.5 187
            # for the back transform, using only the central 27mm diameter
            validinds = np.where((xsplat>182.5) * (xsplat<187) * (emat>0) * (abs(ydata)<.0135) )
            nfeatures = 2
            ntruths = 1
            featuresvec = np.zeros((len(xsplat[validinds]),nfeatures),dtype=float)
            truthsvec = np.zeros((len(xsplat[validinds]),ntruths),dtype=float)
            #featuresvec[:,0] = np.log(-1.*vsetvec[validinds])
            featuresvec[:,0] = np.log(tdata[validinds])
            featuresvec[:,1] = ydata[validinds]
            truthsvec[:,0] = np.log(emat[validinds])
            
            if len(x_all)<1:
                x_all = featuresvec.copy()
                y_all = truthsvec.copy()
            else:
                x_all = np.row_stack((x_all,featuresvec))
                y_all = np.row_stack((y_all,truthsvec))
    return np.array(x_all),np.array(y_all)

def loadT2Escaledata():
    x_all,y_all = loadT2Edata()
    return scaledata(x_all,y_all)


def loadT2Edata():
    x_all = []
    y_all = []
    if len(sys.argv) < 2:
        print("syntax: %s <datafile>"%(sys.argv[0]) )
        return x_all,y_all

    for fname in sys.argv[1:]:
    #if True:# bypassing multiple retardation files
        #print('bypassing multiple retardation files')
        #fname = sys.argv[1]
        f = h5py.File(fname,'r') #  IMORTANT NOTE: it looks like some h5py builds have a resouce lock that prevents easy parallelization... tread lightly and load files sequentially for now.
        if True: #bypassing the for loop on vsetting
            print('bypassing the for loop on vsetting')
            vsetting = list(f.keys())[3]
        #for vsetting in list(f.keys()): # restricting to only the closest couple vsettings to optimal... correction, now only the central (optimal) one, but for each 'logos'
            elist = list(f[vsetting]['energy'])
            alist = list(f[vsetting]['angle'])
            amat = np.tile(alist,(len(elist),1)).flatten()
            emat = np.tile(elist,(len(alist),1)).T.flatten()
            tdata = f[vsetting]['t_offset'][()].flatten()
            ydata = f[vsetting]['y_detector'][()].flatten()
            xsplat = f[vsetting]['splat']['x'][()].flatten()
            vset = f[vsetting][ list(f[vsetting].keys())[0] ][-1][1] # eventually, extract the whole voltage vector as a feature vector for use in GP inference
            vsetvec = np.ones(xsplat.shape,dtype=float)*vset

            # range of good splat[x] 182.5 187
            # for the back transform, using only the central 27mm diameter
            validinds = np.where((xsplat>182.5) * (xsplat<187) * (emat>0) * (abs(ydata)<.0135) )
            nfeatures = 1
            ntruths = 1
            featuresvec = np.zeros((len(xsplat[validinds]),nfeatures),dtype=float)
            truthsvec = np.zeros((len(xsplat[validinds]),ntruths),dtype=float)
            #featuresvec[:,0] = np.log(-1.*vsetvec[validinds])
            featuresvec[:,0] = np.log(tdata[validinds])
            truthsvec[:,0] = np.log(emat[validinds])
            
            if len(x_all)<1:
                x_all = featuresvec.copy()
                y_all = truthsvec.copy()
            else:
                x_all = np.row_stack((x_all,featuresvec))
                y_all = np.row_stack((y_all,truthsvec))

    return np.array(x_all),np.array(y_all)


def loaddata():
    x_all = []
    y_all = []
    if len(sys.argv) < 2:
        print("syntax: %s <datafile>"%(sys.argv[0]) )
        return x_all,y_all

    for fname in sys.argv[1:]:
        f = h5py.File(fname,'r') #  IMORTANT NOTE: it looks like some h5py builds have a resouce lock that prevents easy parallelization... tread lightly and load files sequentially for now.
        for vsetting in list(f.keys())[3]: # restricting to only the closest vsettings to optimal... correction, now only the central (optimal) one, but for each 'logos'
        #for vsetting in list(f.keys()): # restricting to only the closest couple vsettings to optimal... correction, now only the central (optimal) one, but for each 'logos'
            elist = list(f[vsetting]['energy'])
            alist = list(f[vsetting]['angle'])
            amat = np.tile(alist,(len(elist),1)).flatten()
            emat = np.tile(elist,(len(alist),1)).T.flatten()
            tdata = f[vsetting]['t_offset'][()].flatten()
            ydata = f[vsetting]['y_detector'][()].flatten()
            xdata = f[vsetting]['x_detector'][()].flatten()
            xsplat = f[vsetting]['splat']['x'][()].flatten()
            vset = f[vsetting][ list(f[vsetting].keys())[0] ][-1][1] # eventually, extract the whole voltage vector as a feature vector for use in GP inference
            vsetvec = np.ones(xsplat.shape,dtype=float)*vset
            # range of good spat[x] 182.5 187
            validinds = np.where((xsplat>182.5) * (xsplat<187) * (emat>0) * (abs(ydata)<.020))
            nfeatures = 3
            ntruths = 2
            featuresvec = np.zeros((len(xsplat[validinds]),nfeatures),dtype=float)
            truthsvec = np.zeros((len(xsplat[validinds]),ntruths),dtype=float)
            featuresvec[:,0] = np.log(-1.*vsetvec[validinds])
            featuresvec[:,1] = np.log(emat[validinds])
            featuresvec[:,2] = amat[validinds]
            truthsvec[:,0] = np.log(tdata[validinds])
            truthsvec[:,1] = ydata[validinds]
            truthsvec[:,1] *= 1e3 # converting to mm
            if len(x_all)<1:
                x_all = np.copy(featuresvec)
                y_all = np.copy(truthsvec)
            else:
                x_all = np.row_stack((x_all,featuresvec))
                y_all = np.row_stack((y_all,truthsvec))
    return np.array(x_all),np.array(y_all)

def minmaxscaledata(x,y,feature_range = (1,3)):
    Xscaler = preprocessing.MinMaxScaler(copy=False,feature_range = feature_range).fit(x)
    Yscaler = preprocessing.MinMaxScaler(copy=False,feature_range = feature_range).fit(y.reshape(-1,1))
    x = Xscaler.transform(x)
    y = Yscaler.transform(y.reshape(-1,1))
    return x,y,Xscaler,Yscaler

def scaledata(x,y):
    Xscaler = preprocessing.StandardScaler(copy=False).fit(x)
    Yscaler = preprocessing.StandardScaler(copy=False).fit(y.reshape(-1,1))
    x = Xscaler.transform(x)
    y = Yscaler.transform(y.reshape(-1,1))
    return x,y,Xscaler,Yscaler

def loadscaledata(print_mi = False):
    x_all,y_all = loaddata()
    Xscaler = preprocessing.StandardScaler(copy=False).fit(x_all)
    Yscaler = preprocessing.StandardScaler(copy=False).fit(y_all)
    #Xscaler = preprocessing.MinMaxScaler((0,64),copy=False).fit(X_train)
    #Yscaler = preprocessing.MinMaxScaler((0,64),copy=False).fit(Y_train)
    x_all = Xscaler.transform(x_all)
    y_all = Yscaler.transform(y_all)

    if print_mi:
        mi_tof = mutual_info_regression(x_all,y_all[:,0])
        mi_tof /= np.max(mi_tof)
        print('mi for tof time\t',mi_tof)
        mi_pos = mutual_info_regression(x_all,y_all[:,1])
        mi_pos /= np.max(mi_pos)
        print('mi for y_position',mi_pos)

    return x_all,y_all,Xscaler,Yscaler

def crosscorrelation(fname,x,y):
        if x.shape[0] != y.shape[0]:
            print('Failed for x.shape %s and y.shape %s'%(str(x.shape),str(y.shape)))
            return False
        m = np.column_stack((x,y))
        (sz,nf) = m.shape
        c = np.ones((nf,nf),dtype=float)  
        for i in range(c.shape[0]):
            for j in range(1,nf//2):
                c[i,(i+j)%nf] = c[(i+j)%nf,i] = np.correlate(m[:,i],m[:,(i+j)%nf],mode='valid')/sz
        
        print(c)
        headerstring = 'nx_features = %i\tny_features = %i'%(x.shape[1],y.shape[1])
        np.savetxt(fname,c,fmt='%.3f',header=headerstring)
        return True

