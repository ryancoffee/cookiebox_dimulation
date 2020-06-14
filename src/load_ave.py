#!/usr/bin/python3

import time

import numpy as np
import h5py
import sys
import random
import math
from sklearn.multioutput import MultiOutputRegressor
from sklearn import linear_model
from sklearn import metrics
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, ConstantKernel

# Exploring Kernel Ridge Regression -- https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_compare_gpr_krr.html#sphx-glr-auto-examples-gaussian-process-plot-compare-gpr-krr-py
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)


def ydetToLorenzo(y):
    '''
    Lorenzo, e.g. Tixel detector, has 48x48 pixels per tile, each pixel is 100 microns square, r is in meters from Ave/Naoufal
    '''
    q = [2.*math.pi*random.random() for i in range(len(y))]
    return (np.array(q),1e4*np.array(r)*np.cos(q),1e4*np.array(r)*np.sin(q))

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



def loaddata():
    x_all = []
    y_all = []
    if len(sys.argv) < 2:
        print("syntax: %s <datafile>"%(sys.argv[0]) )
        return x_all,y_all

    for fname in sys.argv[1:]:
        f = h5py.File(fname,'r') #  IMORTANT NOTE: it looks like some h5py builds have a resouce lock that prevents easy parallelization... tread lightly and load files sequentially for now.
        for vsetting in list(f.keys()):
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
            validinds = np.where((abs(xsplat-185)<2) * (emat>0))
            nfeatures = 3
            ntruths = 2
            featuresvec = np.zeros((len(xsplat[validinds]),nfeatures),dtype=float)
            truthsvec = np.zeros((len(xsplat[validinds]),ntruths),dtype=float)
            featuresvec[:,0] = vsetvec[validinds]
            featuresvec[:,1] = np.log(emat[validinds])
            featuresvec[:,2] = amat[validinds]
            truthsvec[:,0] = np.log(tdata[validinds])
            truthsvec[:,1] = ydata[validinds]
            if len(x_all)<1:
                x_all = np.copy(featuresvec)
                y_all = np.copy(truthsvec)
            else:
                x_all = np.row_stack((x_all,featuresvec))
                y_all = np.row_stack((y_all,truthsvec))
    return x_all,y_all




def main():
    X_all = []
    Y_all = []
    X_all,Y_all = loaddata()
    if len(Y_all)<1:
        print("no data loaded")
        return
    print("data loaded\tcontinuing to fitting")
    X_train,X_test,X_valid,X_oob,Y_train,Y_test,Y_valid,Y_oob = katiesplit(X_all,Y_all)
    np.savetxt('check.dat',np.column_stack((X_train,Y_train)))
    stime = time.time()
    linmodel = linear_model.LinearRegression().fit(X_train[-2000:,:], Y_train[-2000:])
    print("Time for linear model fitting: %.3f" % (time.time() - stime))
    stime = time.time()
    Y_test_pred = linmodel.predict(X_test)
    Y_valid_pred = linmodel.predict(X_valid)
    print("Time for linear model infernce: %.3f" % (time.time() - stime))
    print ("linear model score (test): ", metrics.r2_score(Y_test, Y_test_pred))
    print ("linear model score (validate): ", metrics.r2_score(Y_valid, Y_valid_pred))

    # Fit KernelRidge with parameter selection based on 5-fold cross validation
    param_grid = {"alpha": [1e0, 1e-1, 1e-2, 1e-3],
                  "kernel": [RationalQuadratic(l, a)
                  for l in np.logspace(-2, 2, 10)
                  for a in np.logspace(0, 2, 10)]}
    kr = GridSearchCV(KernelRidge(), param_grid=param_grid)
    stime = time.time()
    kr.fit(X_train[2000:4000,:], Y_train[2000:4000,:])
    print("Time for KRR fitting: %.3f" % (time.time() - stime))
    #print("KRR kernel: %s" % kr.kernel_)
    stime = time.time()
    Y_test_pred = kr.predict(X_test)
    Y_valid_pred = kr.predict(X_valid)
    print("Time for KRR inference: %.3f" % (time.time() - stime))
    print ("KRR model score (test): ", metrics.r2_score(Y_test, Y_test_pred))
    print ("KRR model score (validate): ", metrics.r2_score(Y_valid, Y_valid_pred))

    k1 = 1.0**2 * RBF(length_scale=1.0) # long term smooth trend
    k2 = 0.5**2 * RationalQuadratic(length_scale=5., alpha=1) # medium term irregularity
    k3 = 0.18**2 * RBF(length_scale=0.134) + WhiteKernel(noise_level=0.1**2)  # noise terms
    k4 = ConstantKernel(constant_value = 10 ) # constant shift
    #kernel_gp = k1 + k2 + k3 + k4
    kernel_gp = k3 + k4
    multiout_gp = GaussianProcessRegressor(kernel=kernel_gp, alpha=0, normalize_y=True)
    stime = time.time()
    multiout_gp.fit(X_train[:2000,:], Y_train[:2000,:])
    print("Time for GP fitting: %.3f" % (time.time() - stime))

    print("GPML kernel: %s" % multiout_gp.kernel_)
    print("Log-marginal-likelihood: %.3f" % multiout_gp.log_marginal_likelihood(multiout_gp.kernel_.theta))
    stime = time.time()
    Y_test_pred = multiout_gp.predict(X_test)
    Y_valid_pred = multiout_gp.predict(X_valid)
    print("Time for KRR inference: %.3f" % (time.time() - stime))
    print('GP score (test): ',  metrics.r2_score(Y_test,Y_test_pred))
    print('GP score (validate): ',  metrics.r2_score(Y_valid,Y_valid_pred))
    Y_test[:,0]=np.exp(Y_test[:,0])
    Y_valid[:,0]=np.exp(Y_valid[:,0])
    print('GP score (test) (exp): ',  metrics.r2_score(Y_test[:,0],Y_test_pred[:,0]))
    print('GP score (validate) (exp): ',  metrics.r2_score(Y_validi[:,0],Y_valid_predi[:,0]))

    return
            
    '''
    features = []
    for v in x:
        features.append((int((v-center)*8) , np.power(int((v-center)*8),int(2))//100 ))
    return features
    '''
            
if __name__ == '__main__':
    main()

