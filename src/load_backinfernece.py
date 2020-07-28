#!/usr/bin/python3

import time
import numpy as np
import sys
import joblib
import re
import os

from sklearn import metrics # remaining printout of GP metrics from main
import scipy.sparse

import PerturbativeUtils
import DataUtils

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def featurize(x):
    return np.column_stack((x.reshape(-1,1),
                np.power(x,int(2)).reshape(-1,1),
                np.power(x,int(3)).reshape(-1,1),
                np.power(x,int(4)).reshape(-1,1)
                ))
def main():

    printascii = False
    tof_jitter = 0.1000 # 75ps of edge uncertainty from edge finding algorithm
    nmodels = 8 
    nsamples = 200

    #./data_ave/ind_25-plate_tune_grid_Range_*/analyzed_data.hdf5
    m = re.match('(.*)/(ind.*(logos_.*)/)analyzed_data.hdf5',sys.argv[-1])
    if not m:
        print('syntax: ./src/load_backinference.py <./data_ave/ind_25-plate_tune_grid_Range_[only one of these]/analyzed_data.hdf5> ')
        return

    modelsfolder = '%s/backinference_%s'%(m.group(1),m.group(3))
    if not os.path.exists(modelsfolder):
        os.makedirs(modelsfolder)

    X_all,Y_all = DataUtils.loadT2Edata()

    X_all = featurize(X_all)

    X_all,Y_all,X_scaler,Y_scaler = DataUtils.minmaxscaledata(X_all,Y_all,feature_range = (-1,1))
    X_train,X_test,Y_train,Y_test = DataUtils.reservesplit(X_all,Y_all,reserve = .4)
    X_train_gp,X_train_taylor,Y_train_gp,Y_train_taylor = DataUtils.reservesplit(X_train,Y_train,reserve = .25)

    theta = DataUtils.pseudoinversemethod(X_train_taylor,Y_train_taylor)
    print(theta.T)
    print(X_train_taylor.shape[0])
    print(X_train_gp.shape[0])


    fname_list_gp = PerturbativeUtils.fit_gp_t2e_ensemble(X_train_gp,Y_train_gp,maternnu=1.5,theta_model0=theta,modelfolder=modelsfolder,nmodels=nmodels,nsamples=nsamples)

    gp_t2e_models = []
    for i in range(nmodels):
        gp_t2e_models += [joblib.load(fname_list_gp[i])]

    X_scaler.inverse_transform(X_test)
    X_test_noise = featurize( np.log( np.exp(X_test[:,0]) + np.random.normal(0,tof_jitter,X_test.shape[0]) ) )
    X_scaler.transform(X_test_noise)

    Y_pred,Y_pred_std,Y_pred_std_hist_array = PerturbativeUtils.ensemble_vote_t2e(X_test_noise,gp_t2e_models,theta_model0 = theta,elitism = 0.5)

    np.savetxt('%s/backinfer.dat'%(modelsfolder),np.column_stack((
        Y_scaler.inverse_transform(Y_test.reshape(-1,1)),
        Y_scaler.inverse_transform(Y_pred.reshape(-1,1)),
        X_scaler.inverse_transform(X_test_noise)
        ))
        )

    true = np.exp(Y_test).flatten()
    pred = np.exp(Y_pred).flatten()
    residual = pred - true
    #truebins = np.logspace(-1,2,41)
    truebins = np.linspace(0,256,51)
    #resbins = np.concatenate((-np.logspace(0.5,-2,11),np.logspace(-2,0.5,11)),axis=0)
    resbins = np.linspace(-.25,.25,81)
    #reshist,xedges,yedges = np.histogram2d(true.reshape(-1),residual.reshape(-1),bins = (truebins,resbins), density = True)
    reshist,xedges,yedges = np.histogram2d(true.reshape(-1),residual.reshape(-1),bins = (truebins,resbins), density = True)
    X,Y = np.meshgrid((xedges[:-1]+xedges[1:])/2.,(yedges[:-1]+yedges[-1:])/2.)

    i,j,z = scipy.sparse.find(scipy.sparse.csr_matrix(reshist))
    x = xedges[i]
    y = yedges[j]

    #np.savetxt('%s/backinfer.hist.dat'%(modelsfolder),np.column_stack((Y.flatten(),X.flatten(),reshist.flatten())))
    np.savetxt('%s/backinfer.hist.dat'%(modelsfolder),reshist)

    print('Left to do: ')
    print('Print our the histogram of residuals versus true energy')

    return

if __name__ == '__main__':
    main()
