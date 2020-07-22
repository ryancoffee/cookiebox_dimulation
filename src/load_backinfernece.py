#!/usr/bin/python3

import time
import numpy as np
import sys
import joblib
import re
import os

from sklearn import metrics # remaining printout of GP metrics from main

import PerturbativeUtils
import DataUtils

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def featurize(x):
    x = np.concatenate((x,
                np.power(x[:,1],int(2)).reshape(-1,1),
                np.power(x[:,0],int(2)).reshape(-1,1),
                np.power(x[:,1],int(3)).reshape(-1,1),
                np.power(x[:,0],int(3)).reshape(-1,1),
                (x[:,0]*x[:,1]).reshape(-1,1),
                (x[:,0]*np.power(x[:,1],int(2))).reshape(-1,1),
                (x[:,1]*np.power(x[:,0],int(2))).reshape(-1,1)
                ),axis=1)
    return x

def main():

    printascii = False

    #./data_ave/ind_25-plate_tune_grid_Range_*/analyzed_data.hdf5
    m = re.match('(.*)/(ind.*/)analyzed_data.hdf5',sys.argv[-1])
    if not m:
        print('syntax: ./src/load_backinference.py <./data_ave/ind_25-plate_tune_grid_Range_*/analyzed_data.hdf5> ')
        return

    modelsfolder = '%s/backinference'%(m.group(1))
    if not os.path.exists(modelsfolder):
        os.makedirs(modelsfolder)

    X_all,Y_all = DataUtils.loadT2Edata()

    X_all = featurize(X_all)

    X_all,Y_all,X_scaler,Y_scaler = DataUtils.minmaxscaledata(X_all,Y_all,feature_range = (-1,1))
    X_train,X_test,Y_train,Y_test = DataUtils.reservesplit(X_all,Y_all,reserve = .2)
    X_train_gp,X_train_taylor,Y_train_gp,Y_train_taylor = DataUtils.reservesplit(X_train,Y_train,reserve = .6)

    theta = DataUtils.pseudoinversemethod(X_train_taylor,Y_train_taylor)
    print(theta.T)
    print(X_train_taylor.shape[0])
    print(X_train_gp.shape[0])

    nmodels = 32 

    fname_list_gp = PerturbativeUtils.fit_gp_t2e_ensemble(X_train_gp,Y_train_gp,maternnu=1.5,theta_model0=theta,modelfolder=modelsfolder,nmodels=nmodels,nsamples=300)

    gp_t2e_models = []
    for i in range(nmodels):
        gp_t2e_models += [joblib.load(fname_list_gp[i])]

    X_scaler.inverse_transform(X_test)
    X_test_tof = np.log( np.exp(X_test[:,1]) + np.random.normal(0,.075,X_test.shape[0]) )
    X_test_noise = np.column_stack((X_test[:,0],X_test_tof))
    X_test_noise = featurize(X_test_noise)
    X_scaler.transform(X_test_noise)

    Y_pred,Y_pred_std,Y_pred_std_hist_array = PerturbativeUtils.ensemble_vote_t2e(X_test_noise,gp_t2e_models,theta_model0 = theta,elitism = 0.2)

    np.savetxt('%s/testpseudoinv.dat'%(modelsfolder),np.column_stack((
        Y_scaler.inverse_transform(Y_test.reshape(-1,1)),
        Y_scaler.inverse_transform(Y_pred.reshape(-1,1)),
        X_scaler.inverse_transform(X_test_noise)
        ))
        )

    print('Left to do: Fit the residuals with a GP... or ensemble of GPs')
    print('\tadd noise to the X_test[:,1] that is equivalent to .100 ns before log() and apply model, compare to nonoise truth')
    print('\tthat should give the error more honestly since now is it unrealistically excellent resolution for well above 100 eV')

    return

if __name__ == '__main__':
    main()
