#!/usr/bin/python3

import time
import numpy as np
import sys
import joblib
import re
import os

from sklearn import metrics # remaining printout of GP metrics from main
import scipy.sparse
from sklearn.preprocessing import PolynomialFeatures,MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import PerturbativeUtils
import DataUtils

from MyPyClasses.InferenceClasses import SimpleInference

def main():

    printascii = False

    #./data_ave/ind_25-plate_tune_grid_Range_*/analyzed_data.hdf5
    m = re.match('(.*)/(ind_(.*)_.*/)analyzed_data.hdf5',sys.argv[-1])
    if not m:
        print('syntax: ./src/load_backinference.py <./data_ave/ind_25-plate_tune_grid_Range_*/analyzed_data.hdf5> ')
        return

    modelsfolder = '%s/backinference_simple%s'%(m.group(1),m.group(3))
    if not os.path.exists(modelsfolder):
        os.makedirs(modelsfolder)

    fnamelist = sys.argv[1:]
    print('processing %i files'%(len(fnamelist)))
    X_all,Y_all = DataUtils.loadData_logt_loge(fnamelist)

    #X_all = DataUtils.polyfeaturize(X_all,order=4)

    feature_range = (np.iinfo(np.int8).min+2,np.iinfo(np.int8).max-2) # giving a buffer of a couple seems to make the Y_0 output fall better inside of int8 bounds.
    si = SimpleInference(6,4)
    si.set_pipes(feature_range)

    #X = xpipe.fit_transform(X_all)
    X = si.pipe0.fit_transform(X_all)
    Y = Y_all.copy().reshape(-1,1)

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=.85)
    X_train1,X_train2,Y_train1,Y_train2 = train_test_split(X_train,Y_train,train_size=.25)

    print('####################### Fitting initial polynomial on X_train1 #################')
    si.fit_theta0(X_train1,Y_train1)
    print('theta from polynomial fitting:\t%s'%(si.theta0.T))

    print('####################### Fitting residual polynomial on X_train2 and Y_0 #################')
    stime = time.time()
    Y_0 = si.theta0.T.dot(X_train2.T).reshape(-1,1)
    X_train2_poly = si.pipe1.fit_transform(np.c_[X_train2[:,1:3].copy(),Y_0])
    si.fit_theta1(X_train2_poly,Y_train2 - Y_0)

    print('For single poly model with all samples it took %.3f seconds'%(time.time()-stime))
    print('########### OK, so far I have fit polytheta ###########')

    print('####################### Simple Polynomial Inference ###############')
    Y_0 = si.theta0.T.dot(X_test.T).reshape(-1,1)
    Y = Y_0.copy()
    ens_true = np.exp(Y_test)
    ens_0 = np.exp( Y )
    print('Polynomial only (Y_0) test set rms error [eV]\t%.3f'%(np.sqrt(np.sum(np.power(ens_true - ens_0,int(2)))/len(ens_true))))

    X_test_poly  = si.pipe1.transform(np.c_[X_test[:,1:3].copy(),Y_0])
    Y_res_poly = si.theta1.T.dot(X_test_poly.T).reshape(-1,1)

    infertime = 1000000.*(time.time()-stime)/float(X_test.shape[0])
    Y_poly = (Y_0 + Y_res_poly)
    ens_poly = np.exp(Y_poly)
    print('inference time Polynomial\t%.3f [musec/Sample]'%(infertime))
    print('Mean Polynomial ensemble test set rms error [eV]\t%.3f'%(np.sqrt(np.sum(np.power(ens_true - ens_poly,int(2)))/len(ens_true))))

    modelsfolder = '%s/backinference_simple%s'%(m.group(1),m.group(3))
    fname = '%s/simplemodel_logT2logE_%s.sav'%(modelsfolder,time.strftime('%Y.%m.%d.%H.%M'))
    joblib.dump(si,fname)


    return

if __name__ == '__main__':
    main()
