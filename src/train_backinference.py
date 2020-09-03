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

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def main():

    printascii = False
    order = 4
    order_ensemble = 4

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
    xpipe = Pipeline([
        ('polyFeature',PolynomialFeatures(degree=order,interaction_only=False,include_bias=True)),
        ('int8scaler',MinMaxScaler(copy=False,feature_range = feature_range))
        ])
    polypipe = Pipeline([
        ('polyFeature',PolynomialFeatures(degree=order_ensemble,interaction_only=False,include_bias=True)),
        ('int8scaler',MinMaxScaler(copy=False,feature_range = feature_range))
        ])

    X = xpipe.fit_transform(X_all)
    Y = Y_all.copy().reshape(-1,1)

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=.85)
    X_train1,X_train2,Y_train1,Y_train2 = train_test_split(X_train,Y_train,train_size=.25)

    print('####################### Fitting initial polynomial on X_train1 #################')
    theta = np.linalg.pinv(X_train1).dot(Y_train1)
    Y_0 = theta.T.dot(X_train1.T).reshape(-1,1)
    Y_residual = Y_train1 - Y_0
    print('theta from polynomial fitting:\t%s'%(theta.T))

    print('####################### Fitting residual polynomial on X_train2 and Y_0 #################')
    stime = time.time()
    Y_0 = theta.T.dot(X_train2.T).reshape(-1,1)
    X_train2_poly = polypipe.fit_transform(np.c_[X_train2[:,1:3].copy(),Y_0])
    polytheta = np.linalg.pinv( X_train2_poly ).dot( Y_train2 - Y_0 )

    print('For single poly model with all samples it took %.3f seconds'%(time.time()-stime))
    print('########### OK, so far I have fit polytheta ###########')

    print('####################### Simple Polynomial Inference (and ensemble) ###############')
    Y_0 = theta.T.dot(X_test.T).reshape(-1,1)
    Y = Y_0.copy()
    ens_true = np.exp(Y_test)
    ens_0 = np.exp( Y )
    print('Polynomial only (Y_0) test set rms error [eV]\t%.3f'%(np.sqrt(np.sum(np.power(ens_true - ens_0,int(2)))/len(ens_true))))

    X_test_poly  = polypipe.transform(np.c_[X_test[:,1:3].copy(),Y_0])
    Y_res_poly = polytheta.T.dot(X_test_poly.T).reshape(-1,1)

    infertime = 1000000.*(time.time()-stime)/float(X_test.shape[0])
    Y_poly = (Y_0 + Y_res_poly)
    ens_poly = np.exp(Y_poly)
    print('inference time Polynomial\t%.3f [musec/Sample]'%(infertime))
    print('Mean Polynomial ensemble test set rms error [eV]\t%.3f'%(np.sqrt(np.sum(np.power(ens_true - ens_poly,int(2)))/len(ens_true))))

    return

if __name__ == '__main__':
    main()
