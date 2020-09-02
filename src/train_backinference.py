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

import PerturbativeUtils
import DataUtils

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def main():

    printascii = False
    tof_jitter = 0.1000 # 75ps of edge uncertainty from edge finding algorithm
    nmodels = 16 
    nmodels_gp = 16 
    nsamples = 512
    order = 4
    includeingp = 6

    #./data_ave/ind_25-plate_tune_grid_Range_*/analyzed_data.hdf5
    m = re.match('(.*)/(ind.*(_.*)/)analyzed_data.hdf5',sys.argv[-1])
    if not m:
        print('syntax: ./src/load_backinference.py <./data_ave/ind_25-plate_tune_grid_Range_*/analyzed_data.hdf5> ')
        return

    modelsfolder = '%s/backinference_%s'%(m.group(1),m.group(3))
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
        ('polyFeature',PolynomialFeatures(degree=2*order,interaction_only=False,include_bias=True)),
        ('int8scaler',MinMaxScaler(copy=False,feature_range = feature_range))
        ])

    X = xpipe.fit_transform(X_all)
    Yscaler = MinMaxScaler(copy=False,feature_range = feature_range).fit(Y_all.reshape(-1,1))
    Y = Yscaler.transform(Y_all.reshape(-1,1))

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=.85)

    theta = np.linalg.pinv(X_train).dot(Y_train)
    Y_0 = theta.T.dot(X_train.T).T
    Y_residual = Y_train - Y_0
    print('theta from polynomial fitting'%theta.T)
    print('Scaling to limits and then Y_0 inside\t%.3f\t%.3f'%(np.min(Y_0),np.max(Y_0)))

    print('####################### Fitting polynomial ensemble #################')
    stime = time.time()
    X_train_poly = polypipe.fit_transform(np.c_[X_train[:,:2].copy(),Y_0])
    polythetas = []
    for i in range(nmodels):
        X = X_train_poly[i*nsamples:(i+1)*nsamples,:]
        Y = Y_residual[i*nsamples:(i+1)*nsamples,0].copy()
        polythetas += [np.linalg.pinv( X ).dot( Y )]

    print('For %i poly models with %i samples it took %.3f seconds'%(nmodels,nsamples,time.time()-stime))

    X_blender = np.zeros((X_train_poly.shape[0],len(polythetas)))
    for i in range(len(polythetas)):
        X_blender[:,i] = polythetas[i].T.dot(X_train_poly.T)

    XblenderScaler = MinMaxScaler(copy=False,feature_range = feature_range).fit(X_blender)
    X = XblenderScaler.transform(X_blender)

    Y_res1 = np.zeros((X_train_poly.shape[0],1))
    for th in polythetas:
        Y_res1 += th.T.dot(X_train_poly.T).reshape(-1,1)
    Y_res1 /= float(len(polythetas))
    Y = Y_train - Y_0 - Y_res1
    
    print('######################## now, training blender ##########################')
    ntrees = 4 
    maxd = 64 
    blenderforest = RandomForestRegressor(n_estimators=ntrees, criterion='mse', max_depth=maxd)
    blenderforest.fit(X,Y.reshape(-1))
    fname_blender_rf = '%s/blender_rf_model_logT2logE_%s.sav'%(modelsfolder,time.strftime('%Y.%m.%d.%H.%M'))
    print('For %i trees with %i max_depth it took %.3f seconds'%(ntrees,maxd,time.time()-stime))



    print('####################### Simple Polynomial Inference (and ensemble) ###############')
    Y_0 = theta.T.dot(X_test.T).reshape(-1,1)
    Y = Yscaler.inverse_transform(Y_0.copy())
    ens_true = np.exp(Yscaler.inverse_transform(Y_test))
    ens_0 = np.exp( Y )
    print('Polynomial only (Y_0) test set rms error [eV]\t%.3f'%(np.sqrt(np.sum(np.power(ens_true - ens_0,int(2)))/len(ens_true))))

    stime = time.time()
    X_test_poly  = polypipe.transform(np.c_[X_test[:,:2].copy(),Y_0])
    Y_res_poly = np.zeros((X_test_poly.shape[0],1))
    for th in polythetas:
        Y_res_poly += th.T.dot(X_test_poly.T).reshape(-1,1)
    Y_res_poly /= float(len(polythetas))

    infertime = 1000000.*(time.time()-stime)/float(X_test.shape[0])
    Y_poly_ensemble = Yscaler.inverse_transform(Y_0 + Y_res_poly)
    ens_poly = np.exp(Y_poly_ensemble)
    print('inference time Polynomial\t%.3f [musec/Sample]'%(infertime))
    print('Polynomial ensemble test set rms error [eV]\t%.3f'%(np.sqrt(np.sum(np.power(ens_true - ens_poly,int(2)))/len(ens_true))))

    print('######################## now, doing blender ##########################')
    stime = time.time()
    X = np.zeros((X_test_poly.shape[0],len(polythetas)))
    for i in range(len(polythetas)):
        X[:,i] = polythetas[i].T.dot(X_test_poly.T)
    XblenderScaler.transform(X)
    Y_res2 = blenderforest.predict(X).reshape(-1,1)
    print(Y_res2.shape)
    Y_full_pred = Y_0 + Y_res_poly + Y_res2
    print(Y_full_pred.shape)
    Yscaler.inverse_transform(Y_full_pred)
    ens_blender = np.exp(Y_full_pred)

    infertime = 1000000.*(time.time()-stime)/float(X_test.shape[0])
    print('inference time blender\t%.3f [musec/Sample]'%(infertime))
    print('Blender ensemble test set rms error [eV]\t%.3f'%(np.sqrt(np.sum(np.power(ens_true - ens_blender,int(2)))/len(ens_true))))


    print('####################### Fitting RandomForest #################')
    Y_0 = theta.T.dot(X_train.T).T
    X_train_rf = np.c_[X_train[:,:includeingp],Y_0]
    stime = time.time()
    maxd = 16 
    ntrees = 32 
    forest = RandomForestRegressor(n_estimators=ntrees, criterion='mse', max_depth=maxd)
    forest.fit(X_train_rf,Y_residual.reshape(-1))
    fname_rf = '%s/rf_model_logT2logE_%s.sav'%(modelsfolder,time.strftime('%Y.%m.%d.%H.%M'))
    print('For %i trees with %i max_depth it took %.3f seconds'%(ntrees,maxd,time.time()-stime))


    print('####################### Fitting GP ensemble #################')
    X_train_gp = np.c_[X_train[:,:includeingp],Y_0]
    stime = time.time()
    maternnu = 1.5
    kern = 1.**2 * Matern(
            length_scale=10.*np.ones(X_train_gp.shape[1],dtype=float)
            ,length_scale_bounds=(1e-2,1e3)
            ,nu=maternnu
            )
    model_gp = GaussianProcessRegressor(kernel=kern, alpha=0, normalize_y=True, n_restarts_optimizer = 2)
    fname_list_gp = []
    if (nsamples * nmodels)>X_train.shape[0]:
        nsamples = X_train.shape[0]//nmodels
        print('reducing the number of samples for ensemble to %i'%(nsamples))
    lasttime = time.time()
    print('number of GP models is %i'%(nmodels_gp))
    for i in range(nmodels_gp):
        model_gp.fit(X_train_gp[i*nsamples:(i+1)*nsamples,:], 
                Y_residual[i*nsamples:(i+1)*nsamples,0].copy().reshape(-1,1)
                )
        print("Time for pertubative GP model fitting: %.3f" % (time.time() - lasttime))
        print("kernel = %s"%model_gp.kernel_)
        print("kernel Log-marginal-likelihood: %.3f" % model_gp.log_marginal_likelihood(model_gp.kernel_.theta))
        if model_gp.log_marginal_likelihood(model_gp.kernel_.theta) > 0:
            fname_list_gp += ['%s/gp_model_logT2logE_%s_%i.sav'%(modelsfolder,time.strftime('%Y.%m.%d.%H.%M'),i)]
            joblib.dump(model_gp,fname_list_gp[-1])
        lasttime = time.time()

    print('For %i models with %i samples it took %.3f seconds'%(nmodels_gp,nsamples,time.time()-stime))


    print('####################### Entering inference phase #################')
    Y_0_rf = theta.T.dot(X_test.T).reshape(-1,1)
    Y_0_gp = theta.T.dot(X_test.T).reshape(-1,1)
    stime = time.time()
    X_test_gp = np.c_[X_test[:,:includeingp],Y_0_gp]
    pred_sum = np.zeros((X_test_gp.shape[0],1))
    std_sum = np.zeros((X_test_gp.shape[0],1))
    for i in range(len(fname_list_gp)):
        model_gp = joblib.load(fname_list_gp[i])
        Y_res,Y_std = model_gp.predict(X_test_gp,return_std=True)
        for s in range(Y_res.shape[0]):
            if Y_std[s] > 0:
                pred_sum[s] += Y_res[s]/Y_std[s]
                std_sum[s] += 1./Y_std[s] 
    Y_res_gp = pred_sum / std_sum
    infertime = 1000000.*(time.time()-stime)/float(X_test.shape[0])
    print('inference time GP\t%.3f [musec/Sample]'%(infertime))
    print('GP test set rms error [in scaled]\t%.3f'%(np.sqrt(np.sum(np.power(Y_test - Y_0_gp - Y_res_gp,int(2)))/len(Y_test))))

    stime = time.time()
    X_test_rf = np.c_[X_test[:,:includeingp],Y_0_rf]
    Y_res_rf = forest.predict(X_test_rf).reshape(-1,1)
    infertime = 1000000.*(time.time()-stime)/float(X_test.shape[0])
    print('inference time RF\t%.3f [musec/Sample]'%(infertime))
    print('RF test set rms error [in scaled]\t%.3f'%(np.sqrt(np.sum(np.power(Y_test - Y_0_rf - Y_res_rf,int(2)))/len(Y_test))))


    Y_rf = Yscaler.inverse_transform(Y_0_rf.copy() + Y_res_rf)
    Y_gp = Yscaler.inverse_transform(Y_0_gp.copy() + Y_res_gp)
    ens_gp = np.exp( Y_gp )
    ens_rf = np.exp( Y_rf )
    print('GP test set rms error [eV]\t%.3f'%(np.sqrt(np.sum(np.power(ens_true - ens_gp,int(2)))/len(ens_true))))
    print('RF test set rms error [eV]\t%.3f'%(np.sqrt(np.sum(np.power(ens_true - ens_rf,int(2)))/len(ens_true))))


    return

if __name__ == '__main__':
    main()
