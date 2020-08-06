#!/usr/bin/python3

import time
import numpy as np
import sys
import os
import joblib
import re

from sklearn import metrics # remaining printout of GP metrics from main

import PerturbativeUtils
import DataUtils
import MathUtils

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def main():
    do_correlation = True
    ensemble_all_train = True

    nmodels = 32 # It's looking like 24 or 32 models and 300 samples is good with an elitism of .125 this means we are averaging 4 model results
    # but, the number of models doesn't hurt the latency in FPGA, so nsamples 300 and data set large enough for at least 24 models
    nsamples = 500 # eventually 500
    printascii = False
    taylor_order = 4
    maternnu_tof = 1.5
    maternnu_pos = 1.5
    binslatency = np.logspace(-1,2.7,200)
    elitism = 0.5


    #./data_ave/ind_25-plate_tune_grid_Range_*/analyzed_data.hdf5
    m = re.match('(.*)/(ind.*)/analyzed_data.hdf5',sys.argv[-1])
    if not m:
        print('syntax: ./src/load_full_perturbatrive.py <./data_ave/ind_25-plate_tune_grid_Range_*/analyzed_data.hdf5> ')
        return


    X_all = []
    Y_all = []
    X_all,Y_all = DataUtils.loaddata()

    if len(Y_all)<1:
        print("no data loaded")
        return
    print("data loaded\tcontinuing to fitting")
    



    #============================== this is important, this needs to become the "transformer" =============================#
    #============================== this is important, this needs to become the "transformer" =============================#


    '''
    OK, keep your thoughts straight...
        we are first linear fitting through, theata[0] + theta[1]*vset + theta[2]*en = tof
        Doing this with matrix inversion
    '''

    ntaylor = 8
    X_train,X_test,Y_train,Y_test = DataUtils.reservesplit(X_all,Y_all,reserve = .1)
    theta0 = DataUtils.pseudoinversemethod(X_train[:,:2],Y_train[:,0])
    print(theta0)
    Y_pred0 = DataUtils.prependOnesToX(X_train[:,:2].copy()).dot(theta0)
    X_prime_f = DataUtils.appendTaylorToX(MathUtils.Rot45(X_train[:,:2]),n=ntaylor)
    theta1 = DataUtils.pseudoinversemethod(X_prime_f,Y_train[:,0]-Y_pred0)
    Y_pred1 = DataUtils.prependOnesToX(X_prime_f.copy()).dot(theta1)

    Y_test_result = DataUtils.prependOnesToX( DataUtils.appendTaylorToX( MathUtils.Rot45(X_test[:,:2]) , n=ntaylor) ).dot(theta1) + DataUtils.prependOnesToX(X_test[:,:2].copy()).dot(theta0)
    print('rmse (test) tof in ns: ',  metrics.mean_squared_error(np.exp(Y_test[:,0]),np.exp(Y_test_result),squared=False))
    print(theta1)

    #============================== this is important, this needs to become the "transformer" =============================#
    #============================== this is important, this needs to become the "transformer" =============================#






    Y_tof = Y_train[:,0].copy()-Y_pred0-Y_pred1
    Y_train_residual = np.column_stack((Y_train[:,0].copy()-Y_pred0-Y_pred1,Y_train[:,1]))

    X_train_residual = np.column_stack((X_prime_f[:,:2],X_train[:,2]))
    

    if ensemble_all_train:
        nmodels = X_train.shape[0]//nsamples
    print('\t\t========= Using %i models in GP e2tof ensemble ===========\n'%(nmodels))

    modelsfolder = '%s/newensemble%imodels%isamples%.2felitism'%(m.group(1),nmodels,nsamples,elitism)
    if not os.path.exists(modelsfolder):
        os.makedirs(modelsfolder)
        modelsfolder = '%s/newensemble%imodels%isamples%.2felitism'%(m.group(1),nmodels,nsamples,elitism)

    X,Y,Xscaler,Yscaler = DataUtils.minmaxscaledata(X_train_residual,Y_train_residual,feature_range = (-1,1))

    fname_elitism = '%s/elitism_%s.sav'%(modelsfolder,time.strftime('%Y.%m.%d.%H.%M'))
    fname_Xscaler = '%s/Xscaler_%s.sav'%(modelsfolder,time.strftime('%Y.%m.%d.%H.%M'))
    fname_Yscaler = '%s/Yscaler_%s.sav'%(modelsfolder,time.strftime('%Y.%m.%d.%H.%M'))
    fname_theta0 = '%s/theta0_%s.sav'%(modelsfolder,time.strftime('%Y.%m.%d.%H.%M'))
    fname_theta1 = '%s/theta1_%s.sav'%(modelsfolder,time.strftime('%Y.%m.%d.%H.%M'))
    f = open(fname_elitism,'wb')
    joblib.dump(elitism,f)
    f.close()
    f = open(fname_theta0,'wb')
    joblib.dump(theta0,f)
    f.close()
    f = open(fname_theta1,'wb')
    joblib.dump(theta1,f)
    f.close()
    f = open(fname_Xscaler,'wb')
    joblib.dump(Xscaler,f)
    f.close()
    f = open(fname_Yscaler,'wb')
    joblib.dump(Yscaler,f)
    f.close()



    print("\n\n\t\t================\tMoving on to perturbative GP\t=================\n")


    fnames_gp_tof,fnames_gp_pos = PerturbativeUtils.fit_gp_new_ensemble(
            X,
            Y,
            maternnu_tof = maternnu_tof,
            maternnu_pos = maternnu_pos,
            modelfolder=modelsfolder,
            nmodels=nmodels,
            nsamples=nsamples)

    X_test_residual = np.column_stack((MathUtils.Rot45(X_test[:,:2]),X_test[:,2]))
    Xscaler.transform(X_test_residual)

    
    gp_tof_models = []
    gp_pos_models = []
    for fname in fnames_gp_tof:
        f = open(fname,'rb')
        gp_tof_models += [joblib.load(f)]
        f.close()
    for fname in fnames_gp_pos:
        f = open(fname,'rb')
        gp_pos_models += [joblib.load(f)]
        f.close()

    Y_tof_residual = PerturbativeUtils.ensemble_vote_new(X_test_residual,gp_tof_models,elitism = 0.5)
    Y_pos_residual = PerturbativeUtils.ensemble_vote_new(X_test_residual,gp_pos_models,elitism = 0.5)

    Y_out_residual = np.column_stack((Y_tof_residual,Y_pos_residual))
    Yscaler.inverse_transform(Y_out_residual)
    Y_out_residual[:,0] += Y_test_result

    print('rmse (test, gp vote) tof in ns: ',  metrics.mean_squared_error(np.exp(Y_test[:,0]),np.exp(Y_out_residual[:,0]),squared=False))
    print('rmse (test, gp vote) pos in mm: ',  metrics.mean_squared_error(Y_test[:,1],Y_out_residual[:,1],squared=False))
    fname_testresult = '%s/test_results_%s.dat'%(modelsfolder,time.strftime('%Y.%m.%d.%H.%M'))
    np.savetxt(fname_testresult,np.column_stack((X_test,Y_test_result,Y_out_residual)))

    return


if __name__ == '__main__':
    main()

