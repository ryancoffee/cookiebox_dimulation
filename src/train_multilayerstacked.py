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
    print('Training the multilayer stacked ensemble')
    do_correlation = True
    usealltrain = True
    nmodels = 32 # It's looking like 24 or 32 models and 300 samples is good with an elitism of .125 this means we are averaging 4 model results
    # but, the number of models doesn't hurt the latency in FPGA, so nsamples 300 and data set large enough for at least 24 models
    nsamples = 200 # eventually 500
    printascii = True
    taylor_order = 4
    maternnu_tof = 1.5
    maternnu_pos = 1.5
    binslatency = np.logspace(-1,2.7,200)
    elitism = 0.25


    #./data_ave/ind_25-plate_tune_grid_Range_*/analyzed_data.hdf5
    m = re.match('(.*)/(ind.*(logos_.*)/)analyzed_data.hdf5',sys.argv[-1])
    if not m:
        print('syntax: ./src/train_multilayerstacked.py <./data_ave/ind_25-plate_tune_grid_Range_*/analyzed_data.hdf5> ')
        return


    print('loading data')
    X_all = []
    Y_all = []
    X_all,Y_all = DataUtils.loaddata()
    print(Y_all)

    if len(Y_all)<1:
        print("no data loaded")
        return
    print("continuing to fitting")

    X_bags,X_test,Y_bags,Y_test = DataUtils.evensplitbags(X_all,Y_all,nsplits=3)
    print(Y_bags[0])


    modelsfolder = '%s/multistacked%imodels%isamples'%(m.group(1),nmodels,nsamples)
    if not os.path.exists(modelsfolder):
        os.makedirs(modelsfolder)
    timestamp = time.strftime('%Y.%m.%d.%H.%M')
    if printascii:
        for i in range(len(X_bags)):
            np.savetxt('%s/bag%i.dat'%(m.group(1),i),np.column_stack((X_bags[i],Y_bags[i])))

    if do_correlation:
        print('doing correlation')
        outname = '%s/X_Y_featurecorr.dat'%(m.group(1))
        if not DataUtils.crosscorrelation(outname,X_all,Y_all):
            print('Failed crosscorrelation somehow')

    y = Y_bags[0][:,0].copy()
    x = X_bags[0].copy()
    #x,y,Xscaler,Yscaler = DataUtils.minmaxscaledata(x,y,feature_range = (-1,1))

    theta0  = np.linalg.pinv(DataUtils.prependOnes(x)).dot(y) 
    print('theta0 = %s'%(theta0))
    Y_pred0 = DataUtils.prependOnes(X_test.copy()).dot(theta0)
    print(np.column_stack((Y_pred0,Y_test[:,0])))
    print ("linear model rmse: \n%s [ns]"%(metrics.mean_squared_error(np.exp(Y_test[:,0]).reshape(-1), np.exp(Y_pred0.reshape(-1)),squared=False) ) )
    if printascii:
        np.savetxt('%s/pred0.dat'%(m.group(1)),np.column_stack((X_test,Y_test,Y_pred0)))


    print('using bag #1')

    nsamples = 300
    nmodels = 64
    if usealltrain:
        nmodels = X_bags[1].shape[0]//nsamples
    nmodels -= 2 
    print('holding out 2 sets of nsamples for validation')
    print('Not the residual idea, adding Ypred0 as feature...\n\tensembling %i models with %i samples'%(nmodels,nsamples))
    Y0 = DataUtils.prependOnes(X_bags[1].copy()).dot(theta0)
    X = np.column_stack((X_bags[1][:-2*nsamples,:],Y0[:-2*nsamples])) # this should handle the removing of the residual from the linear order of the feature... like boosting I think
    X,Polyscaler = DataUtils.polyfeaturize(X,order=6) # this has a sharp cutoff at 6, 7 has no improvement, 5 is nearly up to 1ns 
    X,Y,Xscaler,Yscaler = DataUtils.minmaxscaledata(X,Y_bags[1][:-2*nsamples,0].copy()-Y0[:-2*nsamples],feature_range = (-1,1))
    thetas = [np.linalg.pinv(X[i*nsamples:(i+1)*nsamples,:]).dot(Y[i*nsamples:(i+1)*nsamples]) for i in range(nmodels)]
    #print('Ensemble thetas:')
    #[print('%s'%(th)) for th in thetas]
    X = np.column_stack((X_bags[1][-2*nsamples:,:].copy(),Y0[-2*nsamples:]))
    X = Xscaler.transform( Polyscaler.transform(X) )
    Y_preds = [Yscaler.inverse_transform( X.dot(th).reshape(-1,1) ) for th in thetas]

    tof_true = np.exp(Y_bags[1][-2*nsamples:,0])
    tof_y0 = np.exp(Y0[-2*nsamples:])
    errors = [metrics.mean_squared_error(tof_true, [np.exp(Y_pred[i])*tof_y0[i] for i in range(len(Y_pred))],squared=False) for Y_pred in Y_preds]
    #errors = [metrics.mean_squared_error(np.exp(Y_bags[1][-2*nsamples:,0]), np.exp(Y_pred + Y0[-2*nsamples:]),squared=False) for Y_pred in Y_preds]
    good_thetas = [ thetas[ i ] for i in np.argsort(errors).copy()[:int(nmodels*elitism)] ]
    b = np.linspace(0,1,41)
    h,b = np.histogram(errors,b)
    print ("histogram of polynomial ensemble model rmse [ns]:")
    [print('%.2f\t%s'%(b[i],'.'*h[i])) for i in range(len(h))]
    #([metrics.mean_squared_error(np.exp(Y_test[:,0]), np.exp(Y_pred),squared=False) for Y_pred in Y_preds]) )
    Y_good_preds = [Yscaler.inverse_transform( X.dot(th).reshape(-1,1) ) for th in good_thetas]
    Y_preds_mean = np.sum(np.column_stack((Y_good_preds)),axis = 1 )/len(Y_good_preds)
    tof_residual2 = np.exp(Y_preds_mean)
    print ("polynomial model rmse (mean) [ns]: \n%s"%( metrics.mean_squared_error(tof_true, tof_y0 * tof_residual2 ,squared=False) ) )

    print('Keep in mind that the residual depends on the 45 deg rotation of the X[:,0] and X[:,1] features')
    print('It may be getting time to build a transformer pipeline')


    print('using bag #2 for blender')
    
    Y0 = DataUtils.prependOnes(X_bags[2].copy()).dot(theta0)
    X = np.column_stack((X_bags[2],Y0)) # this should handle the removing of the residual from the linear order of the feature... like boosting I think
    X = Xscaler.transform( Polyscaler.transform(X) )
    Y_preds = [Yscaler.inverse_transform( X.dot(th).reshape(-1,1) ) for th in good_thetas]
    Y_preds_mean = np.sum(np.column_stack((Y_preds)),axis = 1 )/len(Y_preds)
    X2 = np.column_stack((X_bags[2].copy(),np.column_stack(Y_preds)))
    X2,Polyscaler2 = DataUtils.polyfeaturize(X2,order = 2)
    #Xrot = np.column_stack((MathUtils.Rot45(X_bags[2][:,:2]) , X_bags[2][:,2], np.column_stack(Y_preds)))
    #Xrot,Yrot,XrotScaler,YrotScaler = DataUtils.minmaxscaledata(Xrot,Y_bags[2][:,0].copy(),feature_range = (-1,1))
    X2,Y2,X2Scaler,Y2Scaler = DataUtils.minmaxscaledata(X2,Y_bags[2][:,0].copy()-Y_preds_mean-Y0,feature_range = (-1,1))


    nmodels =64 
    print('fitting random forest\t... next to try maybe GP instead of RF')
    rf_model = PerturbativeUtils.fit_forest(X2,Y2,nmodels = nmodels)
    print('done fitting')

    print('Now running on test set')
    Y0 = DataUtils.prependOnes(X_test.copy()).dot(theta0)
    X = np.column_stack((X_test.copy(),Y0)) # this should handle the removing of the residual from the linear order of the feature... like boosting I think
    X = Xscaler.transform( Polyscaler.transform(X) )
    Y_preds = [Yscaler.inverse_transform( X.dot(th).reshape(-1,1) ) for th in good_thetas]
    Y_preds_mean = np.sum(np.column_stack((Y_preds)),axis = 1 )/len(Y_preds)
    #Xrot = np.column_stack((MathUtils.Rot45(X_test[:,:2].copy()) , X_test[:,2].copy(), np.column_stack(Y_preds)))
    #Xrot = XrotScaler.transform(Xrot)
    X2 = np.column_stack((X_test.copy(),np.column_stack(Y_preds)))
    X2 = X2Scaler.transform( Polyscaler2.transform(X2) )
    Y_pred_test = Y2Scaler.inverse_transform(PerturbativeUtils.vote_forest(X2,rf_model).reshape(-1,1))
    tof_true = np.exp(Y_test[:,0])
    tof_residual3 = np.exp( Y_pred_test )
    tof_residual2 = np.exp(Y_preds_mean)
    tof_y0 = np.exp(Y0)


    print ("random forest blender model rmse (mean) [ns]: \n%s"%( metrics.mean_squared_error(tof_true, [tof_y0[i] * tof_residual2[i] * tof_residual3[i] for i in range(len(tof_true))] ,squared=False) ) )

    print('Now it is pretty darn close to the GP errors')

    return


if __name__ == '__main__':
    main()

