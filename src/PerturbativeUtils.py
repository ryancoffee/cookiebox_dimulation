import time

import numpy as np
import h5py
import sys
import random
import math
import joblib

from sklearn.multioutput import MultiOutputRegressor
from sklearn import linear_model
from sklearn import metrics
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, ConstantKernel, DotProduct, Matern
# Exploring Kernel Ridge Regression -- https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_compare_gpr_krr.html#sphx-glr-auto-examples-gaussian-process-plot-compare-gpr-krr-py
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

import DataUtils

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def featurizeX_taylor(X,n=4):
    result = [X.copy()]
    for p in range(n):
        result += [result[0] * result[-1]]
    return np.column_stack(result)

def featurizeX(X):
    first = X.copy()
    second = 1./2*np.power(first,int(2))
    third = 1./(2*3)*np.power(first,int(3))
    fourth = 1./(2*3*4)*np.power(first,int(4))
    return np.column_stack((first,second,third,fourth))

def fit_linear_tof(x,y,modelfolder):
    stime = time.time()
    model = linear_model.LinearRegression().fit(x[:,1].reshape(-1,1),y[:,0].reshape(-1,1))
    print("Time for initial linear model fitting on out-of-bag samples: %.3f" % (time.time() - stime))
    fname_linearmodel_tof = '%s/linear_model_tof_%s.sav'%(modelfolder,time.strftime('%Y.%m.%d.%H.%M'))
    joblib.dump(model,fname_linearmodel_tof)
    return fname_linearmodel_tof,model


def fit_taylor(x,y,featurefunc,ntaylor,modelfolder):
    stime = time.time()
    x_f = featurefunc(x,n=ntaylor) 
    taylormodel = linear_model.LinearRegression().fit(x_f,y.reshape(-1,1))
    print("Time for inverse taylor model fitting: %.3f" % (time.time() - stime))
    fname_taylormodel = '%s/inverse_taylor_model_%s.sav'%(modelfolder,time.strftime('%Y.%m.%d.%H.%M'))
    joblib.dump(taylormodel,fname_taylormodel)
    return fname_taylormodel,taylormodel

def fit_taylor_perturbative(x,y,featurefunc,model0,ntaylor,modelfolder):
    stime = time.time()
    x_f = featurefunc(x,n=ntaylor) 
    y_tof_res = y[:,0].copy().reshape(-1,1) - model0.predict(x[:,1].reshape(-1,1))
    y_pos = y[:,1].copy().reshape(-1,1)
    perturbmodel_tof = linear_model.LinearRegression().fit(x_f, y_tof_res)
    perturbmodel_pos = linear_model.LinearRegression().fit(x_f, y_pos)
    print("Time for pertubative linear model fitting: %.3f" % (time.time() - stime))
    fname_perturbmodel_tof = '%s/perturb_taylor_model_tof.order%i_%s.sav'%(modelfolder,ntaylor,time.strftime('%Y.%m.%d.%H.%M'))
    fname_perturbmodel_pos = '%s/perturb_taylor_model_pos.order%i_%s.sav'%(modelfolder,ntaylor,time.strftime('%Y.%m.%d.%H.%M'))
    joblib.dump(perturbmodel_tof,fname_perturbmodel_tof)
    joblib.dump(perturbmodel_pos,fname_perturbmodel_pos)
    return fname_perturbmodel_tof,perturbmodel_tof,fname_perturbmodel_pos,perturbmodel_pos

def ensemble_vote_t2e(x,gp_t2e_models,theta_model0,elitism = 0.2):
    print("ensembling t2e")
    nmodels = len(gp_t2e_models)
    y_pred = np.zeros(x.shape[0])
    y_std = np.zeros(x.shape[0])
    y_pred_0 = DataUtils.prependOnesToX(x.copy()).dot(theta_model0)
    y_pred_array = np.zeros((x.shape[0],nmodels),dtype=float)
    y_std_array = np.zeros((x.shape[0],nmodels),dtype=float)
    nhistbins = 100
    histbins = np.logspace(-6,-1,nhistbins)
    y_std_hist_array = np.zeros((nhistbins-1,nmodels+1),dtype=float)
    for i in range(nmodels):
        gp_t2e_model = gp_t2e_models[i]
        y_pred,y_std = gp_t2e_model.predict(x,return_std=True)
        y_pred_array[:,i] = y_pred.copy().reshape(-1)
        y_std_array[:,i] = y_std.copy().reshape(-1)
        h,b = np.histogram(y_std_array[:,i],histbins)
        if i==0:
            y_std_hist_array[:,i] = histbins[:-1]
        y_std_hist_array[:,i+1] = h

    y_pred = np.zeros(x.shape[0])
    y_std = np.zeros(x.shape[0])
    naverage = int(elitism*nmodels)
    srtinds = np.argsort(y_std_array,axis=1)
    y_pred_array = np.take_along_axis(y_pred_array, srtinds, axis=1)
    y_std_array = np.take_along_axis(y_std_array, srtinds, axis=1)
    y_pred = np.sum(y_pred_array[:,:naverage],axis=1)/float(naverage)
    y_std = np.sum(y_std_array[:,:naverage],axis=1)/float(naverage)
    h,b = np.histogram(y_std,histbins)
    y_std_hist_array = np.column_stack((y_std_hist_array,h))

    out = y_pred.copy() + y_pred_0.reshape(-1)

    return out,y_std.copy(),y_std_hist_array

def fit_gp_t2e_ensemble(x,y,maternnu,theta_model0,modelfolder,nmodels=8,nsamples=100):

    stime = time.time()
    kern = 1.**2 * Matern(
            length_scale=np.ones(x.shape[1],dtype=float)
            ,length_scale_bounds=(1e-5,100)
            ,nu=maternnu
            )

    model_gp = GaussianProcessRegressor(kernel=kern, alpha=0, normalize_y=True, n_restarts_optimizer = 2)

    fname_list_gp = []
    if (nsamples * nmodels)>x.shape[0]:
        nsamples = x.shape[0]//nmodels
        print('reducing the number of samples for ensemble to %i'%(nsamples))
    for i in range(nmodels):
        model_gp.fit(x[i*nsamples:(i+1)*nsamples,:], 
                y[i*nsamples:(i+1)*nsamples,0].copy().reshape(-1,1)
                - DataUtils.prependOnesToX(x[i*nsamples:(i+1)*nsamples,:].copy()).dot(theta_model0)
                )
        print("Time for pertubative GP model fitting: %.3f" % (time.time() - stime))
        print("kernel = %s"%model_gp.kernel_)
        print("kernel Log-marginal-likelihood: %.3f" % model_gp.log_marginal_likelihood(model_gp.kernel_.theta))
        fname_list_gp += ['%s/gp_model_tof_%s_%i.sav'%(modelfolder,time.strftime('%Y.%m.%d.%H.%M'),i)]
        joblib.dump(model_gp,fname_list_gp[i])
    print("Total time for pertubative GP model ensemble fitting: %.3f" % (time.time() - stime))
    return fname_list_gp

def fit_gp_perturbative_ensemble(x,y,maternnu_tof,maternnu_pos,model1_tof,model1_pos,featurefunc,ntaylor,model0,modelfolder,nmodels=8,nsamples=100):
    fname_lin_tof = '%s/lin_tof-%s'%(modelfolder,time.strftime('%Y.%m.%d.%H.%M'))
    fname_taylor_tof = '%s/taylor_tof-%s'%(modelfolder,time.strftime('%Y.%m.%d.%H.%M'))
    fname_taylor_pos = '%s/taylor_pos-%s'%(modelfolder,time.strftime('%Y.%m.%d.%H.%M'))

    joblib.dump(model0,fname_lin_tof)
    joblib.dump(model1_tof,fname_taylor_tof)
    joblib.dump(model1_pos,fname_taylor_pos)
    x_f = featurefunc(x,n=ntaylor)

    stime = time.time()
    k_tof = 1.**2 * Matern(
            length_scale=np.ones(x.shape[1],dtype=float)
            ,length_scale_bounds=(1e-5,100)
            ,nu=maternnu_tof
            )
    k_pos = 1.**2 * Matern(
            length_scale=np.ones(x.shape[1],dtype=float)
            ,length_scale_bounds=(1e-5,100)
            ,nu=maternnu_pos
            )
    kernel_gp_tof = k_tof 
    kernel_gp_pos = k_pos

    tof_gp = GaussianProcessRegressor(kernel=kernel_gp_tof, alpha=0, normalize_y=True, n_restarts_optimizer = 2)
    pos_gp = GaussianProcessRegressor(kernel=kernel_gp_pos, alpha=0, normalize_y=True, n_restarts_optimizer = 2)
    fname_list_gp_tof = []
    fname_list_gp_pos = []
    if (nsamples * nmodels)>x.shape[0]:
        nsamples = x.shape[0]//nmodels
    for i in range(nmodels):
        print('\t=========\tfitting %i of %i models\t========'%(i,nmodels))
        tof_gp.fit(x[i*nsamples:(i+1)*nsamples,:], 
                y[i*nsamples:(i+1)*nsamples,0].copy().reshape(-1,1)
                - model1_tof.predict(x_f[i*nsamples:(i+1)*nsamples,:])
                - model0.predict(x[i*nsamples:(i+1)*nsamples,1].copy().reshape(-1,1))
                )
        pos_gp.fit(x[i*nsamples:(i+1)*nsamples,:], 
                y[i*nsamples:(i+1)*nsamples,1].reshape(-1,1)
                - model1_pos.predict(x_f[i*nsamples:(i+1)*nsamples,:])
                )
        print("Time for pertubative GP model fitting: %.3f" % (time.time() - stime))
        print("tof kernel = %s"%tof_gp.kernel_)
        print("tof kernel Log-marginal-likelihood: %.3f" % tof_gp.log_marginal_likelihood(tof_gp.kernel_.theta))
        print("pos kernel = %s"%pos_gp.kernel_)
        print("pos kernel Log-marginal-likelihood: %.3f" % pos_gp.log_marginal_likelihood(pos_gp.kernel_.theta))
        fname_list_gp_tof += ['%s/gp_model_tof_%s_%i.sav'%(modelfolder,time.strftime('%Y.%m.%d.%H.%M'),i)]
        fname_list_gp_pos += ['%s/gp_model_pos_%s_%i.sav'%(modelfolder,time.strftime('%Y.%m.%d.%H.%M'),i)]
        joblib.dump(tof_gp,fname_list_gp_tof[i])
        joblib.dump(pos_gp,fname_list_gp_pos[i])
    print("Total time for pertubative GP model ensemble fitting: %.3f" % (time.time() - stime))
    return (fname_list_gp_tof,fname_list_gp_pos)

def validate_lin_tof(x,y,model):
    stime = time.time()
    y_pred = model.predict(x[:,1].reshape(-1,1))
    print("Average time for linear tof inference : %.3f usec" % ((time.time() - stime)*1e6/float(x.shape[0])))
    print ("linear model rmse: ", metrics.mean_squared_error(y[:,0], y_pred,squared=False))
    return

def validate_perturb_pos(x,y,model,featurefunc,ntaylor):
    stime = time.time()
    x_f = featurefunc(x,ntaylor)
    y_pred = model.predict(x_f)
    print("Average time for perturbative linear pos inference : %.3f usec" % ((time.time() - stime)*1e6/float(x.shape[0])))
    print ("Perturbative linear model pos rmse: ", metrics.mean_squared_error(y[:,1], y_pred,squared=False))
    return

def validate_perturb_tof(x,y,model,featurefunc,ntaylor,model0):
    stime = time.time()
    x_f = featurefunc(x,ntaylor)
    y_pred = model.predict(x_f) + model0.predict(x[:,1].reshape(-1,1))
    print("Average time for perturbative linear tof inference : %.3f usec" % ((time.time() - stime)*1e6/float(x.shape[0])))
    print ("Perturbative linear model tof rmse: ", metrics.mean_squared_error(y[:,0], y_pred,squared=False))
    return
    
def validate_gp_tof(x,y,gp_tof_model,model1_tof,featurefunc,ntaylor,model0):
    stime = time.time()
    x_f = featurefunc(x,ntaylor)
    y_pred = gp_tof_model.predict(x) + model1_tof.predict(x_f) + model0.predict(x[:,1].reshape(-1,1))
    latency = (time.time() - stime)*1e6/float(x.shape[0])
    score = metrics.mean_squared_error(y[:,0],y_pred,squared=False)
    print("Average time for perturbative GP tof inference : %.3f usec\trmse : %.4f" % (latency,score))
    return y_pred,latency,score

def validate_gp_pos(x,y,gp_pos_model,model1_pos,featurefunc,ntaylor):
    stime = time.time()
    x_f = featurefunc(x,ntaylor)
    y_pred = gp_pos_model.predict(x) + model1_pos.predict(x_f)
    latency = (time.time() - stime)*1e6/float(x.shape[0])
    score = metrics.mean_squared_error(y[:,1],y_pred,squared=False)
    print("Average time for perturbative GP pos inference : %.3f usec\trmse : %.4f" % (latency,score))
    return y_pred,latency,score

def inference_gp_tof(x,gp_tof_model,model1_tof,featurefunc,ntaylor,model0):
    x_f = featurefunc(x,ntaylor)
    y_pred,y_std = gp_tof_model.predict(x)
    return y_pred + model1_tof.predict(x_f) + model0.predict(x[:,1].reshape(-1,1)), y_std

def inference_gp_pos(x,gp_pos_model,model1_pos,featurefunc,ntaylor):
    x_f = featurefunc(x,ntaylor)
    y_pred,y_std = gp_pos_model.predict(x)
    return y_pred + model1_pos.predict(x_f) , y_std

def inference_taylor(x,taylor_model,featurefunc,ntaylor=2):
    return taylor_model.predict(featurefunc(x,ntaylor))

# in future, allow elitism to accept an array of weights, this couls allow for different voting blocks to have diminishing weights
# furthermore, we could set a minimum limit on "agreement" like voting theory in order to trigger a "bad sample" or "anomaly" event
# baratza:~/papers/voting/*
def ensemble_vote_tof(x,gp_tof_models,model1_tof,featurefunc,ntaylor,model0,elitism = 0.2):
    print("ensembling tof")
    nmodels = len(gp_tof_models)
    x_f = featurefunc(x,ntaylor)
    y_pred = np.zeros(x.shape[0])
    y_antipred = np.zeros(x.shape[0])
    y_std = np.zeros(x.shape[0])
    y_antistd = np.zeros(x.shape[0])
    y_pred_model1 = model1_tof.predict(x_f)
    y_pred_model0 = model0.predict(x[:,1].reshape(-1, 1))
    y_pred_array = np.zeros((x.shape[0],nmodels),dtype=float)
    y_antipred_array = np.zeros((x.shape[0],nmodels),dtype=float)
    y_std_array = np.zeros((x.shape[0],nmodels),dtype=float)
    nhistbins = 100
    histbins = np.logspace(-6,-1,nhistbins)
    y_std_hist_array = np.zeros((nhistbins-1,nmodels+1),dtype=float)
    for i in range(nmodels):
        gp_tof_model = gp_tof_models[i]
        y_pred,y_std = gp_tof_model.predict(x,return_std=True)
        y_pred_array[:,i] = y_pred.copy().reshape(-1)
        y_std_array[:,i] = y_std.copy().reshape(-1)
        h,b = np.histogram(y_std_array[:,i],histbins)
        if i==0:
            y_std_hist_array[:,i] = histbins[:-1]
        y_std_hist_array[:,i+1] = h

    y_pred = np.zeros(x.shape[0])
    y_std = np.zeros(x.shape[0])
    naverage = int(elitism*nmodels)
    srtinds = np.argsort(y_std_array,axis=1)
    y_pred_array = np.take_along_axis(y_pred_array, srtinds, axis=1)
    y_std_array = np.take_along_axis(y_std_array, srtinds, axis=1)
    y_pred = np.sum(y_pred_array[:,:naverage],axis=1)/float(naverage)
    y_std = np.sum(y_std_array[:,:naverage],axis=1)/float(naverage)
    h,b = np.histogram(y_std,histbins)
    y_std_hist_array = np.column_stack((y_std_hist_array,h))

    out = y_pred.copy() + y_pred_model1.reshape(-1) + y_pred_model0.reshape(-1)

    y_antistd = np.sum(y_std_array[:,-naverage-1:],axis=1)/float(naverage)
    h,b = np.histogram(y_antistd,histbins)
    y_std_hist_array = np.column_stack((y_std_hist_array,h))
    return out,y_std.copy(),y_std_hist_array

def ensemble_vote_pos(x,gp_pos_models,model1_pos,featurefunc,ntaylor,elitism = 0.2):
    print("ensembling pos")
    nmodels = len(gp_pos_models)
    x_f = featurefunc(x,ntaylor)
    y_pred = np.zeros(x.shape[0])
    y_antipred = np.zeros(x.shape[0])
    y_std = np.zeros(x.shape[0])
    y_antistd = np.zeros(x.shape[0])
    y_pred_model1 = model1_pos.predict(x_f)
    y_pred_array = np.zeros((x.shape[0],nmodels),dtype=float)
    y_std_array = np.zeros((x.shape[0],nmodels),dtype=float)
    nhistbins = 100
    histbins = np.logspace(-5,1,nhistbins)
    y_std_hist_array = np.zeros((nhistbins-1,nmodels+1),dtype=float)
    for i in range(nmodels):
        gp_pos_model = gp_pos_models[i]
        y_pred,y_std = gp_pos_model.predict(x,return_std=True)
        y_pred_array[:,i] = y_pred.copy().reshape(-1)
        y_std_array[:,i] = y_std.copy().reshape(-1)
        h,b = np.histogram(y_std_array[:,i],histbins)
        if i==0:
            y_std_hist_array[:,i] = histbins[:-1]
        y_std_hist_array[:,i+1] = h

    y_pred = np.zeros(x.shape[0])
    y_std = np.zeros(x.shape[0])
    naverage = int(elitism*nmodels)
    srtinds = np.argsort(y_std_array,axis=1)
    y_pred_array = np.take_along_axis(y_pred_array, srtinds, axis=1)
    y_std_array = np.take_along_axis(y_std_array, srtinds, axis=1)
    y_pred = np.sum(y_pred_array[:,:naverage],axis=1)/float(naverage)
    y_std = np.sum(y_std_array[:,:naverage],axis=1)/float(naverage)
    h,b = np.histogram(y_std,histbins)
    y_std_hist_array = np.column_stack((y_std_hist_array,h))

    out = y_pred.copy() + y_pred_model1.reshape(-1)

    y_antistd = np.sum(y_std_array[:,-naverage-1:],axis=1)/float(naverage)
    h,b = np.histogram(y_antistd,histbins)
    y_std_hist_array = np.column_stack((y_std_hist_array,h))
    return out,y_std.copy(),y_std_hist_array



'''
This is being ignored
def fit_krr_perturbative(x,y,linmodel_tof,taylormodel_tof,taylormodel_pos,taylororder,modelfolder=modelsfolder,nmodels=nmodels,nsamples=nsamples):
    # Fit KernelRidge with parameter selection based on 5-fold cross validation
    param_grid = {"alpha": [1e0, 1e-1, 1e-2, 1e-3],
                  "kernel": [RationalQuadratic(l, a)
                  for l in np.logspace(-2, 2, 10)
                  for a in np.logspace(0, 2, 10)]}
    kr = GridSearchCV(KernelRidge(), param_grid=param_grid)
    stime = time.time()
    kr.fit(x[2000:4000,:], Y_train[2000:4000,:])
    print("Time for KRR fitting: %.3f" % (time.time() - stime))
    #print("KRR kernel: %s" % kr.kernel_)
    return fname_KRRmodel_tof,KRRmodel_tof,fname_KRRmodel_pos,KRRmodel_pos
'''

'''
def validate_krr_pos(X_test,Y_test,model=perturb_pos,featurefunc=featurizeX_taylor,n=ntaylor):
    stime = time.time()
    Y_test_pred = kr.predict(X_test)
    Y_valid_pred = kr.predict(X_valid)
    print("Time for KRR inference: %.3f" % (time.time() - stime))
    print ("KRR model rmse (test): ", metrics.mean_squared_error(Y_test, Y_test_pred,squared=False))
    print ("KRR model rmse (validate): ", metrics.mean_squared_error(Y_valid, Y_valid_pred,squared=False))

'''

