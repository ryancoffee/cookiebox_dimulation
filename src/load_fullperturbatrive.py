#!/usr/bin/python3

import time

import numpy as np
import h5py
import sys
import random
import math
from sklearn.multioutput import MultiOutputRegressor
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics
from sklearn.feature_selection import mutual_info_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, ConstantKernel, DotProduct, Matern
#from sklearn.externals import joblib
import joblib
import re

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
        for vsetting in list(f.keys())[3:-2]: # restricting to only the closest couple vsettings to optimal... correction, now only the central (optimal) one, but for each 'logos'
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
            validinds = np.where((xsplat>182.5) * (xsplat<187) * (emat>0) * (abs(ydata)<.050))
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
    return x_all,y_all

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

def fit_taylor_perturbative(x,y,featurefunc,model0,ntaylor,modelfolder):
    stime = time.time()
    x_f = featurefunc(x,n=ntaylor) 
    y_tof_res = y[:,0].copy().reshape(-1,1) - model0.predict(x[:,1].reshape(-1,1))
    y_pos = y[:,1].copy().reshape(-1,1)
    perturbmodel_tof = linear_model.LinearRegression().fit(x_f, y_tof_res)
    perturbmodel_pos = linear_model.LinearRegression().fit(x_f, y_pos)
    print("Time for pertubative linear model fitting: %.3f" % (time.time() - stime))
    fname_perturbmodel_tof = '%s/perturb_taylor_model_tof_%s.sav'%(modelfolder,time.strftime('%Y.%m.%d.%H.%M'))
    fname_perturbmodel_pos = '%s/perturb_taylor_model_pos_%s.sav'%(modelfolder,time.strftime('%Y.%m.%d.%H.%M'))
    joblib.dump(perturbmodel_tof,fname_perturbmodel_tof)
    joblib.dump(perturbmodel_pos,fname_perturbmodel_pos)
    return fname_perturbmodel_tof,perturbmodel_tof,fname_perturbmodel_pos,perturbmodel_pos

def fit_gp_perturbative_ensemble(x,y,model1_tof,model1_pos,featurefunc,ntaylor,model0,modelfolder,nmodels=8,nsamples=100):
    fname_lin_tof = '%s/lin_tof-%s'%(modelfolder,time.strftime('%Y.%m.%d.%H.%M'))
    fname_taylor_tof = '%s/taylor_tof-%s'%(modelfolder,time.strftime('%Y.%m.%d.%H.%M'))
    fname_taylor_pos = '%s/taylor_pos-%s'%(modelfolder,time.strftime('%Y.%m.%d.%H.%M'))

    joblib.dump(model0,fname_lin_tof)
    joblib.dump(model1_tof,fname_taylor_tof)
    joblib.dump(model1_pos,fname_taylor_pos)
    x_f = featurefunc(x,n=ntaylor)

    stime = time.time()
    k1 = 1.0**2 * RBF(
            length_scale=np.ones(x.shape[1],dtype=float)
            ,length_scale_bounds=(1e-5,100)
            ) 
    k2 = 1.0**2 * RationalQuadratic(
            length_scale=1. 
            ,alpha=0.1 
            ,length_scale_bounds=(1e-5,20)
            ) 

    k3 = .5*2 * WhiteKernel(noise_level=0.01**2)  # noise terms
    k4 = ConstantKernel(constant_value = .01 ) # constant shift

    k1p = 0.05**2 * RBF(
            length_scale=np.ones(x.shape[1],dtype=float)
            ,length_scale_bounds=(1e-5,20)
            ) 
    k2p = 1.**2 * Matern(
            length_scale=np.ones(x.shape[1],dtype=float)
            ,length_scale_bounds=(1e-5,20)
            ,nu=1.5
            )
    k3p = .5*2 * WhiteKernel(noise_level=0.01**2)  # noise terms
    k4p = ConstantKernel(constant_value = .01 ) # constant shift

    kernel_gp_tof = k1 + k2 #+ k3 + k4
    kernel_gp_pos = k1p + k2p + k3p + k4p

    tof_gp = GaussianProcessRegressor(kernel=kernel_gp_tof, alpha=0, normalize_y=True, n_restarts_optimizer = 2)
    pos_gp = GaussianProcessRegressor(kernel=kernel_gp_pos, alpha=0, normalize_y=True, n_restarts_optimizer = 2)
    fname_list_gp_tof = []
    fname_list_gp_pos = []
    for i in range(nmodels):
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
    print ("linear model score: ", metrics.r2_score(y[:,0], y_pred))
    return

def validate_perturb_pos(x,y,model,featurefunc,ntaylor):
    stime = time.time()
    x_f = featurefunc(x,ntaylor)
    y_pred = model.predict(x_f)
    print("Average time for perturbative linear pos inference : %.3f usec" % ((time.time() - stime)*1e6/float(x.shape[0])))
    print ("Perturbative linear model pos score: ", metrics.r2_score(y[:,1], y_pred))
    return

def validate_perturb_tof(x,y,model,featurefunc,ntaylor,model0):
    stime = time.time()
    x_f = featurefunc(x,ntaylor)
    y_pred = model.predict(x_f) + model0.predict(x[:,1].reshape(-1,1))
    print("Average time for perturbative linear tof inference : %.3f usec" % ((time.time() - stime)*1e6/float(x.shape[0])))
    print ("Perturbative linear model tof score: ", metrics.r2_score(y[:,0], y_pred))
    return
    
def validate_gp_tof(x,y,gp_tof_model,model1_tof,featurefunc,ntaylor,model0):
    stime = time.time()
    x_f = featurefunc(x,ntaylor)
    y_pred = gp_tof_model.predict(x) + model1_tof.predict(x_f) + model0.predict(x[:,1].reshape(-1,1))
    print("Average time for perturbative GP tof inference : %.3f usec" % ((time.time() - stime)*1e6/float(x.shape[0])))
    print('GP score tof: ',  metrics.r2_score(y[:,0],y_pred))
    return y_pred

def validate_gp_pos(x,y,gp_pos_model,model1_pos,featurefunc,ntaylor):
    stime = time.time()
    x_f = featurefunc(x,ntaylor)
    y_pred = gp_pos_model.predict(x) + model1_pos.predict(x_f)
    print("Average time for perturbative GP pos inference : %.3f usec" % ((time.time() - stime)*1e6/float(x.shape[0])))
    print('GP score pos: ',  metrics.r2_score(y[:,1],y_pred))
    return y_pred

def inference_gp_tof(x,y,gp_tof_model,model1_tof,featurefunc,ntaylor,model0):
    x_f = featurefunc(x,ntaylor)
    y_pred,y_std = gp_tof_model.predict(x)
    return y_pred + model1_tof.predict(x_f) + model0.predict(x[:,1].reshape(-1,1)), y_std

def inference_gp_pos(x,y,gp_pos_model,model1_pos,featurefunc,ntaylor):
    x_f = featurefunc(x,ntaylor)
    y_pred,y_std = gp_pos_model.predict(x)
    return y_pred + model1_pos.predict(x_f) , y_std

def ensemble_vote_tof(x,y,fnames_gp_tof_models,model1_tof,featurefunc,ntaylor,model0):
    print('HERE HERE HERE HERE, need to make the ensemble voting using inference_gp_* with all nmodels')
    return

def ensemble_vote_pos(x,y,fnames_gp_pos_models,model1_pos,featurefunc,ntaylor):
    print('HERE HERE HERE HERE, need to make the ensemble voting using inference_gp_* with all nmodels')
    return

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
    print ("KRR model score (test): ", metrics.r2_score(Y_test, Y_test_pred))
    print ("KRR model score (validate): ", metrics.r2_score(Y_valid, Y_valid_pred))

'''



def main():
    do_correlation = True
    nmodels = 32 
    nsamples = 800 # eventually 500
    printascii = False
    taylor_order = 4

    #./data_ave/ind_25-plate_tune_grid_Range_*/analyzed_data.hdf5
    m = re.match('(.*)/(ind.*/)analyzed_data.hdf5',sys.argv[-1])

    modelsfolder = './models%i'%(nsamples)

    X_all = []
    Y_all = []
    X_all,Y_all,Xscaler,Yscaler = loadscaledata(print_mi=True)

    if len(Y_all)<1:
        print("no data loaded")
        return
    print("data loaded\tcontinuing to fitting")

    X_train,X_test,X_valid,X_oob,Y_train,Y_test,Y_valid,Y_oob = katiesplit(X_all,Y_all)

    if m:
        modelsfolder = '%s/models%i'%(m.group(1),nsamples)
        np.savetxt('%s/train_transformed.dat'%(m.group(1)),np.column_stack((X_train,Y_train)))
        np.savetxt('%s/oob_transformed.dat'%(m.group(1)),np.column_stack((X_oob,Y_oob)))

        if do_correlation:
            print('doing correlation')
            outname = '%s/X_Y_featurecorr.dat'%(m.group(1))
            if not crosscorrelation(outname,X_all,Y_all):
                print('Failed crosscorrelation somehow')
                return
    else:
        print('Going to fail model saving/recalling')
        return


    fname_lin_tof,lin_tof = fit_linear_tof(X_oob,Y_oob,modelfolder=modelsfolder)

    # passing the models, not the filenames
    validate_lin_tof(X_test,Y_test,lin_tof)
    validate_lin_tof(X_valid,Y_valid,lin_tof)

    if printascii and m:
        headstring = 'vsetting\tlog(en)\tangle\tlog(tof)\typos\tpredtof'
        np.savetxt('%s/test.dat'%(m.group(1)),np.column_stack((X_test,Y_test,Y_test_pred)),header=headstring)
        np.savetxt('%s/valid.dat'%(m.group(1)),np.column_stack((X_valid,Y_valid,Y_valid_pred)),header=headstring)

    print("Now moving to a taylor expansion of the residuals")

    fname_perturb_tof,perturb_tof,fname_perturb_pos,perturb_pos = fit_taylor_perturbative(
            X_train,
            Y_train,
            featurefunc=featurizeX_taylor,
            ntaylor=taylor_order,
            model0=lin_tof,
            modelfolder='%s/models'%m.group(1))

    # passing the models, not the filenames
    validate_perturb_tof(X_test,Y_test,model=perturb_tof,featurefunc=featurizeX_taylor,ntaylor=taylor_order,model0=lin_tof)
    validate_perturb_tof(X_valid,Y_valid,model=perturb_tof,featurefunc=featurizeX_taylor,ntaylor=taylor_order,model0=lin_tof)
    validate_perturb_pos(X_test,Y_test,model=perturb_pos,featurefunc=featurizeX_taylor,ntaylor=taylor_order)
    validate_perturb_pos(X_valid,Y_valid,model=perturb_pos,featurefunc=featurizeX_taylor,ntaylor=taylor_order)


    if printascii and m:
        headstring = 'vsetting\tlog(en)\tangle\tlog(tof)\typos\tpredtof\tpredypos'
        np.savetxt('%s/test_perturb_linear.dat'%(m.group(1)),np.column_stack((X_test,Y_test,Y_test_pred_tof,Y_test_pred_pos)),header=headstring)
        np.savetxt('%s/valid_perturb_linear.dat'%(m.group(1)),np.column_stack((X_valid,Y_valid,Y_valid_pred_tof,Y_valid_pred_pos)),header=headstring)



    print("Skipping KRR")
    '''
    fname_KRRmodel_tof,KRRmodel_tof,fname_KRRmodel_pos,KRRmodel_pos = fit_krr_perturbative(X_train,Y_train,linmodel_tof,taylormodel_tof,taylormodel_pos,taylororder,modelfolder=modelsfolder,nmodels=nmodels,nsamples=nsamples)

    validate_krr_tof(X_test,Y_test,model=perturb_tof,featurefunc=featurizeX_taylor,model0=lin_tof,n=taylor_order)
    validate_krr_pos(X_test,Y_test,model=perturb_pos,featurefunc=featurizeX_taylor,n=taylor_order)
    '''


    print("Moving on to perturbative GP")


    fnames_gp_tof,fnames_gp_pos = fit_gp_perturbative_ensemble(
            X_train,
            Y_train,
            model1_tof = perturb_tof,
            model1_pos = perturb_pos,
            featurefunc = featurizeX_taylor,
            ntaylor = taylor_order,
            model0=lin_tof,
            modelfolder=modelsfolder,
            nmodels=nmodels,
            nsamples=nsamples)

    Y_test_pred_collect = []
    Y_valid_pred_collect = []

    lin_tof = joblib.load(fname_lin_tof)
    taylor_tof = joblib.load(fname_perturb_tof)
    taylor_pos = joblib.load(fname_perturb_pos)

    gp_tof_models = []
    gp_pos_models = []
    for i in range(nmodels):
        gp_tof_models += [joblib.load(fnames_gp_tof[i])]
        gp_pos_models += [joblib.load(fnames_gp_pos[i])]

    for i in range(nmodels):
        Y_test_pred_tof = validate_gp_tof(X_test,Y_test,gp_tof_models[i],model1_tof = perturb_tof,featurefunc = featurizeX_taylor,ntaylor = taylor_order,model0=lin_tof)
        Y_test_pred_pos = validate_gp_pos(X_test,Y_test,gp_pos_models[i],model1_pos = perturb_pos,featurefunc = featurizeX_taylor,ntaylor = taylor_order)
        Y_valid_pred_tof = validate_gp_tof(X_valid,Y_valid,gp_tof_models[i],model1_tof = perturb_tof,featurefunc = featurizeX_taylor,ntaylor = taylor_order,model0=lin_tof)
        Y_valid_pred_pos = validate_gp_pos(X_valid,Y_valid,gp_pos_models[i],model1_pos = perturb_pos,featurefunc = featurizeX_taylor,ntaylor = taylor_order)

        if len(Y_test_pred_collect)<1:
            Y_test_pred_collect = Yscaler.inverse_transform(np.column_stack((Y_test_pred_tof,Y_test_pred_pos)))
            Y_valid_pred_collect = Yscaler.inverse_transform(np.column_stack((Y_valid_pred_tof,Y_valid_pred_pos)))
        else:
            Y_test_pred_collect = np.column_stack( (Y_test_pred_collect,Yscaler.inverse_transform(np.column_stack((Y_test_pred_tof,Y_test_pred_pos)))) )
            Y_valid_pred_collect = np.column_stack( (Y_valid_pred_collect,Yscaler.inverse_transform(np.column_stack((Y_valid_pred_tof,Y_valid_pred_pos)))) )

    if printascii and m:
        headstring = 'log(vsetting)\tlog(en)\tangle\tlog(tof)\typos\tpredlog(tof)\tpredpos\t...'
        np.savetxt('%s/test_GPperturb.dat'%(m.group(1)),np.column_stack((Xscaler.inverse_transform(X_test),Yscaler.inverse_transform(Y_test),Y_test_pred_collect)),header=headstring)
        np.savetxt('%s/valid_GPperturb.dat'%(m.group(1)),np.column_stack((Xscaler.inverse_transform(X_valid),Yscaler.inverse_transform(Y_valid),Y_valid_pred_collect)),header=headstring)

    return

if __name__ == '__main__':
    main()

