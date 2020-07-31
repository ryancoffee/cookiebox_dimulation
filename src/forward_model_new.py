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

    nmodels = 64 # It's looking like 24 or 32 models and 300 samples is good with an elitism of .125 this means we are averaging 4 model results
    # but, the number of models doesn't hurt the latency in FPGA, so nsamples 300 and data set large enough for at least 24 models
    nsamples = 500 # eventually 500
    printascii = False
    taylor_order = 4
    maternnu_tof = 1.5
    maternnu_pos = 1.5
    binslatency = np.logspace(-1,2.7,200)
    elitism = 0.5


    #./data_ave/ind_25-plate_tune_grid_Range_*/analyzed_data.hdf5
    m = re.match('(.*)/(ind.*(logos_.*)/)analyzed_data.hdf5',sys.argv[-1])
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
    X_prime = MathUtils.Rot45(X_train[:,:2])
    X_prime_f = DataUtils.appendTaylorToX(X_prime,n=ntaylor)
    theta1 = DataUtils.pseudoinversemethod(X_prime_f,Y_train[:,0]-Y_pred0)
    Y_pred1 = DataUtils.prependOnesToX(X_prime_f.copy()).dot(theta1)
    np.savetxt('debugging_data.dat',np.column_stack((X_train,Y_train,Y_pred0,Y_pred1)))
    Y_test_result = DataUtils.prependOnesToX( DataUtils.appendTaylorToX( MathUtils.Rot45(X_test[:,:2]) , n=ntaylor) ).dot(theta1) + DataUtils.prependOnesToX(X_test[:,:2].copy()).dot(theta0)
    print('rmse (test) tof in ns: ',  metrics.mean_squared_error(np.exp(Y_test[:,0]),np.exp(Y_test_result),squared=False))
    np.savetxt('debugging_test_data.dat',np.column_stack((X_test,Y_test,Y_test_result)))
    print(theta1)

    Y_tof = Y_train[:,0].copy()-Y_pred0-Y_pred1
    print('Y_tof.shape',Y_tof.shape)
    Y_train_residual = np.column_stack((Y_train[:,0].copy()-Y_pred0-Y_pred1,Y_train[:,1]))
    print('Y_train_residual.shape',Y_train_residual.shape)

    X_train_residual = np.column_stack((X_prime,X_train[:,2]))
    

    if ensemble_all_train:
        nmodels = X_train.shape[0]//nsamples
    print('\t\t========= Using %i models in GP e2tof ensemble ===========\n'%(nmodels))

    modelsfolder = '%s/newensemble%imodels%isamples%.2felitism'%(m.group(1),nmodels,nsamples,elitism)
    if not os.path.exists(modelsfolder):
        os.makedirs(modelsfolder)
        modelsfolder = '%s/newensemble%imodels%isamples%.2felitism'%(m.group(1),nmodels,nsamples,elitism)

    X,Y,Xscaler,Yscaler = DataUtils.minmaxscaledata(X_train_residual,Y_train_residual,feature_range = (-1,1))

    fname_Xscaler = '%s/Xscaler_%s.sav'%(modelsfolder,time.strftime('%Y.%m.%d.%H.%M'))
    fname_Yscaler = '%s/Yscaler_%s.sav'%(modelsfolder,time.strftime('%Y.%m.%d.%H.%M'))
    joblib.dump(Xscaler,fname_Xscaler)
    joblib.dump(Yscaler,fname_Yscaler)



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
        gp_tof_models += [joblib.load(fname,'r')]
    for fname in fnames_gp_pos:
        gp_pos_models += [joblib.load(fname,'r')]

    Y_tof_residual = PerturbativeUtils.ensemble_vote_new(X_test_residual,gp_tof_models,elitism = 0.5)
    Y_pos_residual = PerturbativeUtils.ensemble_vote_new(X_test_residual,gp_pos_models,elitism = 0.5)

    Y_out_residual = np.column_stack((Y_tof_residual,Y_pos_residual))
    Yscaler.inverse_transform(Y_out_residual)
    Y_out_residual[:,0] += Y_test_result

    print('rmse (test) tof in ns: ',  metrics.mean_squared_error(np.exp(Y_test[:,0]),np.exp(Y_out_residual[:,0]),squared=False))
    print('rmse (test) pos in mm: ',  metrics.mean_squared_error(Y_test[:,1],Y_out_residual[:,1],squared=False))





    return


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


    tof_latency = []
    tof_score = []
    pos_latency = []
    pos_score = []
    for i in range(nmodels):
        Y_test_pred_tof,latency,score = PerturbativeUtils.validate_gp_tof(X_test,Y_test,gp_tof_models[i],model1_tof = perturb_tof,featurefunc = PerturbativeUtils.featurizeX_taylor,ntaylor = taylor_order,model0=lin_tof)
        tof_latency += [latency]
        tof_score += [score]
        Y_test_pred_pos,latency,score = PerturbativeUtils.validate_gp_pos(X_test,Y_test,gp_pos_models[i],model1_pos = perturb_pos,featurefunc = PerturbativeUtils.featurizeX_taylor,ntaylor = taylor_order)
        pos_latency += [latency]
        pos_score += [score]
        Y_valid_pred_tof,latency,score = PerturbativeUtils.validate_gp_tof(X_valid,Y_valid,gp_tof_models[i],model1_tof = perturb_tof,featurefunc = PerturbativeUtils.featurizeX_taylor,ntaylor = taylor_order,model0=lin_tof)
        tof_latency += [latency]
        tof_score += [score]
        Y_valid_pred_pos,latency,score = PerturbativeUtils.validate_gp_pos(X_valid,Y_valid,gp_pos_models[i],model1_pos = perturb_pos,featurefunc = PerturbativeUtils.featurizeX_taylor,ntaylor = taylor_order)
        pos_latency += [latency]
        pos_score += [score]

        if len(Y_test_pred_collect)<1:
            Y_test_pred_collect = Yscaler.inverse_transform(np.column_stack((Y_test_pred_tof,Y_test_pred_pos)))
            Y_valid_pred_collect = Yscaler.inverse_transform(np.column_stack((Y_valid_pred_tof,Y_valid_pred_pos)))
        else:
            Y_test_pred_collect = np.column_stack( (Y_test_pred_collect,Yscaler.inverse_transform(np.column_stack((Y_test_pred_tof,Y_test_pred_pos)))) )
            Y_valid_pred_collect = np.column_stack( (Y_valid_pred_collect,Yscaler.inverse_transform(np.column_stack((Y_valid_pred_tof,Y_valid_pred_pos)))) )

    hl,bl = np.histogram(tof_latency,binslatency)
    headstring = '%i ensembles tof latency'
    np.savetxt('%s/latency_hist_tof_matern-nu%.1f.nsamples%i.taylororder%i.dat'%(m.group(1),maternnu_tof,nsamples,taylor_order),np.column_stack((bl[:-1],hl)),fmt='%.3f')
    hl,bl = np.histogram(tof_latency,binslatency)
    headstring = '%i ensembles pos latency'
    np.savetxt('%s/latency_hist_pos-matern-nu%.1f.nsamples%i.taylororder%i.dat'%(m.group(1),maternnu_pos,nsamples,taylor_order),np.column_stack((bl[:-1],hl)),fmt='%.3f')



    stime = time.time()
    Y_oob_pred_tof,Y_oob_std_tof,Y_oob_std_hist_tof = PerturbativeUtils.ensemble_vote_tof(X_oob,gp_tof_models,model1_tof=perturb_tof,featurefunc=PerturbativeUtils.featurizeX_taylor,ntaylor=taylor_order,model0=lin_tof,elitism = elitism)
    tof_latency = (time.time() - stime)*1e6/float(X_oob.shape[0])
    headstring = '%i ensembles tof latency = %i\n#tof_std_bins\tmodel1\t...\tmodeln\tbesteach\tworsteach'%(nmodels,int(tof_latency))
    np.savetxt('%s/std_hist_tof.matern-nu%.1f.nsamples%i.taylororder%i.dat'%(m.group(1),maternnu_tof,nsamples,taylor_order),Y_oob_std_hist_tof)

    stime = time.time()
    Y_oob_pred_pos,Y_oob_std_pos,Y_oob_std_hist_pos = PerturbativeUtils.ensemble_vote_pos(X_oob,gp_pos_models,model1_pos=perturb_pos,featurefunc=PerturbativeUtils.featurizeX_taylor,ntaylor=taylor_order,elitism=elitism)
    pos_latency = (time.time() - stime)*1e6/float(X_oob.shape[0])
    headstring = '%i ensembles pos latency = %i\n#pos_std_bins\tmodel1\t...\tmodeln\tbesteach\tworsteach'%(nmodels,int(pos_latency))
    np.savetxt('%s/std_hist_pos.matern-nu%.1f.nsamples%i.taylororder%i.dat'%(m.group(1),maternnu_pos,nsamples,taylor_order),Y_oob_std_hist_pos)

    print('GP rmse (out-of-bag) log(tof) with voting (before Ysacaler.inverse_transform()): ',  metrics.mean_squared_error(Y_oob[:,0],Y_oob_pred_tof,squared=False))
    print('GP rmse (out-of-bag) pos with voting (before Ysacaler.inverse_transform()): ',  metrics.mean_squared_error(Y_oob[:,1],Y_oob_pred_pos,squared=False))

    Y_oob_result = np.column_stack((Y_oob_pred_tof,Y_oob_pred_pos)).copy()
    Y_oob_result_std = np.column_stack((Y_oob_std_tof,Y_oob_std_pos)).copy()
    headstring = '%i ensembles tof latency = %i\n#X_oob\tY_oob\tY_pred\tY_std'%(nmodels,int(tof_latency))
    np.savetxt('%s/tof_pos_pred_logs.tofmaternnu%.1f.posmaternnu%.1f.nsamples%i.taylororder%i.dat'%(m.group(1),maternnu_tof,maternnu_pos,nsamples,taylor_order),
            np.column_stack((Xscaler.inverse_transform(X_oob),
                Yscaler.inverse_transform(Y_oob),
                Yscaler.inverse_transform(Y_oob_result),
                Yscaler.inverse_transform(Y_oob_result_std)
                ))
            )
    X_oob[:,0] = np.exp(X_oob[:,0])
    X_oob[:,1] = np.exp(X_oob[:,1])
    Y_oob[:,0] = np.exp(Y_oob[:,0])
    Y_oob_result[:,0] = np.exp(Y_oob_result[:,0])
    tof_score = metrics.mean_squared_error(Y_oob[:,0],Y_oob_result[:,0],squared=False)
    pos_score = metrics.mean_squared_error(Y_oob[:,1],Y_oob_result[:,1],squared=False)
    print('GP rmse (out-of-bag) tof with voting in ns: ',  metrics.mean_squared_error(Y_oob[:,0],Y_oob_result[:,0],squared=False))
    print('GP rmse (out-of-bag) pos with voting in mm: ',  metrics.mean_squared_error(Y_oob[:,1],Y_oob_result[:,1],squared=False))
    headstring = '%i ensembles tof latency = %i\n#X_oob\tY_oob\tY_pred\tY_std'%(nmodels,int(tof_latency))
    np.savetxt('%s/tof_pos_pred_eV_ns.tofmaternnu%.1f.posmaternnu%.1f.nsamples%i.taylororder%i.dat'%(m.group(1),maternnu_tof,maternnu_pos,nsamples,taylor_order),
            np.column_stack((X_oob,
                Y_oob,
                Y_oob_result,
                Y_oob_result_std
                ))
            )

    if printascii and m:
        headstring = 'log(vsetting)\tlog(en)\tangle\tlog(tof)\typos\tpredlog(tof)\tpredpos\t...'
        np.savetxt('%s/test_GPperturb.dat'%(m.group(1)),np.column_stack((Xscaler.inverse_transform(X_test),Yscaler.inverse_transform(Y_test),Y_test_pred_collect)),header=headstring)
        np.savetxt('%s/valid_GPperturb.dat'%(m.group(1)),np.column_stack((Xscaler.inverse_transform(X_valid),Yscaler.inverse_transform(Y_valid),Y_valid_pred_collect)),header=headstring)

    fname = 'performanceVparams.dat'
    f = open(fname,'a')
    outrowstring = '#nsamples\tnmodels\ttaylor_order\telitism\ttof_latency\tpos_latency\ttof_score\tpos_score\n'
    outrowstring += '%i\t%i\t%i\t%.2f\t%.1f\t%.1f\t%.3f\t%.3f'%(nsamples,nmodels,taylor_order,elitism,tof_latency,pos_latency,tof_score,pos_score)
    print(outrowstring,file=f)
    f.close()
if __name__ == '__main__':
    main()

