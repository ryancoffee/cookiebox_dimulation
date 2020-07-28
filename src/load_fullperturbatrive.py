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

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def main():
    do_correlation = True
    usealltrain = True
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
    m = re.match('(.*)/(ind.*(logos_.*)/)analyzed_data.hdf5',sys.argv[-1])
    if not m:
        print('syntax: ./src/load_full_perturbatrive.py <./data_ave/ind_25-plate_tune_grid_Range_*/analyzed_data.hdf5> ')
        return


    X_all = []
    Y_all = []
    X_all,Y_all,Xscaler,Yscaler = DataUtils.loadscaledata(print_mi=True)

    if len(Y_all)<1:
        print("no data loaded")
        return
    print("data loaded\tcontinuing to fitting")

    #X_train,X_test,X_valid,X_oob,Y_train,Y_test,Y_valid,Y_oob = DataUtils.katiesplit(X_all,Y_all)
    X_train,X_test,Y_train,Y_test = DataUtils.reservesplit(X_all,Y_all,reserve = .1)
    X_train,X_valid,Y_train,Y_valid = DataUtils.reservesplit(X_train,Y_train,reserve = .1)
    X_train,X_oob,Y_train,Y_oob = DataUtils.reservesplit(X_train,Y_train,reserve = .1)

    if usealltrain:
        nmodels = X_train.shape[0]//nsamples
        print('\t\t========= Using %i models in GP e2tof ensemble ===========\n'%(nmodels))

    modelsfolder = '%s/ensembletests%imodels%isamples'%(m.group(1),nmodels,nsamples)
    if not os.path.exists(modelsfolder):
        os.makedirs(modelsfolder)
        modelsfolder = '%s/ensembletests%imodels%isamples'%(m.group(1),nmodels,nsamples)
    np.savetxt('%s/train_transformed.dat'%(m.group(1)),np.column_stack((X_train,Y_train)))
    np.savetxt('%s/oob_transformed.dat'%(m.group(1)),np.column_stack((X_oob,Y_oob)))
    fname_Xscaler = '%s/Xscaler_%s.sav'%(modelsfolder,time.strftime('%Y.%m.%d.%H.%M'))
    fname_Yscaler = '%s/Yscaler_%s.sav'%(modelsfolder,time.strftime('%Y.%m.%d.%H.%M'))
    joblib.dump(Xscaler,fname_Xscaler)
    joblib.dump(Yscaler,fname_Yscaler)

    if do_correlation:
        print('doing correlation')
        outname = '%s/X_Y_featurecorr.dat'%(m.group(1))
        if not DataUtils.crosscorrelation(outname,X_all,Y_all):
            print('Failed crosscorrelation somehow')


    print('Reserving X_oob/Y_oob for ensemble testing')
    fname_lin_tof,lin_tof = PerturbativeUtils.fit_linear_tof(X_train,Y_train,modelfolder=modelsfolder)

    print('Reserving X_valid/Y_valid for ensemble validation')
    # passing the models, not the filenames
    PerturbativeUtils.validate_lin_tof(X_test,Y_test,lin_tof)
    #PerturbativeUtils.validate_lin_tof(X_valid,Y_valid,lin_tof)

    if printascii and m:
        headstring = 'vsetting\tlog(en)\tangle\tlog(tof)\typos\tpredtof'
        np.savetxt('%s/test.dat'%(m.group(1)),np.column_stack((X_test,Y_test,Y_test_pred)),header=headstring)
        np.savetxt('%s/valid.dat'%(m.group(1)),np.column_stack((X_valid,Y_valid,Y_valid_pred)),header=headstring)

    print("Now moving to a taylor expansion of the residuals")

    fname_perturb_tof,perturb_tof,fname_perturb_pos,perturb_pos = PerturbativeUtils.fit_taylor_perturbative(
            X_train,
            Y_train,
            featurefunc=PerturbativeUtils.featurizeX_taylor,
            ntaylor=taylor_order,
            model0=lin_tof,
            modelfolder=modelsfolder)

    # passing the models, not the filenames
    PerturbativeUtils.validate_perturb_tof(X_test,Y_test,model=perturb_tof,featurefunc=PerturbativeUtils.featurizeX_taylor,ntaylor=taylor_order,model0=lin_tof)
    PerturbativeUtils.validate_perturb_pos(X_test,Y_test,model=perturb_pos,featurefunc=PerturbativeUtils.featurizeX_taylor,ntaylor=taylor_order)

    if printascii and m:
        headstring = 'vsetting\tlog(en)\tangle\tlog(tof)\typos\tpredtof\tpredypos'
        np.savetxt('%s/test_perturb_linear.dat'%(m.group(1)),np.column_stack((X_test,Y_test,Y_test_pred_tof,Y_test_pred_pos)),header=headstring)
        np.savetxt('%s/valid_perturb_linear.dat'%(m.group(1)),np.column_stack((X_valid,Y_valid,Y_valid_pred_tof,Y_valid_pred_pos)),header=headstring)


    print("\n\n\t\t================\tMoving on to perturbative GP\t=================\n")


    fnames_gp_tof,fnames_gp_pos = PerturbativeUtils.fit_gp_perturbative_ensemble(
            X_train,
            Y_train,
            maternnu_tof = maternnu_tof,
            maternnu_pos = maternnu_pos,
            model1_tof = perturb_tof,
            model1_pos = perturb_pos,
            featurefunc = PerturbativeUtils.featurizeX_taylor,
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

