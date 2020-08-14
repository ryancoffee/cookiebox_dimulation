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
    print('Training the multilayer stacked ensemble')
    do_correlation = True
    usealltrain = False
    nmodels = 32 # It's looking like 24 or 32 models and 300 samples is good with an elitism of .125 this means we are averaging 4 model results
    # but, the number of models doesn't hurt the latency in FPGA, so nsamples 300 and data set large enough for at least 24 models
    nsamples = 100 # eventually 500
    printascii = True
    taylor_order = 4
    maternnu_tof = 1.5
    maternnu_pos = 1.5
    binslatency = np.logspace(-1,2.7,200)
    elitism = 0.5


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

    if usealltrain:
        nmodels = X_bags[0].shape[0]//nsamples
        print('\t\t========= Using %i models in GP e2tof ensemble ===========\n'%(nmodels))

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


    print('moving onto the ensemble after removing principle dependence with theta0 linear model')

    '''
    print('scaling data')
    X_all,Y_all,Xscaler,Yscaler = DataUtils.minmaxscaledata(X_all,Y_all,feature_range = (-1,1))

    fname_Xscaler = '%s/Xscaler_%s.sav'%(modelsfolder,timestamp)
    fname_Yscaler = '%s/Yscaler_%s.sav'%(modelsfolder,timestamp)
    joblib.dump(Xscaler,fname_Xscaler)
    joblib.dump(Yscaler,fname_Yscaler)
    '''

    nsamples = 300
    nmodels = X_bags[1].shape[0]//nsamples
    print('ensembling %i models with %i samples'%(nmodels,nsamples))
    X = np.column_stack((X_bags[1],DataUtils.prependOnes(X_bags[1].copy()).dot(theta0))) # this should handle the removing of the residual from the linear order of the feature... like boosting I think
    X,Polyscaler = DataUtils.polyfeaturize(X,order=6) # this has a sharp cutoff at 6, 7 has no improvement, 5 is nearly up to 1ns 
    X,Y,Xscaler,Yscaler = DataUtils.minmaxscaledata(X,Y_bags[1][:,0].copy(),feature_range = (-1,1))
    thetas = [np.linalg.pinv(X[i*nsamples:(i+1)*nsamples,:]).dot(Y[i*nsamples:(i+1)*nsamples]) for i in range(nmodels)]
    #print('Ensemble thetas:')
    #[print('%s'%(th)) for th in thetas]
    X = np.column_stack((X_test.copy(),DataUtils.prependOnes(X_test.copy()).dot(theta0)))
    X = Xscaler.transform( Polyscaler.transform(X) )
    Y_preds = [Yscaler.inverse_transform( X.dot(th).reshape(-1,1) ) for th in thetas]

    errors = [metrics.mean_squared_error(np.exp(Y_test[:,0]), np.exp(Y_pred),squared=False) for Y_pred in Y_preds]
    h,b = np.histogram(errors,10)
    print ("histogram of polynomial ensemble model rmse [ns]:")
    [print('%.2f\t%s'%(b[i],'.'*h[i])) for i in range(len(h))]
    #([metrics.mean_squared_error(np.exp(Y_test[:,0]), np.exp(Y_pred),squared=False) for Y_pred in Y_preds]) )
    print ("polynomial model rmse (mean) [ns]: \n%s"%( metrics.mean_squared_error(np.exp(Y_test[:,0]), np.exp( np.sum(np.column_stack((Y_preds)),axis = 1 )/len(Y_preds)) ,squared=False) ) )

    print('Next on deck is to route the predistions as features into the aggregator')

    return

    '''
    PerturbativeUtils.validate_lin_tof(X_test,Y_test,lin_tof)

    print("Now moving to a taylor expansion of the residuals")
    theta = [ DataUtils.pseudoinversemethod(X_bag1[i*nsamples:(i+1)*nsamples,:],Y_bag1[i*nsamples:(i+1)*nsamples] - lin_tof.predict(X_bag1[i*nsamples:(i+1)*nsamples,:])) for i in range(nmodels)]
    print(theta.T)



    print("\n\n\t\t================\tMoving on to ensemble of blenders\t=================\n")


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
    '''

if __name__ == '__main__':
    main()

