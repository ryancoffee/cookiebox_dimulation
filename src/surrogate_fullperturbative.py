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



def parsemodels(nameslist):
    fname_Xscaler = ''
    fname_Yscaler = ''
    fname_linear_tof = ''
    fname_taylor_tof = ''
    fname_taylor_pos = ''
    taylor_tof_order = 0
    taylor_pos_order = 0
    fnames_gp_tof_ensemble = []
    fnames_gp_pos_ensemble = []
    for name in nameslist:
        m = re.match('.*Xscaler.*',name)
        if m:
            fname_Xscaler = m.group(0)
        m = re.match('.*Yscaler.*',name)
        if m:
            fname_Yscaler = m.group(0)
        m = re.match('.*linear_model_tof.*',name)
        if m:
            fname_linear_tof = m.group(0)
        m = re.match('.*perturb_taylor_model_tof.order(\d+).*',name)
        if m:
            fname_taylor_tof = m.group(0)
            taylor_tof_order = int(m.group(1))
        m = re.match('.*perturb_taylor_model_pos.order(\d+).*',name)
        if m:
            fname_taylor_pos = m.group(0)
            taylor_pos_order = int(m.group(1))
        m = re.match('.*gp_model_tof.*',name)
        if m:
            fnames_gp_tof_ensemble += [m.group(0)]
        m = re.match('.*gp_model_pos.*',name)
        if m:
            fnames_gp_pos_ensemble += [m.group(0)]

    return fname_Xscaler,fname_Yscaler,fname_linear_tof,fname_taylor_tof,taylor_tof_order,fname_taylor_pos,taylor_pos_order,fnames_gp_tof_ensemble,fnames_gp_pos_ensemble
        


def main():
    fname_Xscaler,fname_Yscaler,fname_linear_tof,fname_taylor_tof,taylor_tof_order,fname_taylor_pos,taylor_pos_order,fnames_gp_tof_ensemble,fnames_gp_pos_ensemble = parsemodels(sys.argv[1:])
    f = open(fname_Xscaler,'rb')
    Xscaler = joblib.load(f)
    f.close()
    f = open(fname_Yscaler,'rb')
    Yscaler = joblib.load(f)
    f.close()
    f = open(fname_linear_tof,'rb')
    linear_tof = joblib.load(f)
    f.close()
    f = open(fname_taylor_tof,'rb')
    taylor_tof = joblib.load(f)
    f.close()
    f = open(fname_taylor_pos,'rb')
    taylor_pos = joblib.load(f)
    f.close()

    elitism = .25

    gp_tof = []
    for name in fnames_gp_tof_ensemble:
        f = open(name,'rb')
        gp_tof += [joblib.load(f)]
        f.close()
    gp_pos = []
    for name in fnames_gp_pos_ensemble:
        f = open(name,'rb')
        gp_pos += [joblib.load(f)]
        f.close()

    print(len(fnames_gp_tof_ensemble),len(gp_tof))

    nsamples = int(1e3)
    vsettings = np.ones(nsamples,dtype=float) * 100.
    energies = np.random.normal(0,.25,nsamples) + np.random.uniform(10,90,nsamples)
    angles = np.abs(np.random.normal(0.,2.,nsamples))
    phis = np.random.uniform(-np.pi,np.pi,nsamples)

    X = np.column_stack((np.log(vsettings),np.log(energies),angles))
    Xscaler.transform(X)

    stime = time.time()
    Y_gp_pred_tof,Y_std_tof,Y_std_hist_tof = PerturbativeUtils.ensemble_vote_tof(X,gp_tof,model1_tof=taylor_tof,featurefunc=PerturbativeUtils.featurizeX_taylor,ntaylor=taylor_tof_order,model0=linear_tof,elitism = elitism)
    Y_gp_pred_pos,Y_std_pos,Y_std_hist_pos = PerturbativeUtils.ensemble_vote_pos(X,gp_pos,model1_pos=taylor_pos,featurefunc=PerturbativeUtils.featurizeX_taylor,ntaylor=taylor_pos_order,elitism = elitism)

    Y_vote_gp = Yscaler.inverse_transform(np.column_stack((Y_gp_pred_tof,Y_gp_pred_pos)))
    Y_vote_gp[:,0] = np.exp(Y_vote_gp[:,0])


    m = re.match('(.*ensemble.*)/.*sav',sys.argv[-1])
    outname = 'out_time_%s.out'%(time.strftime('%Y.%m.%d.%H.%M'))
    path = 'data_ave/%s'%(outname)
    if m:
        path = '%s/surroggate_out'%(m.group(1))
    if not os.path.exists(path):
        os.makedirs(path)

    X_f = PerturbativeUtils.featurizeX_taylor(X,taylor_pos_order)
    Y_pred_tof = linear_tof.predict(X[:,1].reshape(-1,1)) + taylor_tof.predict(X_f)
    Y_pred_pos = taylor_pos.predict(X_f)
    Y_pred = np.column_stack((Y_pred_tof,Y_pred_pos))
    Yscaler.inverse_transform(Y_pred)
    Xscaler.inverse_transform(X)
    X[:,:2] = np.exp(X[:,:2])
    Y_pred[:,0] = np.exp(Y_pred[:,0])

    #np.savetxt('%s/%s'%(path,outname),np.column_stack((X,Y_pred_tof,Y_pred_pos,Y_std_tof,Y_std_pos)))
    np.savetxt('%s/%s'%(path,outname),np.column_stack((X,Y_pred,Y_vote_gp)),fmt='%.2f')
    print(Y_pred[:10,:])

    return

if __name__ == '__main__':
    main()
