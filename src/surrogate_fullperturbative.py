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
        fnames_gp_tof_ensemble = []
        m = re.match('.*gp_model_tof.*',name)
        if m:
            fnames_gp_tof_ensemble += [m.group(0)]
        fnames_gp_pos_ensemble = []
        m = re.match('.*gp_model_pos.*',name)
        if m:
            fnames_gp_pos_ensemble += [m.group(0)]

    return fname_Xscaler,fname_Yscaler,fname_linear_tof,fname_taylor_tof,taylor_tof_order,fname_taylor_pos,taylor_pos_order,fnames_gp_tof_ensemble,fnames_gp_pos_ensemble
        


def main():
    fname_Xscaler,fname_Yscaler,fname_linear_tof,fname_taylor_tof,taylor_tof_order,fname_taylor_pos,taylor_pos_order,fnames_gp_tof_ensemble,fnames_gp_pos_ensemble = parsemodels(sys.argv[1:])
    Xscaler = joblib.load(fname_Xscaler,'r')
    Yscaler = joblib.load(fname_Yscaler,'r')
    linear_tof = joblib.load(fname_linear_tof,'r')
    taylor_tof = joblib.load(fname_taylor_tof,'r')
    taylor_pos = joblib.load(fname_taylor_pos,'r')

    return

if __name__ == '__main__':
    main()
