#!/usr/bin/python3

import time
import numpy as np
import sys
import joblib
import re
import os

from sklearn import metrics # remaining printout of GP metrics from main

import PerturbativeUtils
import DataUtils

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def main():

    printascii = False

    #./data_ave/ind_25-plate_tune_grid_Range_*/analyzed_data.hdf5
    m = re.match('(.*)/(ind.*/)analyzed_data.hdf5',sys.argv[-1])
    if not m:
        print('syntax: ./src/load_backinference.py <./data_ave/ind_25-plate_tune_grid_Range_*/analyzed_data.hdf5> ')
        return

    modelsfolder = '%s/backinference'%(m.group(1))
    if not os.path.exists(modelsfolder):
        os.makedirs(modelsfolder)

    X_all,Y_all = DataUtils.loadT2Edata()
    X_all = np.concatenate(
            (X_all,
                np.power(X_all[:,1],int(2)).reshape(-1,1),
                np.power(X_all[:,0],int(2)).reshape(-1,1),
                #np.power(X_all[:,1],int(3)).reshape(-1,1),
                #np.power(X_all[:,0],int(3)).reshape(-1,1),
                (X_all[:,0]*X_all[:,1]).reshape(-1,1),
                (X_all[:,0]*np.power(X_all[:,1],int(2))).reshape(-1,1),
                (X_all[:,1]*np.power(X_all[:,0],int(2))).reshape(-1,1),
                (X_all[:,0]*np.power(X_all[:,1],int(3))).reshape(-1,1),
                (X_all[:,1]*np.power(X_all[:,0],int(3))).reshape(-1,1),
                (np.power(X_all[:,0],int(2))*np.power(X_all[:,1],int(2))).reshape(-1,1)
                ),axis=1)

    X_all,Y_all,X_scaler,Y_scaler = DataUtils.scaledata(X_all,Y_all)
    X_train,X_test,Y_train,Y_test = DataUtils.reservesplit(X_all,Y_all,reserve = .2)

    thetamin = DataUtils.pseudoinversemethod(X_train,Y_train)
    print(thetamin.T)
    print('Left to do: add noise to the X_test[:,1] that is equivalent to .100 ns before log() and apply model, compare to nonoise truth')
    print('\tthat should give the error more honestly since now is it unrealistically excellent resolution for well above 100 eV')
    Y_pred = DataUtils.prependOnesToX(X_test.copy()).dot(thetamin)
    np.savetxt('%s/testpseudoinv.dat'%(modelsfolder),np.column_stack(
        (X_scaler.inverse_transform(X_test),
        Y_scaler.inverse_transform(Y_test),
        Y_scaler.inverse_transform(Y_pred)
        ))
        )

    return

if __name__ == '__main__':
    main()
