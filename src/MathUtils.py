#!/urs/bin/python3

import numpy as np

def Rot45(X):
    if X.shape[1] == 2:
        return X.copy().dot(1./np.sqrt(2.)*np.array(((1,-1),(1,1)))) # note this is a transpose to how it is usually represented, since we take columns as x1,x2
    print('shapes failed in DataUtils.Rot45()')
    return X.copy()

def RotPIovr4(X):
    return Rot45(X)
