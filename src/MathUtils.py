#!/urs/bin/python3

import numpy as np

def Rot45(X):
    if len(X.shape)<2:
        print('shapes failed in DataUtils.Rot45()')
        return X 
    return X[:,:2].copy().dot(1./np.sqrt(2.)*np.array(((1,-1),(1,1)))) # note this is a transpose to how it is usually represented, since we take columns as x1,x2

def RotPIovr4(X):
    return Rot45(X)
