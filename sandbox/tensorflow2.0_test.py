#!/usr/bin/python3
#Python 3.6.7 (default, Oct 22 2018, 11:32:17) 
#[GCC 8.2.0] on linux
#Type "help", "copyright", "credits" or "license" for more information.

# updating for tensorflow 2.0

import numpy as np
import tensorflow as tf

def main():
    print('{}'.format(tf.executing_eagerly()))
    init_val = np.random.normal(5,2,(10,2)).T
    w = np.arange(10).reshape((-1,1))

    var = tf.matmul(init_val,w)
    print("pre-run: \n{}".format(var))
    a = tf.constant([[1.,  0.4,  0.5],
                 [0.4, 0.2,  0.25],
                 [0.5, 0.25, 0.35]])
    res = tf.matmul(tf.linalg.pinv(a), a)
    print('{}'.format(res))

    t = np.arange(1024,dtype=float).reshape((-1,1))
    t0=64.
    w=16.
    phi=1.
    y = np.exp(-((t-t0)/w)**2)*np.cos(t*2*np.pi/w+phi).T
    Y = tf.signal.dct(y,type=2)
    #Y = tf.signal.dct(np.row_stack((y,np.flip(y,axis=0))),type=2)
    #y_back = tf.signal.dct(Y,type=3)
    #y_back2 = tf.signal.idct(Y*np.arange(len(Y))/len(Y),type=4)
    #np.savetxt('temp.dat',np.column_stack((t,y,y_back[:len(y),0],y_back2[:len(y),0])),fmt='%.3f')
    np.savetxt('temp_y.dat',y.T,fmt='%.3f')
    np.savetxt('temp_Y.dat',np.array(Y).T,fmt='%.3f')

    return 0

if __name__ == '__main__':
    main()
