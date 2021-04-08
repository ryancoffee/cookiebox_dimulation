#!/usr/bin/python3
#Python 3.6.7 (default, Oct 22 2018, 11:32:17) 
#[GCC 8.2.0] on linux
#Type "help", "copyright", "credits" or "license" for more information.

# updating for tensorflow 2.0

import numpy as np
import tensorflow as tf

def my_matmul():
    x_data = np.random.randn(5,10)
    w_data = np.arange(10)
    w_data.shape = (10,1)
    print("x_data = \n{}".format(x_data))
    print("w_data = \n{}".format(w_data))
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32,shape = (5,10))
        w = tf.placeholder(tf.float32,shape = (10,1))
        b = tf.fill((5,1),float(1))
        xw = tf.matmul(x,w)
        wxb = xw + b
        s = tf.reduce_max(wxb)
        with tf.Session() as sess:
            wxb_out = sess.run(wxb,feed_dict={x: x_data, w: w_data})
            s_out = sess.run(s,feed_dict={x: x_data, w: w_data})

        print("wxb_out = \n{}".format(wxb_out))
        print("s_out = {}".format(s_out))

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

    #my_matmul()

    return 0

if __name__ == '__main__':
    main()
