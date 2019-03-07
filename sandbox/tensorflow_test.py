#!/usr/bin/python3
#Python 3.6.7 (default, Oct 22 2018, 11:32:17) 
#[GCC 8.2.0] on linux
#Type "help", "copyright", "credits" or "license" for more information.

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

    init_val = tf.random_normal((10,5),5,2)
    var = tf.Variable(init_val,name='var')
    print("pre-run: \n{}".format(var))
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        post_var = sess.run(var)
    print("\npost run: \n{}".format(post_var))

    my_matmul()

    return 0

if __name__ == '__main__':
    main()
