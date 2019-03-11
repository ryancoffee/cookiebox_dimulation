#!/home/coffee/anaconda3/bin/python3
import numpy as np
import tensorflow as tf

def mysum(a,x):
    return a + x
elems = np.array(['T','e','n','s','o','r',' ','F','l','o','w'])
scan_sum = tf.scan(mysum, elems)
init = tf.global_variables_initializer()
scan_sum2 = tf.scan(lambda a,x: a+x,elems)

sess1 = tf.Session()
sess2 = tf.Session()
sess1.run(tf.global_variables_initializer())
sess2.run(tf.global_variables_initializer())
out1 = sess1.run(scan_sum)
out2 = sess2.run(scan_sum2)

print(out1,out2)
