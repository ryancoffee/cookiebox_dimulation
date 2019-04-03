#!/bin/python3

import numpy as np
import tensorflow as tf

print(tf.VERSION)
path = './data_fs/raw/'
filename = path + 'CookieBox.tfrecords'
print(filename)
#'./data_fs/raw/CookieBox.1pulses.image0001.tfrecords'

dataset = tf.data.TFRecordDataset(filename)
print(dataset)


