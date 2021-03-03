#!/usr/bin/python3

import numpy as np
import h5py
import sys
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.data.experimental import TFRecordWriter
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Feature, Features, Example

def phase2id(phase):
    # careful, playing tricks with the 8 bit overflow in order to wrap the phase into 0..2pi and discritize into 0..15
    p = np.uint8(float(phase)/2./np.pi*256)
    return np.uint8(int(p)*16//256)

def energy2id(energy,e0,ewin):
    return np.uint8(float(energy-e0)*256/ewin)

def serialize_example(feature0, label0):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        'feature0': tf.train.Int64List(feature0),
        'label0': tf.train.Int64List(label0)
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def main():
    if len(sys.argv)<2:
        print('syntax: %s <h5 file to convert> <optional nshards/h5>'%sys.argv[0])
    h5filenames = [name for name in sys.argv[1:]]
    nshardsPfile = 4

    print(h5filenames)
    maxnchannels = 8

    for fname in h5filenames:
        f = h5py.File(fname,'r')
        tfrec_writer = tf.io.TFRecordWriter('%s.tfrecord'%fname)
        imkeys = [k for k in list(f.keys()) if re.match('^img\d+',k)]
        for k in imkeys:
            pulses = [p for p in list(f[k].keys()) if re.match('^pulse\d+',p)]
            carrier = f[k].attrs['carrier']
            shp = f[k][pulses[0]]['hist'][()].shape
            mchist = np.zeros((shp[0],shp[1],maxnchannels),dtype=int)
            ymat = np.zeros((maxnchannels,2))
            if len(pulses)<maxnchannels:
                pid = 0
                for p in pulses:
                    i = phase2id( carrier - f[k][p].attrs['phase'] )
                    e = energy2id(f[k][p].attrs['esase'] , 450,128)
                    mchist[:,:,pid] = f[k][p]['hist'][()]
                    ymat[pid,:] = [i , e]
                    pid += 1

            imexample_proto = serialize_example(mchist,ymat);
            print(imexample_proto)
            '''
            imexample = Example(
                    features = Features(
                        feature = {
                            'mchist': Feature(int64_list = Int64List([ mchist.tostring() ]))
                            , 'ymat': Feature(int64_list = Int64List([ ymat.tostring() ]))
                            }
                        )
                    )
            f.write(imexample.SerializeToString())
            '''
            f.write(imexample_proto)

        f.close()
        tfrec_writer.close()
    return

if __name__=='__main__':
    main()
