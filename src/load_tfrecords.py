#!/usr/bin/python3.5
import tensorflow as tf
import numpy as np
import sys
import re

def generate_filelist(npulses=int(1),recorddirectory='./data_fs/raw/tf_record_files/',indexname = 'tfrecord.index'):
    ifile=open(recorddirectory+indexname,'r')
    indexlines=ifile.readlines()
    ifile.close()
    d = {}
    headers = []
    for line in indexlines:
        h=re.match('^#\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$',line)
        if h:
            #print('{}\t{}\t{}\t{}'.format(h.group(1),h.group(2),h.group(3),h.group(4)))
            if len(headers) < 1:
                headers = (h.group(1),h.group(2),h.group(3),h.group(4))
        else:
            s=re.match('^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$',line)
            if s:
                #print('{}\t{}\t{}\t{}'.format(s.group(1),s.group(2),s.group(3),s.group(4)))
                d.update({s.group(4) : (int(s.group(1)),int(s.group(2)),float(s.group(3)))})

    tf_filelist = []
    for key in list(d):
        if d[key][0] == int(npulses):
            tf_fname = recorddirectory + 'tfrecord.' + key
            #print('{}\t{}: {}\t{}: {}\t{}: {}'.format(tf_fname,headers[0],d[key][0],headers[1],d[key][1],headers[2],d[key][2]))
            tf_filelist.append(tf_fname)
    return (d,tf_filelist,headers)

def extract_record():
    
    '''
    simsample_tf = tf_train.Example(features = tf_train.Features(feature={
                'nangles': tf_train.Feature(int64_list=tf_train.Int64List(value = [nchannels])),
                'ntbins': tf_train.Feature(int64_list=tf_train.Int64List(value = [ntbins])),
                'nebins': tf_train.Feature(int64_list=tf_train.Int64List(value = [nebins])),
                'npulses': tf_train.Feature(int64_list=tf_train.Int64List(value = [npulses])),
                'waveforms': tf_train.Feature(bytes_list=tf_train.BytesList(value = [WaveForms.tostring()])),
                'tofs': tf_train.Feature(bytes_list=tf_train.BytesList(value = [ToFs.tostring()])),
                'energies': tf_train.Feature(bytes_list=tf_train.BytesList(value = [Energies.tostring()])),
                'timeenergy': tf_train.Feature(bytes_list=tf_train.BytesList(value = [timeenergy.tostring()]))
                }
                ))
    '''


def main():
    recorddirectory = './data_fs/raw/tf_record_files/'
    indexname = 'tfrecord.index'
    npulses = int(1)
    if len(sys.argv)<2:
        print('Need an input npulses to slice on, (default dir = ./data_fs/raw/tf_record_files/\tdefault index = tfrecord.index')
    else:
        npulses = int(sys.argv[1])
    d={}
    tf_filelist = []
    (d,tf_filelist,headers) = generate_filelist(npulses)
    return

if __name__ == '__main__':
    main()
