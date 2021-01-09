#!/usr/bin/python3

import numpy as np
import re
import sys
import h5py
import cv2

def main():
    fnames = ['./data_sinograms/debug.ImgSegSim.h5']
    wrappings = 3
    if len(sys.argv)==1:
        print('syntax: %s <hdf5 path/filename>'%sys.argv[0])
    if len(sys.argv)>1:
        fnames = sys.argv[1:]
    for fname in fnames:
        f = h5py.File(fname,'r')

        imkeys = []
        for key in list(f.keys()):
            m = re.search('^img\d+',key)
            if m != None:
                imkeys += [key]

        images = [f[k] for k in imkeys]
        imnum = 0
        for image in images:
            pulses = [image[p] for p in list(image.keys())]
            chan = 0
            h,w = pulses[0]['hist'][()].shape
            c = 3
            outimg = np.zeros((h,wrappings*w,c),dtype=np.uint8)
            wrapmask = .5*(1-np.cos(np.arange(w)/w*np.pi))

            for pulse in pulses:
                if chan < 3:
                    mat = np.tile(pulse['hist'][()],(wrappings,))
                    for j in range(mat.shape[0]):
                        mat[j,:w] = (256*(1-np.cos(np.arange(w)/w*np.pi)) * mat[j,:w])//256
                        mat[j,-w:] = (256*(1+np.cos(np.arange(1,w+1)/w*np.pi)) * mat[j,-w:])//256
                    outimg[:,:,chan] = mat
                chan += 1
            outimg = cv2.normalize(outimg,  None, 0, 255, cv2.NORM_MINMAX)
            cv2.imshow('image%05i'%imnum,outimg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print(np.max(outimg))
            imnum += 1



        f.close()
    return

if __name__ == '__main__':
    main()
