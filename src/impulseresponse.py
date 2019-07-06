#!/usr/bin/python3.5

import numpy as np
import sys

def main():
    if len(sys.argv)<2:
        print('add files to process on command line')
        return
    filelist = sys.argv[1:]
    ir = np.zeros(None,dtype=float)
    signals = []
    dt = 0
    for fname in filelist:
        fi = open(fname, "r")
        for passline in range(6):
            headline = '# ' + fi.readline()
        (t,v) = fi.readline().split()
        v_vec=[float(v)]
        t_vec=[float(t)]
        for line in fi:
            (t,v) = line.split()
            v_vec = v_vec + [float(v)/1.*float(np.power(int(2),int(11)))]
            t_vec = t_vec + [float(t)*1.e9]
        fi.close()
        #Get the time-step for sake of frequencies
        v_vec = np.roll(v_vec,int(-0.45*len(v_vec)))
        oname = fname + '.roll'
        np.savetxt(oname,np.column_stack((t_vec,v_vec)),fmt='%.1f')
        if len(signals)<2:
            headstring = '#' + '\t'.join(str(t_vec))
            dt = t_vec[1]-t_vec[0]

        sigmin = np.min(v_vec[:300])
        if ((sigmin>-600) * (sigmin<-200) and not (np.min(v_vec[400:])<-100)):
            if len(signals)<2:
                signals = np.array(v_vec)
            else:
                signals = np.row_stack((signals,v_vec))

    outfile = './data_fs/processed/rolledout.dat'
    np.savetxt(outfile,signals,fmt='%.2f')
    SIGNALS = np.fft.fft(signals,axis=1)
    f = np.fft.fftfreq(SIGNALS.shape[1],dt)
    AMPS=np.array(np.abs(SIGNALS))
    PHASES = np.angle(SIGNALS)
    PHASES_ = np.copy(PHASES)
    R = SIGNALS.shape[0]
    L = SIGNALS.shape[1]
    P = np.zeros(L,dtype=float)
    P[:L//2] = np.mean(np.unwrap(PHASES_[:,:L//2],axis=1),axis=0)-np.pi
    P[L//2:] = np.mean(np.fliplr(np.unwrap(np.fliplr(PHASES_[:,L//2:]),axis=1)),axis=0)+np.pi
    outfile = './data_fs/processed/phasesout.dat'
    np.savetxt(outfile,P,fmt='%.2f')
    slope = np.mean(np.diff(P[:150]))/(f[1]-f[0])
    P = np.tile(slope*f,R)
    P.shape = AMPS.shape
    SIGNALSOUT = AMPS * np.exp(1j*(PHASES-P))
    signalsout = np.fft.ifft(SIGNALSOUT).real
    signalsout = np.roll(signalsout,L//5,axis=1)
    outfile = './data_fs/processed/signalsout.dat'
    np.savetxt(outfile,signalsout,fmt='%.2f')

    return

if __name__ == '__main__':
    main()
