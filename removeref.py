#!/usr/bin/python3

import sys
import numpy as np
import h5py
from sklearn.decomposition import PCA

def weiner(freq):
	#2e5/abs(x**2)+8e8
	f = freq.copy()
	f[0] += 1e-6
	s = 1.*np.power(np.abs(f),int(-3))
	n = 1e6 
	return s/(s+n)

def main():
    runNum = 299
    if len(sys.argv)>1:
        runNum = sys.argv[1]
    h5f = h5py.File('data/tt_refandsig_run%i.h5'%int(runNum),'r')
    rolloff = 100

    refs = h5f['refspectra'][()]
    means = np.mean(refs,axis=0)
    refs = np.c_[ [ refs[i,:]-means for i in range(refs.shape[0])] ]
    pca = PCA(n_components=5)
    pca.fit(refs)
    sigs = h5f['sigspectra'][()]
    sigbgs = np.c_[ [sigs[i,:]-means for i in range(sigs.shape[0])] ]
    sigbgs_back = pca.inverse_transform( pca.transform(sigbgs) )
    sigout = sigbgs - sigbgs_back
    np.savetxt('data/tt_pcaout_run%i.dat'%int(runNum),sigout.T)

    SIGOUT = np.fft.fft(sigout,axis=1)
    OUT = np.power(np.abs(np.fft.fft(sigout,axis=1)),int(2)).real
    np.savetxt('data/tt_pcaOUT_run%i.dat'%int(runNum),OUT.T)

    F = np.fft.fftfreq(SIGOUT.shape[1])


    sigback = np.fft.ifft(np.c_[ [1j*F*SIGOUT[i,:] * weiner(F) for i in range(SIGOUT.shape[0])] ], axis=1).real
    np.savetxt('data/tt_sigback_run%i.dat'%int(runNum),sigback.T)
    inds = np.argmax(sigback[:,rolloff:-rolloff],axis=1) + rolloff
    np.savetxt('data/tt_siginds_run%i.dat'%int(runNum),inds,fmt='%i')

    

    h5f.close()
    return

if __name__ == '__main__':
    main()
