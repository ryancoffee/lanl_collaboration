#!/usr/bin/python3

import sys
import numpy as np
import h5py
from sklearn.decomposition import PCA

def weiner(freq):
	#2e5/abs(x**2)+8e8
	f = freq.copy()
	f[0] += 1e-6
	s = 1.*np.power(np.abs(f),int(-2))
	n = 1e6 
	return s/(s+n)

def main():
    runNum = 299
    if len(sys.argv)>1:
        runNum = sys.argv[1]
    h5f = h5py.File('data/tt_refandsig_run%i.h5'%int(runNum),'r+')
    rollfront = 200
    rollback = 100


    refs = np.log2(h5f['refspectra'][()].astype(float))
    means = np.mean(refs,axis=0)
    refs = np.c_[ [ refs[i,:]-means for i in range(refs.shape[0])] ]
    pca = PCA(n_components=5)
    pca.fit(refs)
    sigs = np.log2(h5f['sigspectra'][()].astype(float))
    sigbgs = np.c_[ [sigs[i,:]-means for i in range(sigs.shape[0])] ]
    sigbgs_back = pca.inverse_transform( pca.transform(sigbgs) )
    sigout = sigbgs - sigbgs_back
    np.savetxt('data/tt_pcaout_log2_run%i.dat'%int(runNum),sigout.T)

    SIGOUT = np.fft.fft(sigout,axis=1)
    OUT = np.power(np.abs(np.fft.fft(sigout,axis=1)),int(2)).real
    np.savetxt('data/tt_pcaOUT_log2_run%i.dat'%int(runNum),OUT.T)

    F = np.fft.fftfreq(SIGOUT.shape[1])


    sigback = 2.**32*(np.power(float(2),np.fft.ifft(np.c_[ [1j*F*SIGOUT[i,:] * weiner(F) for i in range(SIGOUT.shape[0])] ], axis=1).real) - 1.)
    np.savetxt('data/tt_sigback_log2_run%i.dat'%int(runNum),sigback.T)
    inds = np.argmax(sigback[:,rollfront:-rollback],axis=1) + rollfront
    centroids = np.zeros(len(inds),dtype=np.uint32)
    stopinds = np.full(len(inds),1023,dtype=np.uint32)
    startinds = np.full(len(inds),0,dtype=np.uint32)
    for i in range(sigback.shape[0]):
        stopinds[i] = np.argmin(sigback[i,inds[i]:(inds[i]+1024)//2]) + inds[i]
        startinds[i] = np.argmin(sigback[i,inds[i]//2:inds[i]]) + inds[i]//2
        v = sigback[i,startinds[i]:stopinds[i]].copy()
        v *= (v>0)
        c = np.sum(v*np.arange(v.shape[0]))/np.sum(v) + float(startinds[i])
        if c>0 and c<900:
            centroids[i] = int(c) 
        else:
            centroids[i] = int(0)
    if 'sigcentroids' in h5f.keys():
        h5f['sigcentroids'].resize(centroids.shape[0],axis=0)
        h5f['sigcentroids'][:] = centroids
    else:
        h5f.create_dataset('sigcentroids', data=centroids, compression="gzip", chunks=True) 
    #np.savetxt('data/tt_siginds_log2_run%i.dat'%int(runNum),np.column_stack((inds,startinds,stopinds,centroids)),fmt='%i')
    if 'refmeans' in h5f.keys():
        h5f['refmeans'].resize(means.shape[0],axis=0)
        h5f['refmeans'][:] = means 
    else:
        h5f.create_dataset('refmeans', data=means, compression="gzip", chunks=True) 
    if 'pcamat' in h5f.keys():
        h5f['pcamat'].resize(pca.components_.shape[0]*pca.components_.shape[1],axis=0)
        h5f['pcamat'].reshape(pca.components_.shape)
        h5f['pcamat'][:,:] = pca.components_ 
    else:
        h5f.create_dataset('pcamat', data=pca.components_, compression="gzip", chunks=True) 
    h5f.close()


    print('testing PCA phase')

    centered = (sigs[100,:] - means).reshape(1,-1)
    S100 = pca.transform(centered)
    mapping = pca.components_
    print(S100)
    print(np.inner(mapping,centered))
    

    return

if __name__ == '__main__':
    main()
