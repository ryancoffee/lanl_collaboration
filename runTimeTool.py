#!/reg/g/psdm/sw/conda/inst/miniconda2-prod-rhel7/envs/ana-1.5.28/bin/python

from psana import *
import numpy as np
import sys
import h5py

def weiner(freq):
	#2e5/abs(x**2)+8e8
	f = freq.copy()
	f[0] += 1e-6
	s = 2.*np.power(f,int(-2))
	n = 8e3
	return s/(s+n*16.)

def rolledge(s,w):
	filt = 0.5*(1. - np.cos(np.pi*np.arange(int(w))/float(w)))
	s[:int(w)] *= filt
	s[-1:-int(w)-1:-1] *= filt
	return s

def main():
	runNum = 230
	rolloff = 200
	if len(sys.argv) == 1:
		print('Syntax: %s <runnum> <ipmLowLim=3000> <ipmHighLim=1e6> '%sys.argv[0])
	if len(sys.argv) > 1:
		runNum = int(sys.argv[1])
	expName = 'xcslv3118'
	print('%s\t%s'%(expName,runNum))
	ds = DataSource('exp=%s:run=%d'% (expName,runNum))
	print(DetNames())

	evr0 = Detector('evr0')
	evr1 = Detector('evr1')
	epics=ds.env().epicsStore()

	opal_1 = Detector('opal_1')
	ipm=Detector('XCS-SB1-BMMON')

	ttImg = np.zeros((1,),dtype=float)
	sumImg = np.zeros((1,),dtype=float)
	refImg = np.zeros((1,),dtype=float)
	nref = int(0)
	(r1,r2,c1,c2) = (130,160,0,1024)
	SIG = np.zeros((1,),dtype=float)
	FREQS = np.zeros((1,),dtype=float)
	ipm4_inten = []
	ipmbins = [2**v for v in np.arange(0,16,.125)]

	out = []
	shotlist = []
	loclist = []
	confidencelist = []

	refspectra = []
	sigspectra = []

	h5f = h5py.File('./data/tt_refandsig_run%i.h5'%(runNum), 'w')

	h5f.create_dataset('sigspectra', data=np.zeros((1,1024),dtype=np.uint32), compression="gzip", chunks=True, maxshape=(None,1024)) 
	h5f.create_dataset('refspectra', data=np.zeros((1,1024),dtype=np.uint32), compression="gzip", chunks=True, maxshape=(None,1024)) 

	
	for nevt,evt in enumerate(ds.events()):
		ipm4 = ipm.get(evt)
		lx_delay = epics.getPV('lxt_vitara').data()[0]*1e12 #('XCS:USER:LXTTC').data()[0]
		ttImg = opal_1.raw(evt)
		if (type(ttImg) == type(None)):
			continue
		if ((ipm4 == None) or type(ipm4.TotalIntensity()) == type(None)):
			continue
		''' used only for finding the signal rows in ttImg
		if sumImg.shape[0] == 1:
			sumImg = ttImg.copy()
		else:
			sumImg += ttImg

		if nevt>10:
			sumrows = np.mean(sumImg,axis=1).astype(int)
			for i,v in enumerate(sumrows):
				print('%i: '%i + '+'*int(v/100))
			return
		'''
		ipm4_inten += [ipm4.TotalIntensity()]
		evrcodes = evr0.eventCodes(evt)
		if type(evrcodes) == type(None):
			continue
		shotcodes = list(evrcodes)
		#print(shotcodes,ipm4_inten[-1])
		if FREQS.shape[0] == 1:
			FREQS = np.fft.fftfreq(len(refImg))
		if (int(88) in shotcodes):
			if (not(int(137) in shotcodes) or ipm4_inten[-1] < 500): #A 137 means that there were x-rays
				refspectra += [np.sum(ttImg[r1:r2,c1:c2],axis=0)]
				if refImg.shape[0]==1:
					refImg = refspectra[-1].copy()
				else:
					refImg = 0.9*refImg + 0.1*refspectra[-1].copy()
				nref += 1
				continue
			if (ipm4_inten[-1] > 8000):
				sigspectra += [np.sum(ttImg[r1:r2,c1:c2],axis=0)]
				#print(len(sigspectra))
			if (ipm4_inten[-1] > 3000):
				sig = rolledge(np.sum(ttImg[r1:r2,c1:c2],axis=0).astype(float) - refImg, int(rolloff))
				if SIG.shape[0]==1:
					SIG = np.power(np.abs(np.fft.fft(sig)),int(2))
					continue
				SIG += np.power(np.abs(np.fft.fft(sig)),int(2))
				back = np.fft.ifft(1j*FREQS*np.fft.fft(sig)*weiner(FREQS)).real
				back *= (back>0)
				out += [back]
				pix = np.argmax(back[rolloff:-rolloff]) + rolloff
				weights = 1./np.power(np.abs(np.arange(len(back),dtype=float)-float(pix) + 1j),int(2))
				weights[:rolloff]=0.
				weights[-rolloff:]=0.
				confidence = np.sum(weights*back)
				loc = np.sum(np.arange(len(back))*(weights*back))/np.sum(weights*back)
				if (loc>rolloff and loc<len(back)-rolloff):
					shotlist += [nevt]
					loclist += [loc]
					#loclist += [pix]
					confidencelist += [confidence]
		if len(sigspectra)%100 == 0 and len(sigspectra) > 10:
			print('appending to sigspectra at nevt %i'%nevt)
			data = np.c_[sigspectra]
			if h5f['sigspectra'].shape[0] > 1:
				h5f['sigspectra'].resize(h5f['sigspectra'].shape[0] + data.shape[0], axis=0)
				h5f['sigspectra'][-data.shape[0]:,:] = data
			else:
				h5f['sigspectra'].resize(data.shape[0], axis=0)
				h5f['sigspectra'][:,:] = data
			sigspectra = []

		if len(refspectra)%100 == 0 and len(refspectra) > 10:
			print('appending to refspectra at nevt %i'%nevt)
			data = np.c_[refspectra]
			if h5f['refspectra'].shape[0] > 1:
				h5f['refspectra'].resize(h5f['refspectra'].shape[0] + data.shape[0], axis=0)
				h5f['refspectra'][-data.shape[0]:,:] = data
			else:
				h5f['refspectra'].resize(data.shape[0], axis=0)
				h5f['refspectra'][:,:] = data
			refspectra = []

		if (nevt%100 == 0):
			if (SIG.shape[0]>1):
				np.savetxt('./data/tt_powerspec.run%i.dat'%runNum,np.column_stack((FREQS,SIG)),fmt='%3e')
				np.savetxt('./data/ipm4hist.run%i.dat'%runNum,np.column_stack((ipmbins[:-1],np.histogram(ipm4_inten,ipmbins)[0])),fmt='%3e')

			if len(out)>2:
				np.savetxt('./data/tt_backsignals.run%i.dat'%runNum,np.column_stack((out)),fmt='%3e')
				np.savetxt('./data/tt_locations.run%i.dat'%runNum,np.column_stack((shotlist,loclist,confidencelist)),fmt='%3e')

			'''
			if len(refspectra)>2:
				np.savetxt('./data/tt_refspectra.run%i.dat'%runNum,np.column_stack((refspectra)),fmt='%i')
			if len(sigspectra)>2:
				np.savetxt('./data/tt_sigspectra.run%i.dat'%runNum,np.column_stack((sigspectra)),fmt='%i')
			'''


	h5f.close()


	return

if __name__ == '__main__':
	main()
