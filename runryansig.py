#!/reg/g/psdm/sw/conda/inst/miniconda2-prod-rhel7/envs/ana-1.5.28/bin/python
import sys
from psana import *
import numpy as np

def main():
	if len(sys.argv) == 1:
		print('Syntax: %s <runnum> <ipmLowLim=3000> <ipmHighLim=1e6> '%argv[0])
	runNum = 22
	ipmBgLim = 500
	ipmLowLim = 2500
	ipmHighLim = 1000000
	if len(sys.argv)>1:
		runNum=int(sys.argv[1])
	if len(sys.argv)>2:
		ipmLowLim = int(sys.argv[2])
	if len(sys.argv)>3:
		ipmHighLim = int(sys.argv[3])

	expName = 'xcslv3118'
	print('%s\t%s'%(expName,runNum))
	ds = DataSource('exp=%s:run=%d'% (expName,runNum))
	print(DetNames())
	zyla_ladm = Detector('zyla_ladm')
	epics=ds.env().epicsStore()

	IMG = np.zeros((1,),dtype=float)
	(r1,r2,c1,c2) = (100,300,300,550)

	ipm=Detector('XCS-SB1-BMMON')
	sigbins = np.array([i for i in range(2**16)])
	outmat = []
	sumimg = np.zeros((1,),dtype=np.uint64)
	bgImg = np.zeros((1,),dtype=np.uint64)
	nbg = np.uint64(0)
	ipm4_inten = []
	ipmbins = [2**v for v in np.arange(0,16,.125)]
	ipmhist = np.ones((len(ipmbins)-1),dtype=np.uint64) # using ones since I'm doing the log2 of the result.
	for nstep,step in enumerate(ds.steps()):
		nimg = np.uint64(0)
		print('working step%i'%nstep)
		sigimg = np.zeros((1,),dtype=np.uint64)
		sighist = np.ones((sigbins.shape[0]-1),dtype=np.uint64) # using ones since I'm doing the log2 of the result.
		for nevt,evt in enumerate(step.events()):
			chival = epics.getPV('snd_t4_chi1').data()[0]
			chi_y = epics.getPV('snd_t4_y1').data()[0]
			lx_delay = epics.getPV('lxt_vitara').data()[0]*1e12 #('XCS:USER:LXTTC').data()[0]
			xx_delay = epics.getPV('snd_delay').data()[0]
			ipm4 = ipm.get(evt)
			if (type(ipm4) != type(None)):
				ipm4_inten += [ipm4.TotalIntensity()]
				sigimg = zyla_ladm.raw(evt)
				if (type(sigimg) != type(None)):
					if (ipm4_inten[-1] < ipmBgLim):
						if (bgImg.shape[0] == 1):
							bgImg = sigimg.copy().astype(np.uint64)
							nbg = 1
						else:
							bgImg *= nbg
							bgImg += sigimg.copy().astype(np.uint64)
							nbg += 1
							bgImg = bgImg//nbg
					if ((ipm4_inten[-1]>ipmLowLim) and (ipm4_inten[-1]<ipmHighLim) and (nbg>5)):
						if sumimg.shape!=sigimg.shape:
							sumimg = sigimg.copy().astype(np.uint64)
						else:
							sumimg += sigimg.astype(np.uint64)
						#sighist += np.histogram(sigimg[r1:r2,c1:c2].astype(float)-bgImg[r1:r2,c1:c2].astype(float),sigbins)[0].astype(np.uint64)
						sighist += np.histogram(sigimg[r1:r2,c1:c2],sigbins)[0].astype(np.uint64)
						nimg += 1
	
		headstring = 'chival=%.3f [mdeg]\tchi_y=%.6f [raw]\txx_delay=%.3f[ps]\tlx_delay=%.3f[ps]\tnbg=%i\tnimg=%i'%(chival,chi_y,xx_delay,lx_delay,nbg,nimg)
		print(headstring)
		out = np.log2(sighist) 
		if (nimg>0):
			out -= np.log2(nimg)
		#np.savetxt('./data/sighistbg.run%i.step%i.dat'%(runNum,nstep),out,fmt='%.3f', header = headstring)
		np.savetxt('./data/sighist.run%i.step%i.dat'%(runNum,nstep),out,fmt='%.3f', header = headstring)
		if len(outmat)>0:
			outmat = np.column_stack((outmat,out))
		else:
			outmat = out
		np.savetxt('./data/sighistbg.run%i.dat'%(runNum),outmat,fmt='%.3f', header = headstring)
		np.savetxt('./data/sumimg.run%i.step%i.dat'%(runNum,nstep),sumimg,fmt='%.3f', header = headstring)
		np.savetxt('./data/bgImg.run%i.step%i.dat'%(runNum,nstep),bgImg,fmt='%.3f', header = headstring)
		np.savetxt('./data/sumimg.crop.run%i.step%i.dat'%(runNum,nstep),sumimg[170:220,360:480],fmt='%.3f', header = headstring)
	return

if __name__ == '__main__':
	main()
