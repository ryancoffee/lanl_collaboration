#!/reg/g/psdm/sw/conda/inst/miniconda2-prod-rhel7/envs/ana-1.5.28/bin/python
import sys
from psana import *
import numpy as np

def main():
	runNum = 22
	ipmLowLim = 3000
	if len(sys.argv)>1:
		runNum=int(sys.argv[1])
	if len(sys.argv)>2:
		nevents = int(sys.argv[2])
	if len(sys.argv)>3:
		ipmLowLim = int(sys.argv[4])

	expName = 'xcslv3118'
	print('%s\t%s'%(expName,runNum))
	ds = DataSource('exp=%s:run=%d:smd'% (expName,runNum))
	print(DetNames())
	zyla_ladm = Detector('zyla_ladm')
	zyla = Detector('zyla')
	epics=ds.env().epicsStore()

	IMG = np.zeros((1,),dtype=float)

	ipm=Detector('XCS-SB1-BMMON')
	sigbins = np.array([i for i in range(2**16)])
	for nstep,step in enumerate(ds.steps()):
		print('working step%i'%nstep)
		IMG = np.zeros((512,2000),dtype=float)
		sigimg = np.zeros((2000,512),dtype=np.uint32)
		sighist = np.zeros((sigbins.shape[0]-1),dtype=int)
		for nevt,evt in enumerate(step.events()):
			chival = epics.getPV('snd_t4_chi1').data()[0]
			ipm4 = ipm.get(evt).TotalIntensity()
			if (ipm4>ipmLowLim):
				img = zyla.raw(evt)
				sigimg = zyla_ladm.raw(evt)
				sighist += np.histogram(img,sigbins)[0]
				IMG += np.power(np.abs(np.fft.fft2(img)),int(2)).real
	
		IMG[0,0]=0
		np.savetxt('./data/power2d.run%i.step%i.dat'%(runNum,nstep),np.roll(np.roll(IMG,IMG.shape[0]//2,axis=0),IMG.shape[1]//2,axis=1),fmt='%.4f')
		np.savetxt('./data/img2d.run%i.step%i.dat'%(runNum,nstep),img,fmt='%.4f')
		np.savetxt('./data/sighist.run%i.step%i.dat'%(runNum,nstep),sighist,fmt='%i')
	return

if __name__ == '__main__':
	main()
