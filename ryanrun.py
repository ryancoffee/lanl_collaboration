#!/reg/g/psdm/sw/conda/inst/miniconda2-prod-rhel7/envs/ana-1.5.28/bin/python
import sys
from psana import *
import numpy as np
from matplotlib import pyplot as plt

'''
def masklow(mat,dx,dy):
	szh,szv = mat.shape
	mask = np.ones(mat.shape,dtype=float)
	suppress = mask[:dx//2,:dy//2]
	return mat
'''

def get_spacing(pix,wrap=False):
	#fringe(x)=(512./(x)*.6685)
	if wrap:
		return .6685 * 512./(256+(256-x))
	else:
		return .6685 * 512./x

def main():
	if len(sys.argv) == 1:
		print('Syntax: %s <runnum> <ipmLowLim=3000> <ipmHighLim=1e6> '%argv[0])
	runNum = 22
	ipmLowLim = 3000
	ipmHighLim = 1000000
	if len(sys.argv)>1:
		runNum=int(sys.argv[1])
	if len(sys.argv)>2:
		ipmLowLim = int(sys.argv[2])
	if len(sys.argv)>3:
		ipmHighLim = int(sys.argv[3])

	expName = 'xcslv3118'
	print('%s\t%s'%(expName,runNum))
	ds = DataSource('exp=%s:run=%d:smd'% (expName,runNum))
	print(DetNames())
	zyla = Detector('zyla')
	epics=ds.env().epicsStore()

	IMG = np.zeros((1,),dtype=float)

	ipm=Detector('XCS-SB1-BMMON')
	for nstep,step in enumerate(ds.steps()):
		print('working step%i'%nstep)
		IMG = np.zeros((512,1800),dtype=float)
		for nevt,evt in enumerate(step.events()):
			ipmdet = ipm.get(evt)
			if (type(ipmdet) != type(None)):
				chival = epics.getPV('snd_t4_chi1').data()[0]
				chi_y = epics.getPV('snd_t4_y1').data()[0]
				xx_delay = epics.getPV('snd_delay').data()[0]
				ipm4 = ipmdet.TotalIntensity()
				if (ipm4>ipmLowLim and ipm4<ipmHighLim):
					img = zyla.raw(evt)
					if (type(img) != type(None)):
						IMG += np.power(np.abs(np.fft.fft2(img)),int(2)).real
	
		IMG[0,0]=0
		szx,szy = IMG.shape
		spectout = np.log2(np.sum(IMG,axis=1))[:IMG.shape[0]//2]
		chivals = [chival]*len(spectout)
		delayvals = [xx_delay]*len(spectout)

		headstring = 'chival=%.3f [mdeg]\tchi_y=%.6f [raw]\txx_delay=%.3f'%(chival,chi_y,xx_delay)
		print(headstring)
		np.savetxt('./data/spect.run%i.step%i.dat'%(runNum,nstep),np.c_[chivals,delayvals,spectout],fmt='%.4f',header=headstring)
		np.savetxt('./data/power2d.run%i.step%i.dat'%(runNum,nstep),np.roll(IMG,IMG.shape[1]//2,axis=1)[:szx//2,szy//4:(3*szy)//4],fmt='%.4f',header=headstring)
	#	np.savetxt('./data/img2d.run%i.step%i.dat'%(runNum,nstep),img,fmt='%.4f')
	return

if __name__ == '__main__':
	main()
