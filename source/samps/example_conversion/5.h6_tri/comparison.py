import h5py
import numpy

flmps0 = h5py.File('lmps1F','r')
flmps1 = h5py.File('lmps1T','r')
nsite = flmps0['nsite'].value
for isite in range(nsite):
   key = 'site'+str(isite)
   s0 = flmps0[key].value
   s1 = flmps1[key].value
   print '\nisite=',isite,s0.shape,s1.shape,numpy.linalg.norm(s0+s1)
   #print 's0=\n',s0
   #print 's1=\n',s1
