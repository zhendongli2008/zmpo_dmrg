#!/usr/bin/env python

#Read FCIDUMP and store it in numpy file
import numpy

def loadRDMs(n):
   print '\n[tools_io.loadRDMs] from spatial_twopdm.0.0.txt'
   x = numpy.loadtxt('spatial_twopdm.0.0.txt')
   rdm2 = x[:,4].reshape((n,n,n,n))
   # G[ijkl]=1/2*<ais1+ajs2^+aks2als1>
   twopdm = 2.0*rdm2
   onepdm = numpy.einsum('ijjl->il',rdm2)/(n-1.0)*2.0
   print 'ne_act=',numpy.trace(onepdm)
   numpy.save('onepdm',onepdm)
   numpy.save('twopdm',twopdm)
   print 'finished'
   return onepdm,twopdm

def loadERIs():
   print '\n[tools_io.loadERIs] from FCIDUMP'
   with open('FCIDUMP','r') as f:
     line = f.readline().split(',')[0].split(' ')[-1]
     print  'Num of orb: ', int(line)
     f.readline()
     f.readline()
     f.readline()
     n = int(line)
     e = 0.0
     int1e = numpy.zeros((n,n))
     int2e = numpy.zeros((n,n,n,n))
     for line in f.readlines():
       data = line.split()
       ind = [int(x)-1 for x in data[1:]]
       if ind[2] == -1 and ind[3]== -1:
         if ind[0] == -1 and ind[1] ==-1:
           e = float(data[0])
         else :
           int1e[ind[0],ind[1]] = float(data[0])
           int1e[ind[1],ind[0]] = float(data[0])
       else:
         int2e[ind[0],ind[1], ind[2], ind[3]] = float(data[0])
         int2e[ind[1],ind[0], ind[2], ind[3]] = float(data[0])
         int2e[ind[0],ind[1], ind[3], ind[2]] = float(data[0])
         int2e[ind[1],ind[0], ind[3], ind[2]] = float(data[0])
         int2e[ind[2],ind[3], ind[0], ind[1]] = float(data[0])
         int2e[ind[3],ind[2], ind[0], ind[1]] = float(data[0])
         int2e[ind[2],ind[3], ind[1], ind[0]] = float(data[0])
         int2e[ind[3],ind[2], ind[1], ind[0]] = float(data[0])
   numpy.save('int2e',int1e)
   numpy.save('int1e',int2e)
   print 'finished'
   return e,int1e,int2e

def loadMOLMF(chkfile='hs_bp86.chk'):
   print '\n[tools_io.loadMOLMF] from PYSCF.chk'
   from pyscf import scf
   mol,mf = scf.chkfile.load_scf(chkfile)
   return mol,mf

if __name__ == '__main__': 
   e,int1e,int2e = loadERIs()
   print e
