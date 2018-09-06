#!/usr/bin/env python
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
#
# def compact_rotL(flmps1,path):
# def merged_rotL(rotL,qr1,qr2,qr3,debug=False):
#
import os
import numpy
import shutil
from util_spinsym import dpt_red,offs_red 
from util_tensor import genQredDic

def compact_rotL(flmps1,path,ifaux=False):
   print '\n[block_itrf.compact_rotL] path =',path
   try:
      shutil.rmtree(path)
   except:  
      pass	   
   os.mkdir(path)
   # Start conversion
   nsite = flmps1['nsite'].value
   qnums = [None]*(nsite+1)
   for isite in range(nsite+1):
      key = 'qnumNS'+str(isite)
      qnums[isite] = flmps1[key].value
   # to compact rotL
   qnumsN = numpy.array([[0,0.,1],[1,0.5,1],[2,0.,1]])
   # qnumNS0
   qnum = flmps1['qnumNS0'].value
   sval = qnum[0][1]
   naux = int(2*sval)
  
   # Put auxilliary sites explicitly
   if ifaux:
      print ' naux=',naux
      for isite in range(naux):
         qnumISite = numpy.array([isite,isite*0.5,1,1])
         fp = numpy.memmap(path+'/quanta'+str(isite),dtype='float64',\
      		     mode='write',order='C',shape=qnumISite.shape)
         fp[:] = qnumISite[:]
         rotL1 = numpy.ones(1)
         fp = numpy.memmap(path+'/rotL'+str(isite),dtype='float64',\
		           mode='write',order='C',shape=rotL1.shape)
         fp[:] = rotL1[:]
      ioff = naux
   else:
      ioff = 0

   qnumISite = numpy.array([2*sval,sval,1,1])
   fp = numpy.memmap(path+'/quanta'+str(ioff),dtype='float64',\
		     mode='write',order='C',shape=qnumISite.shape)
   fp[:] = qnumISite[:]
   for isite in range(nsite):
      key0 = 'rotL'+str(isite)
      rotL = flmps1[key0].value
      print ' * isite=',isite,' rotL=',rotL.shape
      rdic = merged_rotL(rotL,qnums[isite],qnumsN,qnums[isite+1])
      # Qnums
      qnum = qnums[isite+1]
      qnumISite = []
      rotL1 = []
      for idx in range(qnum.shape[0]):
	 n1 = qnum[idx][0]
	 s1 = qnum[idx][1]
	 key = str(qnum[idx][:2])
	 d1,d2 = rdic[key].shape
	 qnumISite.append([n1,s1,d1,d2])
	 rotL1.append(rdic[key].reshape(d1*d2))
      qnumISite = numpy.array(qnumISite)
      fp = numpy.memmap(path+'/quanta'+str(ioff+isite+1),dtype='float64',\
		        mode='write',order='C',shape=qnumISite.shape)
      fp[:] = qnumISite[:]
      rotL1 = numpy.hstack(rotL1)
      fp = numpy.memmap(path+'/rotL'+str(ioff+isite),dtype='float64',\
		        mode='write',order='C',shape=rotL1.shape)
      fp[:] = rotL1[:]
   return 0

# Get the map
def merged_rotL(rotL,qr1,qr2,qr3,debug=False):
   # for (sa,sb,sc) get (na,nb,nc)
   fnorm2 = numpy.linalg.norm(rotL)**2
   tensorDic = {}
   qr12 = dpt_red(qr1,qr2)
   off_red12 = offs_red(qr12)
   off_red3 = offs_red(qr3)
   dic = genQredDic(qr3)
   i12 = -1
   for i1 in range(qr1.shape[0]):
      q1 = qr1[i1]
      n1,s1,d1 = q1
      dg1 = int(2*s1+1); dr1 = int(d1)
      for i2 in range(qr2.shape[0]):
         q2 = qr2[i2]
         n2,s2,d2 = q2
         dg2 = int(2*s2+1); dr2 = int(d2)
         # Combine Sa,Sb=>Sab
         smin = abs(s1-s2)
         smax = s1+s2
         ns = int(smax-smin)+1
         for ij in range(ns):
    	    i12 += 1		 
      	    sij = smin+ij
   	    dg12 = int(2*sij+1); dr12 = dr1*dr2
   	    or12 = off_red12[i12]
   	    # Locate (n1+n2,sij) sectors in the reduced dimensions
   	    key = str(numpy.array([n1+n2,sij]))
   	    if key not in dic: continue
   	    i3 = dic[key]
            q3 = qr3[i3]
            n3,s3,d3 = q3
   	    dg3 = int(2*s3+1); dr3 = int(d3)
   	    or3 = off_red3[i3]
   	    # Reshaping the reduced-dense tensor <SaNaSbNb||ScNc>
   	    tmp = rotL[or12:or12+dr12,or3:or3+dr3].copy()
	    if debug: print '(Sa,Sb,Sc),(na,nb,nc)=',(s1,s2,s3),(dr1,dr2,dr3)
	    # Only block diagonal
	    if key not in tensorDic:
	       tensorDic[key] = [tmp]
	    else:
	       tensorDic[key].append(tmp)
   rdic = {}
   fnorm2b = 0.
   idx = 0
   for key in sorted(tensorDic.keys()):
      rdic[key] = numpy.vstack(tensorDic[key])
      print '   idx=',idx,' key=',key,' dim=',rdic[key].shape
      idx += 1
      fnorm2b += numpy.linalg.norm(rdic[key])**2
   print '   diffFnorm2=',fnorm2-fnorm2b
   print
   assert abs(fnorm2-fnorm2b)<1.e-10
   return rdic
