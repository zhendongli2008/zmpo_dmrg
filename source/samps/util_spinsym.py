#!/usr/bin/env python
#
# Couplings of spin angular momentums
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
# 
# def dim_sym_red(q1):
# def dim_sym_ful(q1):
# def dim_red(qr1):
# def dim_ful(qr1):
# def offs_red(qr1):
# def offs_ful(qr1):
#
# def dpt_red(qr1,qr2):
# def collect(qred):
#
# def expandSM(qr):
# def ifpermissible(k,n,s):
# def ifpermit(k,n,sval,sl):
# def symScreen(isite,nsite,rotL,qred,sigs,ne,sval,debug=True,status='L',ifscreen=True):
#
import numpy
import itertools

# Dimension of one block
def dim_sym_red(q1):
   return int(q1[2])

def dim_sym_ful(q1):
   return int(2*q1[1]+1)*int(q1[2])

# Dimension of one direction
def dim_red(qr1):
   dim = 0
   for i in range(qr1.shape[0]):
      dim += dim_sym_red(qr1[i])
   return dim

def dim_ful(qr1):
   dim = 0
   for i in range(qr1.shape[0]):
      dim += dim_sym_ful(qr1[i])
   return dim

# Offsets
def offs_red(qr1):
   offs = []
   ioff = 0
   for i in range(qr1.shape[0]):
      offs.append(ioff) 
      ioff += dim_sym_red(qr1[i])
   return offs

def offs_ful(qr1):
   offs = []
   ioff = 0
   for i in range(qr1.shape[0]):
      offs.append(ioff)
      ioff += dim_sym_ful(qr1[i])
   return offs

# We use an unmerged block tensor structure for direct product,
# The merge will be implemented later.
def dpt_red(qr1,qr2):
   qr12 = []
   lst = []
   for qi in qr1:
      ni,si,dimi = qi
      for qj in qr2:
         nj,sj,dimj = qj
         smin = abs(si-sj)
         smax = si+sj
         ns = int(smax-smin)+1
         for ij in range(ns):
            sij = smin+ij
            qr12.append([ni+nj,sij,dimi*dimj])
   return numpy.array(qr12)

# Collect qnums results from SVD into format: {[n,s,dim]}
def collect(qred):
   dic = {}
   for idx,val in enumerate(qred):
      dic.setdefault(str(val),[]).append(idx)
   # Ordered counting
   qr = []
   key_current = None
   for item in qred:
      key = str(item)
      if key != key_current:
	 qr.append(item+[len(dic[key])])
         key_current = key
   return numpy.array(qr)

# Expand {[n,s,dim]} into {[n,ms]} for interface
def expandSM(qr):
  qsm = []
  for q1 in qr:
     n1,s1,d1 = q1
     # (2s+1)(2d+1)
     for im in range(int(2*s1+1)):
        m1 = -s1+im
	qsm += [[n1,m1]]*int(d1)
  return numpy.array(qsm)

# (K,N,S) - check for the high-spin det.
def ifpermissible(k,n,s):
   na = int(n/2.0+s) 
   nb = int(n/2.0-s)
   return na>=0 and na<=k and \
   	  nb>=0 and nb<=k and \
	  abs(na+nb-n)<1.e-8 \
	  and abs(na-nb-2.0*s)<1.e-8

def ifpermit(k,n,sval,sl):
   srmin = abs(sval-sl)
   srmax = sval+sl
   ns = int(srmax-srmin)+1
   ifperm = False
   for i in range(ns):
      sr = srmin+i
      ifperm = ifperm or ifpermissible(k,n,sr)
   return ifperm

# Symmetry Screening - only start to be effective for isite>nsite/2 !!!
def symScreen(isite,nsite,rotL,qred,sigs,ne,sval,debug=True,status='L',ifscreen=True,\
	      ifBlockSymScreen=False):
   if status == 'L':
      kres = nsite-1-isite
   # when isite=1, there should be one left.
   elif status == 'R':
      kres = isite 
   if debug: print '\n[mpo_dmrg_conversion.symScreen] dimt=',len(sigs),' (kres,ne,sval)=',(kres,ne,sval)
   dic = {}
   for idx,val in enumerate(qred):
      dic.setdefault(str(val),[]).append(idx)
   # Ordered counting
   qr = []
   key_current = None
   idx = 0
   pop = 0.
   indx = []
   for item in qred:
      key = str(item)
      if key != key_current:
         wt = numpy.sum(sigs[dic[key]])
	 pop += wt
	 n1,s1 = item
	 nres = ne-n1
	 
	 # Check the possibility for coupling to the right
	 if ifBlockSymScreen:
	    ifcouple = True
	    if n1 > ne+1.e-10: ifcouple = False
	    if abs(n1-ne)<1.e-10 and s1>0.0: ifcouple = False
	    if n1 < ne-1.e-10 and s1>(ne-n1)/2.0: ifcouple = False
	    # Not complete for example: H10 isite=6 has quanta=[8,2], but 
	    # the remaining 3 sites cannot contribute to s=2 for singelt!
         else:
	    ifcouple = ifpermit(kres,nres,sval,s1)

	 if not debug:
	    print ' idx=',idx,'item=',item,'dim=',len(dic[key]),\
	          ' sigs=',wt,'pop=',pop,'ifcouple=',ifcouple
	 key_current = key
	 idx += 1 
	 # Save for permitted case
	 if (not ifscreen) or (ifscreen and ifcouple):
	    qr.append(item+[len(dic[key])])
	    indx += dic[key]
   # Update
   qnumsl = numpy.array(qr)
   rotLnew = rotL[:,indx].copy() 
   if debug: print ' Screend from',rotL.shape[-1],'to',rotLnew.shape[-1]
   return qnumsl,rotLnew

if __name__ == '__main__':
   qr1 = numpy.array([[0,0.,1]])
   qr2 = numpy.array([[0,0.,1],[1,0.5,1],[2,0.,1]])
   qr12 = dpt_red(qr1,qr2)
   print qr12
   print dim_ful(qr12)
   qr12 = dpt_red(qr12,qr2)
   print qr12
   print dim_ful(qr12)
   qr12 = dpt_red(qr12,qr2)
   print qr12
   print dim_ful(qr12)

   qred = [[1.0, 0.5], [1.0, 0.5], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 1.0], [3.0, 0.5], [3.0, 0.5]]
   print
   print len(qred)
   print qred
   print collect(qred)
   
   qr2 = numpy.array([[0,0.,1],[1,0.5,1],[2,0.,1]])
   print
   print qr2
   print expandSM(qr2)

   qr2 = numpy.array([[ 1., 0.5, 2. ],[ 2., 0. , 3. ],\
	     	      [ 2.,  1., 1. ],[ 3., 0.5, 2. ]])
   print 
   print qr2
   print expandSM(qr2)

   print
   print ifpermissible(2,1,1)
   print ifpermissible(2,1,0)
   print ifpermissible(2,1,0.5)
   print ifpermissible(3,4,1)
