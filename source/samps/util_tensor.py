#!/usr/bin/env python
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
# 
# def spinCouple(tf0,qr1,qr2):
# def quasiRDM(rdm,qr1):
# def genQredDic(qr3):
# def expandRotL(rotL,qr1,qr2,qr12,qr3):
# def lastSite(site1,qnumsl,ne,ms,sval):
# 
import numpy
import util_spinsym
import util_angular

# Transformation to (S1M1,a)(S2M2,b)=>(S3M3,ab) basis for decimation
def spinCouple(tf0,qr1,qr2):
   # Offsets
   off_ful1 = util_spinsym.offs_ful(qr1)
   off_ful2 = util_spinsym.offs_ful(qr2)
   # Loop over symmetry blocks
   dim1 = util_spinsym.dim_ful(qr1)
   dim2 = util_spinsym.dim_ful(qr2)
   shp = tf0.shape
   assert dim1 == shp[0]
   assert dim2 == shp[1]
   tf1 = numpy.zeros((shp[0]*shp[1],shp[2]))
   qr12 = []
   ioff = 0
   for i1 in range(qr1.shape[0]):
      q1 = qr1[i1]
      n1,s1,d1 = q1
      dg1 = int(2*s1+1); dr1 = int(d1); df1 = dg1*dr1
      of1 = off_ful1[i1]
      for i2 in range(qr2.shape[0]):
         q2 = qr2[i2]
         n2,s2,d2 = q2
         dg2 = int(2*s2+1); dr2 = int(d2); df2 = dg2*dr2
         of2 = off_ful2[i2]
         # SaMa,SbMb
	 tmp = tf0[of1:of1+df1,of2:of2+df2,:].copy() #(ma,nb,X)
	 tmp = tmp.reshape(dg1,dr1,dg2,dr2,shp[2])
	 # Combine SaMa,SbMb=>SM
         smin = abs(s1-s2)
         smax = s1+s2
         ns = int(smax-smin)+1
         for ij in range(ns):
   	    sij = smin+ij
	    dg12 = int(2*sij+1); dr12 = dr1*dr2; df12 = dg12*dr12
   	    qr12.append([n1+n2,sij,dr12])
	    ift,cgt = util_angular.cgtensor(s1,s2,sij)
	    assert ift == True
	    # Transform
	    transM = numpy.tensordot(cgt,tmp,axes=([0,1],[0,2])) # mnl,manbX->labX
	    transM = transM.reshape(df12,shp[2])
	    # Save
	    tf1[ioff:ioff+df12,:] = transM.copy()
	    ioff += df12
   # Check
   qr12 = numpy.array(qr12)
   dim3 = util_spinsym.dim_ful(qr12)
   assert dim3 == dim1*dim2
   assert dim3 == ioff
   return qr12,tf1

# Quasi-RDM as a totally symmetric RDM
def quasiRDM(wA1,qr1):
   # Offsets
   off_red1 = util_spinsym.offs_red(qr1)
   off_ful1 = util_spinsym.offs_ful(qr1)
   # Dimension
   dim_red1 = util_spinsym.dim_red(qr1)
   dim_ful1 = util_spinsym.dim_ful(qr1)
   assert dim_ful1 == wA1.shape[0] 
   # Quasi-RDM is reduced quantity
   qrdm = numpy.zeros((dim_red1,dim_red1))
   qnums = []
   for i1 in range(qr1.shape[0]):
      q1 = qr1[i1]
      n1,s1,d1 = q1
      dg1 = int(2*s1+1); dr1 = int(d1); df1 = dg1*dr1
      or1 = off_red1[i1]
      of1 = off_ful1[i1]
      tmp = wA1[of1:of1+df1,:].copy()
      tmp = tmp.dot(tmp.T)
      tmp = tmp.reshape(dg1,dr1,dg1,dr1)
      qrdm[or1:or1+dr1,or1:or1+dr1] = numpy.einsum('mamb->ab',tmp)
      qnums += [[n1,s1]]*dr1 
   qrdm = qrdm/numpy.trace(qrdm)
   return qnums,qrdm

#
# KEY PART: An expansion of reduced rotation matrix to full-SM dense tensor
# 	    by multiplying the Clebsch-Gordan coefficients.
#
# Key-index map
def genQredDic(qr3):
   dic = {}
   for idx,val in enumerate(qr3):
      key = str(val[:2])
      if key not in dic: 
	 dic[key] = idx
      else:
	 print 'error: repeated key in reduced dimension!'
	 exit(1)
   return dic

def expandRotL(rotL,qr1,qr2,qr12,qr3):
   dic = genQredDic(qr3)
   # Offsets
   off_red3 = util_spinsym.offs_red(qr3)
   off_ful1 = util_spinsym.offs_ful(qr1)
   off_ful2 = util_spinsym.offs_ful(qr2)
   off_ful3 = util_spinsym.offs_ful(qr3)
   off_red12 = util_spinsym.offs_red(qr12)
   off_ful12 = util_spinsym.offs_ful(qr12)
   # Loop over symmetry blocks
   dim1 = util_spinsym.dim_ful(qr1)
   dim2 = util_spinsym.dim_ful(qr2)
   dim3 = util_spinsym.dim_ful(qr3)
   tful = numpy.zeros((dim1,dim2,dim3))
   # Loop over bra and determine ket symmetries
   i12 = -1
   for i1 in range(qr1.shape[0]):
      q1 = qr1[i1]
      n1,s1,d1 = q1
      dg1 = int(2*s1+1); dr1 = int(d1); df1 = dg1*dr1
      of1 = off_ful1[i1]
      for i2 in range(qr2.shape[0]):
         q2 = qr2[i2]
         n2,s2,d2 = q2
         dg2 = int(2*s2+1); dr2 = int(d2); df2 = dg2*dr2
         of2 = off_ful2[i2]
	 # Combine SaMa,SbMb=>SM
         smin = abs(s1-s2)
         smax = s1+s2
         ns = int(smax-smin)+1
         for ij in range(ns):
 	    i12 += 1		 
   	    sij = smin+ij
	    dg12 = int(2*sij+1); dr12 = dr1*dr2; df12 = dg12*dr12
	    or12 = off_red12[i12]
	    of12 = off_ful12[i12]
	    # Locate (n1+n2,sij) sectors in the reduced dimensions
	    key = str(numpy.array([n1+n2,sij]))
	    if key not in dic: continue
	    i3 = dic[key]
            q3 = qr3[i3]
            n3,s3,d3 = q3
	    dg3 = int(2*s3+1); dr3 = int(d3); df3 = dg3*dr3
	    or3 = off_red3[i3]
	    of3 = off_ful3[i3] 
	    # Reshaping the reduced-dense tensor <SaNaSbNb||ScNc>
	    tmp = rotL[or12:or12+dr12,or3:or3+dr3].copy()
	    tmp = tmp.reshape(dr1,dr2,dr3) # (a,b,c)
	    # Get CG coefficient
	    ift,cgt = util_angular.cgtensor(s1,s2,sij)
	    assert ift == True
	    # Expansion
	    tmp = numpy.einsum('mnl,abc->manblc',cgt,tmp)
	    tmp = tmp.reshape(df1,df2,df3)
	    # Save into full tensor
	    tful[of1:of1+df1,of2:of2+df2,of3:of3+df3] = tmp.copy()
   return tful

# |Psi>
def lastSite(rotL1,site1,srotR,qnumsl,ne,ms,sval):
   print '[util_tensor.lastSite] Population analysis of MPS'
   off_red = util_spinsym.offs_red(qnumsl)
   off_ful = util_spinsym.offs_ful(qnumsl)
   pop = 0.0
   for i1 in range(len(qnumsl)):
      n1,s1,d1 = qnumsl[i1] 
      dg1 = int(2*s1+1); dr1 = int(d1); df1 = dg1*dr1
      of1 = off_ful[i1]
      for im1 in range(dg1): 
         coeff = srotR[of1+im1*dr1:of1+(im1+1)*dr1].copy()
         msi = im1-s1 
         wti = numpy.sum(coeff**2)
         pop += wti
         print ' qsym(N,S,M)=',(n1,s1,msi),'dr=',dr1,'wt=',wti,'accum=',pop
         if abs(ms-msi)>1.e-8 and wti > 1.e-10:
            print 'error: The M-adapted MPS is not correct!'
            #exit(1)
   if abs(pop-1.0)>1.e-5:
      print ' warning: losing norm for the normalized left-MPS! pop=',pop
   # Select
   qnum = numpy.array([ne,sval])
   key = str(qnum)
   print ' Target key =',key,' original ms=',ms
   dic = genQredDic(qnumsl)
   if key not in dic:
      print ' >>> No such state for sym=',key
      info = 0
      rotL = numpy.zeros(0)
      site = numpy.zeros(0)
      wt   = 0.
   else:
      info = 1	
      i1 = dic[key]
      n1,s1,d1 = qnumsl[i1] 
      dg1 = int(2*s1+1); dr1 = int(d1); df1 = dg1*dr1
      or1 = off_red[i1]
      of1 = off_ful[i1]
      im1 = int(ms+sval)
      ista = of1+im1*dr1
      iend = ista+dr1
      coeff = srotR[ista:iend].copy()
      wt = numpy.linalg.norm(coeff)
      coeff = coeff/wt
      wt = wt**2
      rotL = numpy.tensordot(rotL1[:,or1:or1+dr1],coeff,axes=([1],[0]))
      site = numpy.tensordot(site1[:,:,ista:iend],coeff,axes=([2],[0])) #lnr,ri->lni
      print ' weight =',wt,' for qsym=',(n1,s1,ms),' site.shape=',site.shape
   qnum = numpy.array([[ne,sval,1]])
   return info,qnum,rotL,site,wt 
