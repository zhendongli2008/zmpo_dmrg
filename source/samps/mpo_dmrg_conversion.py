#!/usr/bin/env python
#
# Convert MPS[N,Ms] to spin pure states MPS[N,S,(Ms)]
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
#
# def sweep_projection(flmps0,flmps1,ifQt,sval,thresh=1.e-4,Dcut=-1,\
# 		       debug=False,ifcompress=True,ifBlockSingletEmbedding=True):
# def left_sweep_projection(flmps0,flmps1,qtarget,thresh,Dcut,debug,\
#			    ifBlockSingletEmbedding=False):
# def right_sweep_projection(flmps0,flmps1,qtarget,thresh,Dcut,debug,\
#			     ifBlockSingletEmbedding=False):
# 
import os
import h5py
import numpy
import itertools
from zmpo_dmrg.source import mpo_dmrg_io
from zmpo_dmrg.source import mpo_dmrg_qparser
import util_tensor
import util_spinsym

#
# * Optimize the memory by dumping into file during transformation.
#
# * Assume left canonical form is to be obtained (NQt version).
#
# * For Qt case, the qtensor_api should be used to convert it into NQt case.
#   Since the MPS is small (~G), this step is not the limiting step compared
#   with the latter energy evaluation.
#
# * Also save rotL (many-body basis coefficient) & reduced dimensions. Note 
#   the last rotL is not necessary for the calculation will construct the last 
#   dot exactly with the rotL[k-1] many-body basis.
#
# * Note that no quantum number information of lmps0 is used at all!
#
def sweep_projection(flmps0,flmps1,ifQt,sval,thresh=1.e-4,Dcut=-1,\
		     debug=False,ifcompress=True,\
		     ifBlockSingletEmbedding=True,\
		     ifBlockSymScreen=True,\
		     ifpermute=False):
   nsite = flmps0['nsite'].value
   # This is the only qnum information used.
   ne,ms = flmps0['qnum'+str(nsite)].value[0]
   print '\n[mpo_dmrg_samps.sweep_projection] ifQt=',ifQt,\
	 ' (nsite,,ne,ms,sval)=',(nsite,ne,ms,sval)
   assert not ifQt
   # Left projection 
   if not ifcompress:
      wt = left_sweep_projection(flmps0,flmps1,[nsite,ne,ms,sval],thresh,Dcut,debug)
   else:
      flmpsL = h5py.File('./lmpsL_tmp','w')
      flmpsR = h5py.File('./lmpsR_tmp','w')
      wt1 = left_sweep_projection(flmps0,flmpsL,[nsite,ne,ms,sval],thresh,Dcut,debug)
      wt2 = right_sweep_projection(flmpsL,flmpsR,[nsite,ne,ms,sval],thresh,Dcut,debug)
      if not ifBlockSingletEmbedding: assert not ifBlockSymScreen
      # In case of singlet embedding: the wts should be 1/(2s+1) (e.g, 0.332 for s=1)
      wt3 = left_sweep_projection(flmpsR,flmps1,[nsite,ne,ms,sval],thresh,Dcut,debug,\
		      	          ifBlockSingletEmbedding,ifBlockSymScreen,ifpermute)
      flmpsL.close()
      flmpsR.close()
      os.system('rm ./lmpsL_tmp')
      os.system('rm ./lmpsR_tmp')
      print '\nWeights[l,r,l]=',(wt1,wt2,wt3)
      wt = wt1*wt2*wt3
   return wt

# ---> 
def left_sweep_projection(flmps0,flmps1,qtarget,thresh,Dcut,debug,\
			  ifBlockSingletEmbedding=False,\
			  ifBlockSymScreen=False,\
			  ifpermute=False):
   nsite,ne,ms,sval = qtarget
   print '\n[mpo_dmrg_samps.left_sweep_projection] (nsite,ne,ms,sval)=',(nsite,ne,ms,sval),\
         'ifBlockSE=',ifBlockSingletEmbedding	   
   flmps1['nsite'] = nsite
   # Reduced qnums - the basis states are ordered to maximize the efficiency.
   qnumsN = numpy.array([[0.,0.,1.],[1.,0.5,1.],[2.,0.,1.]])
   qnums1 = [None]*(nsite+1)
   
   # Do not use singlet embedding for the first purification sweep,
   # otherwise, it will mess up with other spins.
   if ifBlockSingletEmbedding == False:
      qnums1[0] = numpy.array([[0.,0.,1.]])
      wmat = numpy.identity(1)
      # Quantum numbers
      ne_eff = ne
      sval_eff = sval
      ms_eff = ms
   else:
      qnums1[0] = numpy.array([[2*sval,sval,1.]]) # (N,S)=(2*S,S)
      ndeg = int(2*sval+1)
      wmat = numpy.zeros((ndeg,1))
      # Coupled to a noninteracting state |S(-M)>
      # to create a broken symetry state |S(-M)>*|SM>
      # which has the singlet component |00>/sqrt(2*S+1).
      im = int(-ms+sval)
      wmat[im] = 1.0
      # Quantum numbers
      ne_eff = ne + 2.0*sval
      sval_eff = 0.0
      ms_eff = 0.0

   # Start conversion for the following sites
   for isite in range(nsite):

      site0 = mpo_dmrg_io.loadSite(flmps0,isite,False)
  
      # Expand |M> basis to |SM> basis
      wA0 = numpy.tensordot(wmat,site0,axes=([1],[0])) # Ll,lNr->LNr

      # Transform to combined basis
      if ifpermute and ifBlockSingletEmbedding and isite==0:
         wA0 = wA0.transpose(1,0,2)
	 sgn = numpy.array([1.,(-1.0)**(int(2*sval)),(-1.0)**(int(2*sval)),1.])
	 wA0 = numpy.einsum('n,nvr->nvr',sgn,wA0)
         qnumsL,wA1 = util_tensor.spinCouple(wA0,qnumsN,qnums1[isite])
      else:
         qnumsL,wA1 = util_tensor.spinCouple(wA0,qnums1[isite],qnumsN)

      # Quasi-RDM (reduced): each dim is {[(SaSb)S,na*nb]}
      classes,qrdmL = util_tensor.quasiRDM(wA1,qnumsL)
      if debug: print 'trace=',numpy.trace(qrdmL),qrdmL.shape,len(classes)
      
      # Decimation: [(SaSb)S,na*nb]=>[S,r] for all possible Sa,Sb
      # Therefore, rotL is a matrix with dimensions [(SaSb)S,na*nb]*[S,r]!
      if ifBlockSingletEmbedding and isite==0:
         dwts,qred,rotL,sigs = mpo_dmrg_qparser.rdm_blkdiag(qrdmL,classes,-1.e-10,-1,debug)
      else:
         dwts,qred,rotL,sigs = mpo_dmrg_qparser.rdm_blkdiag(qrdmL,classes,thresh,Dcut,debug)
      
      #
      #        |	(0) w[i-1] = <l[i-1]|m[i-1]>: expansion coeff of |l(NM)> to |m(NSM)>
      #  |-----|------| (1) update formula for w[i]
      #	 |     L      | (2) L[i] - isometry 
      #  |     |      | (3) L[i]*t[CG] - new site A[i]
      #  |   t[CG]    |
      #  |    / \     |	   |
      #  |   w---A[i]---A[i+1]---
      #  |------------|
      #

      # Screen
      qnumsl,rotL = util_spinsym.symScreen(isite,nsite,rotL,qred,sigs,ne_eff,sval_eff,debug,\
		      			   ifBlockSymScreen=ifBlockSymScreen)
      if ifpermute and ifBlockSingletEmbedding and isite==0:
         # A[lnr]
         site1 = util_tensor.expandRotL(rotL,qnumsN,qnums1[isite],qnumsL,qnumsl) 
         # srotR = rotL^\dagger * wA
         srotR = numpy.tensordot(site1,wA0,axes=([0,1],[0,1])) # LNR,LNr->Rr
	 site1 = site1.transpose(1,0,2)
      else:
         # A[lnr]
         site1 = util_tensor.expandRotL(rotL,qnums1[isite],qnumsN,qnumsL,qnumsl) 
         # srotR = rotL^\dagger * wA
         srotR = numpy.tensordot(site1,wA0,axes=([0,1],[0,1])) # LNR,LNr->Rr

      # Print: 
      print ' ---> isite=',isite,'site0=',site0.shape,'rotL=',rotL.shape,\
	 	                 'site1=',site1.shape #,'srotR=',srotR.shape
      print 		 
      # Check left canonical form
      if debug:
         tmp = numpy.tensordot(site1,site1,axes=([0,1],[0,1])) # lna,lnb->ab
         d = tmp.shape[0]
         diff = numpy.linalg.norm(tmp-numpy.identity(d))
         print ' diff=',diff
         assert diff < 1.e-10

      # Save
      if isite < nsite-1:
         wmat = srotR.copy() 
	 qnums1[isite+1] = qnumsl.copy()
         flmps1.create_dataset('rotL'+str(isite),data=rotL ,compression='lzf')
         flmps1.create_dataset('site'+str(isite),data=site1,compression='lzf')
      else:
         # Special treatment for the last site by invoking symmetry selection!
	 info,qnumsl,rotL,site1,wt = util_tensor.lastSite(rotL,site1,srotR,qnumsl,ne_eff,ms_eff,sval_eff)
	 if info:
	    qnums1[nsite] = qnumsl.copy()
            flmps1.create_dataset('rotL'+str(isite),data=rotL ,compression='lzf')
            flmps1.create_dataset('site'+str(isite),data=site1,compression='lzf')

   # Dump on flmps1
   if info:
      print '\nfinalize left_sweep ...'
      qnumsm = [None]*(nsite+1)	 
      for isite in range(nsite):
	 qnumsm[isite] = util_spinsym.expandSM(qnums1[isite])
      # Left case
      qnumsm[nsite] = numpy.array([[ne,ms]])
      # DUMP qnums
      for isite in range(nsite+1):
	 flmps1['qnum'+str(isite)] = qnumsm[isite]
	 flmps1['qnumNS'+str(isite)] = qnums1[isite]
      # Check
      for isite in range(nsite+1):
         dimr = util_spinsym.dim_red(qnums1[isite]) 
	 print ' ibond/dim(NS)/dim(NSM)=',isite,dimr,len(qnumsm[isite])
   return wt

# <--- 
def right_sweep_projection(flmps0,flmps1,qtarget,thresh,Dcut,debug,\
			   ifBlockSingletEmbedding=False):
   nsite,ne,ms,sval = qtarget
   print '\n[mpo_dmrg_samps.right_sweep_projection] (nsite,,ne,ms,sval)=',(nsite,ne,ms,sval)
   flmps1['nsite'] = nsite
   # Reduced qnums - the basis states are ordered to maximize the efficiency.
   qnumsN = numpy.array([[0.,0.,1.],[1.,0.5,1.],[2.,0.,1.]])
   qnums1 = [None]*(nsite+1)

   # This works perfectly. 
   qnums1[nsite] = numpy.array([[0.,0.,1.]])
   wmat = numpy.identity(1)

   for isite in range(nsite-1,-1,-1):
      site0 = mpo_dmrg_io.loadSite(flmps0,isite,False)
 
      # *** Change the direction for combinations
      site0 = numpy.einsum('lnr->rnl',site0)
      
      # Expand |M> basis to |SM> basis
      wA0 = numpy.tensordot(wmat,site0,axes=([1],[0])) # Ll,lNr->LNr

      # Transform to combined basis
      qnumsL,wA1 = util_tensor.spinCouple(wA0,qnums1[isite+1],qnumsN)

      # Quasi-RDM (reduced): each dim is {[(SaSb)S,na*nb]}
      classes,qrdmL = util_tensor.quasiRDM(wA1,qnumsL)
      if debug: print 'trace=',numpy.trace(qrdmL),qrdmL.shape,len(classes)
      
      # Decimation: [(SaSb)S,na*nb]=>[S,r] for all possible Sa,Sb
      # Therefore, rotL is a matrix with dimensions [(SaSb)S,na*nb]*[S,r]!
      dwts,qred,rotL,sigs = mpo_dmrg_qparser.rdm_blkdiag(qrdmL,classes,thresh,Dcut,debug) 
      
      #
      #        |	(0) w[i-1] = <l[i-1]|m[i-1]>: expansion coeff of |l(NM)> to |m(NSM)>
      #  |-----|------| (1) update formula for w[i]
      #	 |     L      | (2) L[i] - isometry 
      #  |     |      | (3) L[i]*t[CG] - new site A[i]
      #  |   t[CG]    |
      #  |    / \     |	   |
      #  |   w---A[i]---A[i+1]---
      #  |------------|
      #

      # Screen
      qnumsl,rotL = util_spinsym.symScreen(isite,nsite,rotL,qred,sigs,ne,sval,debug,status='R')
      # A[lnr]
      site1 = util_tensor.expandRotL(rotL,qnums1[isite+1],qnumsN,qnumsL,qnumsl) 
      # srotR = rotL^\dagger * wA
      srotR = numpy.tensordot(site1,wA0,axes=([0,1],[0,1])) # LNR,LNr->Rr

      # Print: 
      print ' ---> isite=',isite,'site0=',site0.shape,'rotL=',rotL.shape,\
	 	                 'site1=',site1.shape #,'srotR=',srotR.shape
      print 		 
      # Check left canonical form
      if debug:
         tmp = numpy.tensordot(site1,site1,axes=([0,1],[0,1])) # lna,lnb->ab
         d = tmp.shape[0]
         diff = numpy.linalg.norm(tmp-numpy.identity(d))
         print ' diff=',diff
         assert diff < 1.e-10

      # Save
      if isite > 0:
         wmat = srotR.copy() 
	 qnums1[isite] = qnumsl.copy()
         flmps1.create_dataset('rotL'+str(isite),data=rotL ,compression='lzf')
         # *** Change the direction back [Right canonical form]
         site1 = numpy.einsum('rnl->lnr',site1)
         flmps1.create_dataset('site'+str(isite),data=site1,compression='lzf')
      else:
	 # In case of singlet embedding, the in bond of the first site
	 # is expanded in the subroutine expandRotL in the left sweep for nonsinglet state.
	 if ifBlockSingletEmbedding and abs(sval)>1.e-10:
	    im = int(ms+sval)
	    srotR = srotR[:,im].reshape(srotR.shape[0],1)
         # Special treatment for the last site by invoking symmetry selection!
	 info,qnumsl,rotL,site1,wt = util_tensor.lastSite(rotL,site1,srotR,qnumsl,ne,ms,sval)
	 if info:
	    qnums1[0] = qnumsl.copy()
            flmps1.create_dataset('rotL'+str(isite),data=rotL ,compression='lzf')
            site1 = numpy.einsum('rnl->lnr',site1)
            flmps1.create_dataset('site'+str(isite),data=site1,compression='lzf')

   # Dump on flmps1
   if info:
      print '\nfinalize right_sweep ...'
      qnumsm = [None]*(nsite+1)	 
      for isite in range(1,nsite+1):
	 qnumsm[isite] = util_spinsym.expandSM(qnums1[isite])
      # Right case
      qnumsm[0] = numpy.array([[ne,ms]])
      # DUMP qnums
      for isite in range(nsite+1):
	 flmps1['qnum'+str(isite)] = qnumsm[isite]
	 flmps1['qnumNS'+str(isite)] = qnums1[isite]
      # Check
      for isite in range(nsite+1):
         dimr = util_spinsym.dim_red(qnums1[isite]) 
	 print ' ibond/dim(NS)/dim(NSM)=',isite,dimr,len(qnumsm[isite])
   return wt
