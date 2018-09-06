#!/usr/bin/env python
#
# Generation of <Eii>: Qt is not implemented yet!
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
#
# def diag(dmrg,fbmps,debug=False):
# def diagNQt(dmrg,fbmps,fname,debug=False):
#
import h5py
import time
import numpy
from zmpo_dmrg.source import mpo_dmrg_io
from zmpo_dmrg.source import mpo_dmrg_init
from zmpo_dmrg.source import mpo_dmrg_opers

def diag(dmrg,fbmps,debug=False):
   print '\n[mpo_dmrg_rdm.diag]'
   if not dmrg.ifs2proj:
      fname = dmrg.path+'/rdm_diag'
      mpo_dmrg_init.genSops(dmrg,fbmps,fbmps,fname+'L','L',debug)
      mpo_dmrg_init.genSops(dmrg,fbmps,fbmps,fname+'R','R',debug)
      if dmrg.ifQt:
         print 'error'
         exit()
         #nii = diagQt(dmrg,fbmps,fname,debug)
      else:   
         nii = diagNQt(dmrg,fbmps,fname,debug)
   else:
      print 'error'
      exit()
      #mpo_dmrg_init.genPops(dmrg,fbmps,fbmps,fname,'R',debug)
   return nii

def diagNQt(dmrg,fbmps,fname,debug=False):
   t0 = time.time()
   nsite = dmrg.nsite
   prefixL = fname+'L_site_'
   prefixR = fname+'R_site_'
   # L->R sweeps 
   mpo_dmrg_init.genBmat(dmrg,fname,-1)
   nii = numpy.zeros(nsite)
   for isite in range(0,nsite):
      fL = h5py.File(prefixL+str(isite-1),"r")
      fR = h5py.File(prefixR+str(isite+1),"r")
      bsite = mpo_dmrg_io.loadSite(fbmps,isite,dmrg.ifQt)
      tmpl = fL['mat'].value
      tmpr = fR['mat'].value
      npmat = mpo_dmrg_opers.genNpMat()
      #
      #   i---*---j
      #  /    |    \
      # * L   *m    * R
      #  \    |    /
      #   a---*---b
      #	
      tmp = numpy.einsum('mn,lnr->lmr',npmat,bsite)
      tmp = numpy.einsum('Ll,lmr->Lmr',tmpl,tmp)
      tmp = numpy.einsum('lmr,rR->lmR',tmp,tmpr)
      nii[isite] = numpy.tensordot(tmp,bsite,axes=([0,1,2],[0,1,2]))
      fL.close()
      fR.close()
   # Final
   print ' sum of nii =',numpy.sum(nii)
   print ' nii =',nii
   t1=time.time()
   print ' time for diagNQt = %.2f s'%(t1-t0),' rank =',dmrg.comm.rank
   return nii
