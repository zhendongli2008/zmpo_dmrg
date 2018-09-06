#!/usr/bin/env python
#
# Blocking for bare operators in order to compute RDMs: NQt
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
#
# def genCAops(norder,dmrg,fbmps,fkmps,fname,status,debug=False):
# def genCAopsNQt(norder,dmrg,fbmps,fkmps,fname,status,debug=False):
# #> def genRDMs(norder,dmrg,fbmps,fkmps,fname,debug=False):
# #> def genRDMsNQt(norder,dmrg,fbmps,fkmps,debug=False):
# def renorm_l1(bsite,ksite,f0,f1,oplst=None):
# def renorm_r1(bsite,ksite,f0,f1,oplst=None):
# def renorm_l2_absorb_lop(bsite,lop):
# def renorm_l2_transform_nop(blop,nop,ksite):
# def renorm_r2_absorb_rop(ksite,rop):
# def renorm_r2_transform_nop(krop,nop,bsite):
# 
import os
import h5py
import time
import numpy
from zmpo_dmrg.source import mpo_dmrg_io
from zmpo_dmrg.source import mpo_dmrg_init
from zmpo_dmrg.source import mpo_dmrg_opers

# Creation and Annihilation operators: {c+,a}, {c+*a+,c+*c+,a*a}
def genCAops(norder,dmrg,fbmps,fkmps,fname,status,debug=False):
   if dmrg.comm.rank == 0: print '\n[mpo_dmrg_block.genCAops] ifQt=',dmrg.ifQt
   if dmrg.ifQt:
      print 'Not implemented yet!'
      exit()	   
      #exphop = mpo_dmrg_blockQt.genCAopsQt(norder,dmrg,fbmps,fkmps,fname,status,debug)
   else:
      exphop = genCAopsNQt(norder,dmrg,fbmps,fkmps,fname,status,debug)
   return exphop

# Creation and Annihilation operators: {c+,a}, {c+*a+,c+*c+,a*a}
def genCAopsNQt(norder,dmrg,fbmps,fkmps,fname,status,debug=False):
   if dmrg.comm.rank == 0: 
      print '[mpo_dmrg_block.genCAopsNQt] status=',status
      print ' fname = ',fname
   t0 = time.time()
   bnsite = fbmps['nsite'].value
   knsite = fkmps['nsite'].value
   assert bnsite == knsite
   nsite  = bnsite
   sbas   = 2*nsite
   prefix = fname+'_site_'
   #
   # On-site operators
   #
   sgnn = numpy.array([1.,-1.,-1.,1.])
   cre = [0]*2
   cre[0] = mpo_dmrg_opers.genElemSpatialMat(0,0,1)
   cre[1] = mpo_dmrg_opers.genElemSpatialMat(1,0,1)
   ann = [0]*2
   ann[0] = mpo_dmrg_opers.genElemSpatialMat(0,0,0)
   ann[1] = mpo_dmrg_opers.genElemSpatialMat(1,0,0)
   cc2 = cre[0].dot(cre[1])
   aa2 = ann[0].dot(ann[1])
   ca2 = [cre[i].dot(ann[j]) for i in range(2) for j in range(2)]
   ac2 = [ann[i].dot(cre[j]) for i in range(2) for j in range(2)]
   #
   # L->R sweeps 
   #
   if status == 'L':

      mpo_dmrg_init.genBmat(dmrg,fname,-1)
      for isite in range(0,nsite):
	 if debug: print ' isite=',isite,' of nsite=',nsite
	 ti = time.time()
	 f0 = h5py.File(prefix+str(isite-1),"r")
         f1name = prefix+str(isite)
	 f1 = h5py.File(f1name,"w")
	 bsite = mpo_dmrg_io.loadSite(fbmps,isite,dmrg.ifQt)
	 ksite = mpo_dmrg_io.loadSite(fkmps,isite,dmrg.ifQt)
	 
	 # 0. Qnums
	 porbs = 2*isite
	 nqum = fkmps['qnum'+str(isite)].value[:,0]
	 sgnl = numpy.power(-1.0,nqum)
	 ksite2 = numpy.einsum('l,lnr->lnr',sgnl,ksite)
	 
	 # 1. Similar to mpo_dmrg_init.genSopsNQt: <li[A]|lj[B]>
	 oplst = ['mat']
	 renorm_l1(bsite,ksite,f0,f1,oplst)

	 # 2. Part-1: <l'n'|op|ln> = <l'|op|l>*<n'|n>  if p < 2k
	 oplst = ['op_C_'+str(p) for p in range(porbs)]\
	       + ['op_A_'+str(p) for p in range(porbs)]
	 renorm_l1(bsite,ksite,f0,f1,oplst)
	 #   Part-2: <l'n'|op|ln> = <l'|l>(-1)^l*<n'|op|n> if p = 2k,2k+1 
	 #   	     <L'|op|L> = A[l'n',L']*<l'|l><n'|op|n>*A[ln,L]
	 tmp0 = renorm_l2_absorb_lop(bsite,f0['mat'])
	 for ispin in range(2):
	    tmp = renorm_l2_transform_nop(tmp0,cre[ispin],ksite2)
	    f1['op_C_'+str(2*isite+ispin)] = tmp 
	    tmp = renorm_l2_transform_nop(tmp0,ann[ispin],ksite2)
	    f1['op_A_'+str(2*isite+ispin)] = tmp
	 
	 # 3. Part-1: <l'n'|aa(p<q)|ln> = <l'|apq|l><n'|n>  if p,q < 2k
	 oplst = ['op_CC_'+str(p)+'_'+str(q) for p in range(porbs) for q in range(p+1,porbs)]\
	       + ['op_AA_'+str(p)+'_'+str(q) for p in range(porbs) for q in range(p+1,porbs)]
	 renorm_l1(bsite,ksite,f0,f1,oplst)
	 #    Part-2: <l'|ap|l>*<n'|aq|n> if p < 2k and q in 2k,2k+1
	 for p in range(porbs):
	    # CC
	    tmp0 = renorm_l2_absorb_lop(bsite,f0['op_C_'+str(p)])
	    for ispin in range(2):
	       tmp = renorm_l2_transform_nop(tmp0,cre[ispin],ksite2)
	       f1['op_CC_'+str(p)+'_'+str(2*isite+ispin)] = tmp
	    # AA
	    tmp0 = renorm_l2_absorb_lop(bsite,f0['op_A_'+str(p)])
	    for ispin in range(2):
	       tmp = renorm_l2_transform_nop(tmp0,ann[ispin],ksite2)
	       f1['op_AA_'+str(p)+'_'+str(2*isite+ispin)] = tmp
	 #    Part-3: <l'|l>*<n'|apq|n> if p,q in 2k,2k+1
	 #   	      <L'|opq|L> = A[l'n',L']*<l'|l><n'|op|n>*A[ln,L]
	 tmp0 = renorm_l2_absorb_lop(bsite,f0['mat'])
	 tmp  = renorm_l2_transform_nop(tmp0,cc2,ksite)
         f1['op_CC_'+str(2*isite+0)+'_'+str(2*isite+1)] = tmp
	 tmp  = renorm_l2_transform_nop(tmp0,aa2,ksite)
         f1['op_AA_'+str(2*isite+0)+'_'+str(2*isite+1)] = tmp

	 # 4. Part-1: <l'n'|cp*aq|ln> = <l'|cp*aq|l><n'|n>  if p,q < 2k
	 oplst = ['op_CA_'+str(p)+'_'+str(q) for p in range(porbs) for q in range(porbs)]\
	       + ['op_AC_'+str(p)+'_'+str(q) for p in range(porbs) for q in range(porbs)]
	 renorm_l1(bsite,ksite,f0,f1,oplst)
	 #    Part-2: cp[l]*aq[n] or cq[n]*ap[l]=-ap[l]*cq[n]
	 #	      ap[l]*cq[n] or aq[n]*cp[l]=-cp[l]*aq[n]
	 #	      for p < 2*k and q in 2k,2k+1 
	 for p in range(porbs):
	    # CA
	    tmp0 = renorm_l2_absorb_lop(bsite,f0['op_C_'+str(p)])
	    for ispin in range(2):
	       tmp = renorm_l2_transform_nop(tmp0,ann[ispin],ksite2)	    
	       f1['op_CA_'+str(p)+'_'+str(2*isite+ispin)] = tmp 
	       f1['op_AC_'+str(2*isite+ispin)+'_'+str(p)] = -tmp
	    # AC
	    tmp0 = renorm_l2_absorb_lop(bsite,f0['op_A_'+str(p)])
	    for ispin in range(2):
	       tmp = renorm_l2_transform_nop(tmp0,cre[ispin],ksite2)
	       f1['op_AC_'+str(p)+'_'+str(2*isite+ispin)] = tmp
	       f1['op_CA_'+str(2*isite+ispin)+'_'+str(p)] = -tmp
	 #    Part-3: <l'|l>*<n'|cp*aq|n> if p,q in 2k,2k+1
	 tmp0 = renorm_l2_absorb_lop(bsite,f0['mat'])
	 for ispin in range(2):
	    for jspin in range(2): 
	       tmp = renorm_l2_transform_nop(tmp0,ca2[ispin*2+jspin],ksite)
	       f1['op_CA_'+str(2*isite+ispin)+'_'+str(2*isite+jspin)] = tmp
	       tmp = renorm_l2_transform_nop(tmp0,ac2[ispin*2+jspin],ksite)
               f1['op_AC_'+str(2*isite+ispin)+'_'+str(2*isite+jspin)] = tmp

         # final isite
	 f0.close()
	 f1.close()
	 tf = time.time()
	 if dmrg.comm.rank == 0:
	    print ' isite =',os.path.split(f1name)[-1],' t = %.2f s'%(tf-ti)
     
   #
   # L<-R sweeps: For properties, left canonical form is assumed even for R sweeps! 
   #
   elif status == 'R':

      mpo_dmrg_init.genBmat(dmrg,fname,nsite)
      for isite in range(nsite-1,-1,-1):
	 if debug: print ' isite=',isite,' of nsite=',nsite
	 ti = time.time()
	 f0 = h5py.File(prefix+str(isite+1),"r")
         f1name = prefix+str(isite)
         f1 = h5py.File(f1name,"w")
	 bsite = mpo_dmrg_io.loadSite(fbmps,isite,dmrg.ifQt)
	 ksite = mpo_dmrg_io.loadSite(fkmps,isite,dmrg.ifQt)
	
	 # 0. Qnums
	 porbs = 2*(isite+1)
	 ksite2 = numpy.einsum('n,lnr->lnr',sgnn,ksite)
       
	 # 1. <ri[A]|rj[B]>
	 oplst = ['mat']
	 renorm_r1(bsite,ksite,f0,f1,oplst)
	
	 # 2. Part-1: <n'r'|op|nr> = <n'|n>(-1)^n*<r'|op|r> if p > 2k
	 # 	      A[l',n',r']<n'|n>(-1)^n*<r'|op|r>*A[l,n,r]
	 oplst = ['op_C_'+str(p) for p in range(porbs,sbas)]\
	       + ['op_A_'+str(p) for p in range(porbs,sbas)]
	 renorm_r1(bsite,ksite2,f0,f1,oplst)
	 #    Part-2: <n'r'|op|nr> = <n'|op|n>*<r'|r> if p = 2k,2k+1
	 #	      <l'|op|l> = A[l',n'r']*<n'|op|n>*<r'|r>*A[l,nr]
	 tmp0 = renorm_r2_absorb_rop(ksite,f0['mat'])
	 for ispin in range(2):
	    tmp = renorm_r2_transform_nop(tmp0,cre[ispin],bsite)
	    f1['op_C_'+str(2*isite+ispin)] = tmp
	    tmp = renorm_r2_transform_nop(tmp0,ann[ispin],bsite)
	    f1['op_A_'+str(2*isite+ispin)] = tmp
	 
	 # 3. Part-1: <n'r'|aa(p<q)|n'r'> = <n'|n><r'|apq|r> if p,q > 2k
	 #	      <l'|aa(p<q)|l> = A[l',n',r']<n'|n><r'|apq|r>A[l,n,r]
	 oplst = ['op_CC_'+str(p)+'_'+str(q) for p in range(porbs,sbas) for q in range(p+1,porbs)]\
	       + ['op_AA_'+str(p)+'_'+str(q) for p in range(porbs,sbas) for q in range(p+1,porbs)]
	 renorm_r1(bsite,ksite,f0,f1,oplst)
	 #    Part-2: <n'r'|aa(p<q)|n'r'> = <n'|ap|n>*S*<r'|aq|r> if p in 2k,2k+1 and q > 2k
	 #	      <l'|aa(p<q)|l> = A[l',n',r']*<n'|ap|n>S*<r'|aq|r>*A[l,n,r]	
	 for p in range(porbs,sbas):
	    # CC
	    tmp0 = renorm_r2_absorb_rop(ksite2,f0['op_C_'+str(p)])
	    for ispin in range(2):
               tmp = renorm_r2_transform_nop(tmp0,cre[ispin],bsite)
	       f1['op_CC_'+str(2*isite+ispin)+'_'+str(p)] = tmp
	    # AA
	    tmp0 = renorm_r2_absorb_rop(ksite2,f0['op_A_'+str(p)])
	    for ispin in range(2):
               tmp = renorm_r2_transform_nop(tmp0,ann[ispin],bsite)
	       f1['op_AA_'+str(2*isite+ispin)+'_'+str(p)] = tmp
	 #    Part-3: <n'r'|aa(p<q)|n'r'> = <n'|apq|n>*<r'|r> if p,q in 2k,2k+1 
	 #	      <l'|aa(p<q)|l> = A[l',n',r]*<n'|apq|n>*<r'|r>*A[l,n,r]
	 tmp0 = renorm_r2_absorb_rop(ksite,f0['mat'])
	 tmp  = renorm_r2_transform_nop(tmp0,cc2,bsite)
	 f1['op_CC_'+str(2*isite+0)+'_'+str(2*isite+1)] = tmp
	 tmp  = renorm_r2_transform_nop(tmp0,aa2,bsite)
	 f1['op_AA_'+str(2*isite+0)+'_'+str(2*isite+1)] = tmp
	 
	 # 4. Part-1: <n'r'|cp*aq|nr> = <n'|n><r'|cp*aq|r> if p,q > 2k
	 #	      <l'|cp*aq|l> = A[l',nr']*<r'|cp*aq|r>*A[l,nr]
	 oplst = ['op_CA_'+str(p)+'_'+str(q) for p in range(porbs,sbas) for q in range(porbs,sbas)]\
	       + ['op_AC_'+str(p)+'_'+str(q) for p in range(porbs,sbas) for q in range(porbs,sbas)]
	 renorm_r1(bsite,ksite,f0,f1,oplst)
	 #    Part-2: <n'r'|cp*aq|nr> = <n'|cp|n>S*<r'|aq|r>
	 for p in range(porbs,sbas):
	    # CA
	    tmp0 = renorm_r2_absorb_rop(ksite2,f0['op_A_'+str(p)])
	    for ispin in range(2):
	       tmp  = renorm_r2_transform_nop(tmp0,cre[ispin],bsite)
	       f1['op_CA_'+str(2*isite+ispin)+'_'+str(p)] = tmp
	       f1['op_AC_'+str(p)+'_'+str(2*isite+ispin)] = -tmp
	    # AA
	    tmp0 = renorm_r2_absorb_rop(ksite2,f0['op_C_'+str(p)])
	    for ispin in range(2):
               tmp  = renorm_r2_transform_nop(tmp0,ann[ispin],bsite)
	       f1['op_AC_'+str(2*isite+ispin)+'_'+str(p)] = tmp
	       f1['op_CA_'+str(p)+'_'+str(2*isite+ispin)] = -tmp
	 #    Part-3: <n'r'|cp*aq|n'r'> = <n'|cp*aq|n>*<r'|r> if p,q in 2k,2k+1 
         tmp0 = renorm_r2_absorb_rop(ksite,f0['mat'])
	 for ispin in range(2):
	    for jspin in range(2): 
	       tmp = renorm_r2_transform_nop(tmp0,ca2[ispin*2+jspin],bsite)
	       f1['op_CA_'+str(2*isite+ispin)+'_'+str(2*isite+jspin)] = tmp
	       tmp = renorm_r2_transform_nop(tmp0,ac2[ispin*2+jspin],bsite)
	       f1['op_AC_'+str(2*isite+ispin)+'_'+str(2*isite+jspin)] = tmp

	 # final isite
	 f0.close()
	 f1.close()
	 tf = time.time()
	 if dmrg.comm.rank == 0:
	    print ' isite =',os.path.split(f1name)[-1],' t = %.2f s'%(tf-ti)
  
   # CHECK
   f = h5py.File(f1name,'r')
   rdm1 = numpy.zeros((sbas,sbas))
   hdm1 = numpy.zeros((sbas,sbas))
   for i in range(sbas):
      for j in range(sbas):
         rdm1[i,j] = f['op_CA_'+str(i)+'_'+str(j)].value
         hdm1[j,i] = f['op_AC_'+str(i)+'_'+str(j)].value
   sab = f['mat'].value[0,0]
   #print rdm1+hdm1
   print ' ovlap=',sab
   print ' P+H-I=',numpy.linalg.norm(rdm1+hdm1-numpy.identity(sbas)*sab) 
   print ' skewP=',numpy.linalg.norm(rdm1-rdm1.T)
   print ' skewH=',numpy.linalg.norm(hdm1-hdm1.T)
   print ' trace=',numpy.trace(rdm1)
   f.close()

   t1=time.time()
   dmrg.comm.Barrier()
   print ' time for genHops = %.2f s'%(t1-t0),' rank =',dmrg.comm.rank
   return rdm1

#> # RDMs
#> def genRDMs(norder,dmrg,fbmps,fkmps,fname,debug=False):
#>    if dmrg.comm.rank == 0: print '\n[mpo_dmrg_block.genRDMs] ifQt=',dmrg.ifQt
#>    # Generate environment operators
#>    fname = 'tmpRDM_envR'
#>    genCAops(norder,dmrg,fbmps,fkmps,fname,'R')
#>    # Start to construct RDMs
#>    if dmrg.ifQt:
#>       exphop = mpo_dmrg_blockQt.genRDMsQt(norder,dmrg,fbmps,fkmps,debug)
#>    else:
#>       exphop = genRDMsNQt(norder,dmrg,fbmps,fkmps,debug)
#>    return exphop
#> 
#> # RDMs
#> def genRDMsNQt(norder,dmrg,fbmps,fkmps,debug=False):
#>    if dmrg.comm.rank == 0: 
#>       print '[mpo_dmrg_block.genRDMsNQt] norder =',norder
#>    t0 = time.time()
#>    bnsite = fbmps['nsite'].value
#>    knsite = fkmps['nsite'].value
#>    assert bnsite == knsite
#>    nsite  = bnsite
#>    sbas   = 2*nsite
#>    prefix = 'tmpRDM_site_'
#>    exit()
#>    #
#>    # On-site operators
#>    #
#>    sgnn = numpy.array([1.,-1.,-1.,1.])
#>    cre = [0]*2
#>    cre[0] = mpo_dmrg_opers.genElemSpatialMat(0,0,1)
#>    cre[1] = mpo_dmrg_opers.genElemSpatialMat(1,0,1)
#>    ann = [0]*2
#>    ann[0] = mpo_dmrg_opers.genElemSpatialMat(0,0,0)
#>    ann[1] = mpo_dmrg_opers.genElemSpatialMat(1,0,0)
#>    cc2 = cre[0].dot(cre[1])
#>    aa2 = ann[0].dot(ann[1])
#>    ca2 = [cre[i].dot(ann[j]) for i in range(2) for j in range(2)]
#>    ac2 = [ann[i].dot(cre[j]) for i in range(2) for j in range(2)]
#>    #
#>    # L->R sweeps 
#>    #
#>    if status == 'L':
#> 
#>       mpo_dmrg_init.genBmat(dmrg,fname,-1)
#>       for isite in range(0,nsite):
#> 	 if debug: print ' isite=',isite,' of nsite=',nsite
#> 	 ti = time.time()
#> 	 f0 = h5py.File(prefix+str(isite-1),"r")
#>          f1name = prefix+str(isite)
#> 	 f1 = h5py.File(f1name,"w")
#> 	 bsite = mpo_dmrg_io.loadSite(fbmps,isite,dmrg.ifQt)
#> 	 ksite = mpo_dmrg_io.loadSite(fkmps,isite,dmrg.ifQt)
#> 	 
#> 	 # 0. Qnums
#> 	 porbs = 2*isite
#> 	 nqum = fkmps['qnum'+str(isite)].value[:,0]
#> 	 sgnl = numpy.power(-1.0,nqum)
#> 	 ksite2 = numpy.einsum('l,lnr->lnr',sgnl,ksite)
#> 	 
#> 	 # 1. Similar to mpo_dmrg_init.genSopsNQt: <li[A]|lj[B]>
#> 	 oplst = ['mat']
#> 	 renorm_l1(bsite,ksite,f0,f1,oplst)
#> 
#> 	 # 2. Part-1: <l'n'|op|ln> = <l'|op|l>*<n'|n>  if p < 2k
#> 	 oplst = ['op_C_'+str(p) for p in range(porbs)]\
#> 	       + ['op_A_'+str(p) for p in range(porbs)]
#> 	 renorm_l1(bsite,ksite,f0,f1,oplst)
#> 	 #   Part-2: <l'n'|op|ln> = <l'|l>(-1)^l*<n'|op|n> if p = 2k,2k+1 
#> 	 #   	     <L'|op|L> = A[l'n',L']*<l'|l><n'|op|n>*A[ln,L]
#> 	 tmp0 = renorm_l2_absorb_lop(bsite,f0['mat'])
#> 	 for ispin in range(2):
#> 	    tmp = renorm_l2_transform_nop(tmp0,cre[ispin],ksite2)
#> 	    f1['op_C_'+str(2*isite+ispin)] = tmp 
#> 	    tmp = renorm_l2_transform_nop(tmp0,ann[ispin],ksite2)
#> 	    f1['op_A_'+str(2*isite+ispin)] = tmp
#> 	 
#> 	 # 3. Part-1: <l'n'|aa(p<q)|ln> = <l'|apq|l><n'|n>  if p,q < 2k
#> 	 oplst = ['op_CC_'+str(p)+'_'+str(q) for p in range(porbs) for q in range(p+1,porbs)]\
#> 	       + ['op_AA_'+str(p)+'_'+str(q) for p in range(porbs) for q in range(p+1,porbs)]
#> 	 renorm_l1(bsite,ksite,f0,f1,oplst)
#> 	 #    Part-2: <l'|ap|l>*<n'|aq|n> if p < 2k and q in 2k,2k+1
#> 	 for p in range(porbs):
#> 	    # CC
#> 	    tmp0 = renorm_l2_absorb_lop(bsite,f0['op_C_'+str(p)])
#> 	    for ispin in range(2):
#> 	       tmp = renorm_l2_transform_nop(tmp0,cre[ispin],ksite2)
#> 	       f1['op_CC_'+str(p)+'_'+str(2*isite+ispin)] = tmp
#> 	    # AA
#> 	    tmp0 = renorm_l2_absorb_lop(bsite,f0['op_A_'+str(p)])
#> 	    for ispin in range(2):
#> 	       tmp = renorm_l2_transform_nop(tmp0,ann[ispin],ksite2)
#> 	       f1['op_AA_'+str(p)+'_'+str(2*isite+ispin)] = tmp
#> 	 #    Part-3: <l'|l>*<n'|apq|n> if p,q in 2k,2k+1
#> 	 #   	      <L'|opq|L> = A[l'n',L']*<l'|l><n'|op|n>*A[ln,L]
#> 	 tmp0 = renorm_l2_absorb_lop(bsite,f0['mat'])
#> 	 tmp  = renorm_l2_transform_nop(tmp0,cc2,ksite)
#>          f1['op_CC_'+str(2*isite+0)+'_'+str(2*isite+1)] = tmp
#> 	 tmp  = renorm_l2_transform_nop(tmp0,aa2,ksite)
#>          f1['op_AA_'+str(2*isite+0)+'_'+str(2*isite+1)] = tmp
#> 
#> 	 # 4. Part-1: <l'n'|cp*aq|ln> = <l'|cp*aq|l><n'|n>  if p,q < 2k
#> 	 oplst = ['op_CA_'+str(p)+'_'+str(q) for p in range(porbs) for q in range(porbs)]\
#> 	       + ['op_AC_'+str(p)+'_'+str(q) for p in range(porbs) for q in range(porbs)]
#> 	 renorm_l1(bsite,ksite,f0,f1,oplst)
#> 	 #    Part-2: cp[l]*aq[n] or cq[n]*ap[l]=-ap[l]*cq[n]
#> 	 #	      ap[l]*cq[n] or aq[n]*cp[l]=-cp[l]*aq[n]
#> 	 #	      for p < 2*k and q in 2k,2k+1 
#> 	 for p in range(porbs):
#> 	    # CA
#> 	    tmp0 = renorm_l2_absorb_lop(bsite,f0['op_C_'+str(p)])
#> 	    for ispin in range(2):
#> 	       tmp = renorm_l2_transform_nop(tmp0,ann[ispin],ksite2)	    
#> 	       f1['op_CA_'+str(p)+'_'+str(2*isite+ispin)] = tmp 
#> 	       f1['op_AC_'+str(2*isite+ispin)+'_'+str(p)] = -tmp
#> 	    # AC
#> 	    tmp0 = renorm_l2_absorb_lop(bsite,f0['op_A_'+str(p)])
#> 	    for ispin in range(2):
#> 	       tmp = renorm_l2_transform_nop(tmp0,cre[ispin],ksite2)
#> 	       f1['op_AC_'+str(p)+'_'+str(2*isite+ispin)] = tmp
#> 	       f1['op_CA_'+str(2*isite+ispin)+'_'+str(p)] = -tmp
#> 	 #    Part-3: <l'|l>*<n'|cp*aq|n> if p,q in 2k,2k+1
#> 	 tmp0 = renorm_l2_absorb_lop(bsite,f0['mat'])
#> 	 for ispin in range(2):
#> 	    for jspin in range(2): 
#> 	       tmp = renorm_l2_transform_nop(tmp0,ca2[ispin*2+jspin],ksite)
#> 	       f1['op_CA_'+str(2*isite+ispin)+'_'+str(2*isite+jspin)] = tmp
#> 	       tmp = renorm_l2_transform_nop(tmp0,ac2[ispin*2+jspin],ksite)
#>                f1['op_AC_'+str(2*isite+ispin)+'_'+str(2*isite+jspin)] = tmp
#> 
#>          # final isite
#> 	 f0.close()
#> 	 f1.close()
#> 	 tf = time.time()
#> 	 if dmrg.comm.rank == 0:
#> 	    print ' isite =',os.path.split(f1name)[-1],' t = %.2f s'%(tf-ti)
#>      
#>    # CHECK
#>    f = h5py.File(f1name,'r')
#>    rdm1 = numpy.zeros((sbas,sbas))
#>    hdm1 = numpy.zeros((sbas,sbas))
#>    for i in range(sbas):
#>       for j in range(sbas):
#>          rdm1[i,j] = f['op_CA_'+str(i)+'_'+str(j)].value
#>          hdm1[j,i] = f['op_AC_'+str(i)+'_'+str(j)].value
#>    sab = f['mat'].value[0,0]
#>    #print rdm1+hdm1
#>    print ' ovlap=',sab
#>    print ' P+H-I=',numpy.linalg.norm(rdm1+hdm1-numpy.identity(sbas)*sab) 
#>    print ' skewP=',numpy.linalg.norm(rdm1-rdm1.T)
#>    print ' skewH=',numpy.linalg.norm(hdm1-hdm1.T)
#>    print ' trace=',numpy.trace(rdm1)
#>    f.close()
#> 
#>    t1=time.time()
#>    dmrg.comm.Barrier()
#>    print ' time for genHops = %.2f s'%(t1-t0),' rank =',dmrg.comm.rank
#>    return rdm1

##############
# SUBROUTINES
##############

# renorm <l'|l>d[n'n]	 
def renorm_l1(bsite,ksite,f0,f1,oplst=None):
   for op in oplst:
      tmp = f0[op].value
      tmp = numpy.tensordot(tmp,bsite,axes=([0],[0])) # (l',l)*(l',n',r')=>(l,n',r')
      tmp = numpy.tensordot(tmp,ksite,axes=([0,1],[0,1])) # (l,n,r')*(l,n,r)
      f1[op] = tmp
   return 0

def renorm_r1(bsite,ksite,f0,f1,oplst=None):
   for op in oplst:
      tmp = f0[op]
      tmp = numpy.tensordot(bsite,tmp,axes=([2],[0])) # r'n'l',l'l=>r'n'l
      tmp = numpy.tensordot(tmp,ksite,axes=([1,2],[1,2])) # r'n'l,d[n'n],rnl=>r'r
      f1[op] = tmp
   return 0

#   Key Contraciton: tmp1[n',L',l] = A[l',n',L']*<l'|l>
#	                tmp2[L',l,n] = tmp1[n',L',l]*<n'|op|n>
#	                <L'|op|L> = tmp2[L',l,n]*A[l,n,L]	
def renorm_l2_absorb_lop(bsite,lop):
   return numpy.tensordot(bsite,lop,axes=([0],[0]))

def renorm_l2_transform_nop(blop,nop,ksite):
   tmp = numpy.tensordot(blop,nop,axes=([0],[0]))
   tmp = numpy.tensordot(tmp,ksite,axes=([1,2],[0,1]))
   return tmp

#   Key Contraciton: tmp1[r',R,n] = <r'|r>*A[R,n,r]
#	                tmp2[n',r',R] = <n'|op|n>*tmp1[r',R,n]
#	                <R'|op|R> = A[R',n,r]*tmp2[n,r,R]
def renorm_r2_absorb_rop(ksite,rop):
   return numpy.tensordot(rop,ksite,axes=([1],[2]))

def renorm_r2_transform_nop(krop,nop,bsite):
   tmp = numpy.tensordot(nop,krop,axes=([1],[2]))
   tmp = numpy.tensordot(bsite,tmp,axes=([1,2],[0,1]))
   return tmp

