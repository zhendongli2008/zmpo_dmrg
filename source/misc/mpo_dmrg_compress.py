#!/usr/bin/env python
#
# Compressions for MPS: Qt version is not implemented yet!
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
#
# def cMPS(fmps0,fmps1,icase=1,thresh=-1,Dcut=-1,debug=False):
# def objectMPS(fmps0,icase,thresh=1.e-12,Dcut=-1):
# 
import numpy
from zmpo_dmrg.source import mpo_dmrg_io
from zmpo_dmrg.source import mpo_dmrg_qphys
from zmpo_dmrg.source.mpsmpo import mps_class

# Load & Compress & Dump of a MPS on file
def cMPS(fmps0,fmps1,icase=1,thresh=-1,Dcut=-1,debug=False):
   nsite = fmps0['nsite'].value
   print '\n[mpo_dmrg_compress.cMPS] icase =',icase,' nsite=',nsite
   # No symmetry - mpslst as a list
   if icase == 0:
      mps = objectMPS(fmps0,icase)
      mps = mps.dcompress()
      mpo_dmrg_io.dumpMPS(fmps1,mps.sites,icase)
   # With qnums and numpy.array
   elif icase == 1:
      mps = objectMPS(fmps0,icase)
      mps = mps.qcompress(thresh=thresh,Dcut=Dcut,debug=True)
      mps.qnums = [[numpy.array([0.0,0.0])]]+mps.qnums
      mpo_dmrg_io.dumpMPS(fmps1,[mps.sites,mps.qnums],icase)
   # With qnums and Qt	   
   elif icase == 2:
      raise NotImplementedError
   return 0

# MPS object
def objectMPS(fmps0,icase,thresh=1.e-12,Dcut=-1):
   mps0 = mpo_dmrg_io.loadMPS(fmps0,icase)
   nsite = fmps0['nsite'].value
   # No symmetry - mpslst as a list
   if icase == 0:
      mps = mps_class.class_mps(nsite,isym=0,sites=mps0,iop=1)
   # With qnums and numpy.array
   elif icase == 1:
      qphys = mpo_dmrg_qphys.initSpatialOrb(nsite,2)
      mps = mps_class.class_mps(nsite,isym=2,sites=mps0[0],iop=1,qphys=qphys,qnums=mps0[1])
   # With qnums and Qt	   
   elif icase == 2:
      mps = mps0 # [sites,qnums]
   return mps
