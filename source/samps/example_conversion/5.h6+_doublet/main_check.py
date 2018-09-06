import h5py
import shutil
import numpy
from mpi4py import MPI
from zmpo_dmrg.source.itools.molinfo import class_molinfo
from zmpo_dmrg.source import mpo_dmrg_class
from zmpo_dmrg.source import mpo_dmrg_schedule

#==================================
# Main program
#==================================
comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank
# MPI init
if size > 0 and rank ==0: print '\n[MPI init]'
comm.Barrier()
print ' Rank= %s of %s processes'%(rank,size)

mol=class_molinfo()
mol.comm=comm
fname = "mole.h5"
mol.loadHam(fname)
mol.isym =0 #2 #WhetherUseSym
mol.symSz=0 #1 #TargetSpin-2*Sz
mol.symS2=0.0 #Total Spin
# Tempory file will be put to this dir
mol.tmpdir = './'
mol.build()

sval = 0.0
sz = 0.0
ifs2proj = True #False

flmps1 = h5py.File('./lmps0','r')
dmrg2 = mpo_dmrg_class.mpo_dmrg()
dmrg2.nsite = mol.sbas/2
dmrg2.sbas  = mol.sbas
dmrg2.isym = 2
dmrg2.build()
dmrg2.comm = mol.comm
dmrg2.qsectors = {str([mol.nelec,sz]):1} 
#---------------------------
if ifs2proj:
   dmrg2.ifs2proj = True
   dmrg2.npts = 4
   dmrg2.s2quad(sval,sz)
#---------------------------
mol.build()
dmrg2.path = mol.path
dmrg2.ifQt = False # KEY
dmrg2.partition()
dmrg2.loadInts(mol)
dmrg2.dumpMPO()
dmrg2.checkMPS(flmps1)

# <S2>	 
if not ifs2proj:
   info=None
else:
   info=[dmrg2.npts,sval,sz]
from zmpo_dmrg.source.properties import mpo_dmrg_propsItrf
expect = mpo_dmrg_propsItrf.eval_S2Global(dmrg2,flmps1,spinfo=info)
print 'expect_S2=',expect

# New L-MPS
dmrg2.final()
flmps1.close()
