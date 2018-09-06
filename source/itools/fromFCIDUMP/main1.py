import numpy
import h5py
from mpi4py import MPI
from zmpo_dmrg.source.itools.molinfo import class_molinfo
from zmpo_dmrg.source.qtensor import qtensor_api
from zmpo_dmrg.source import mpo_dmrg_class
from zmpo_dmrg.source import mpo_dmrg_schedule
import shutil

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

# To be changed.
mol.tmpdir ="/scratch/global/zhendong/spmps_oec1/"
mol.build()

# To be set manually
nelec = 10
sval = 4
sz = 4
conf = [1,1]+[1,0,]*8+[0,0]

# 1. Using an MPS in Qt form
flmps1 = h5py.File('./lmpsQ0','r')
dmrg2 = mpo_dmrg_class.mpo_dmrg()
dmrg2.const = mol.ecor
dmrg2.nsite = mol.sbas/2
dmrg2.sbas  = mol.sbas
dmrg2.isym = 2
dmrg2.build()
dmrg2.comm = mol.comm
dmrg2.qsectors = {str([nelec,sz]):1}

np = 1
sc2 = mpo_dmrg_schedule.schedule()
sc2.MaxMs  = [1] + [50]*np + [50]*2 #+ [100]*(2*np) + [100]*(2*np)
ns = len(sc2.MaxMs)
sc2.Sweeps = range(ns)
sc2.Tols   = [1.e-2] + [1.e-3]*np + [1.e-4]*2 #[1.e-4]*(2*np) + [1.e-5]*(2*np)
sc2.Noises = [1.e-4] + [1.e-4]*np + [0.0]*2 #[0.0]*(4*np)
sc2.coff = 0

sc2.Tag = 'Normal2'
sc2.collect()
sc2.maxiter = ns
sc2.prt()

#---------------------------
dmrg2.ifs2proj = True
dmrg2.npts = 4
dmrg2.s2quad(sval,sz)
#---------------------------
mol.build()
dmrg2.path = mol.path
dmrg2.ifQt = True
dmrg2.partition()
dmrg2.loadInts(mol)
dmrg2.dumpMPO()
dmrg2.default(sc2,flmps1)
# New L-MPS
dmrg2.checkMPS()
dmrg2.final()
flmps1.close()

if rank == 0:
   shutil.copy( dmrg2.path+'/lmps','./lmpsQs')
   print 'Energy',dmrg2.Energy
