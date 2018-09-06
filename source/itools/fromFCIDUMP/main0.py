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

################################
# 0. Initialize an MPS(N,Sz) 
################################
dmrg = mpo_dmrg_class.mpo_dmrg()
dmrg.occun = numpy.array(conf)
dmrg.const = mol.ecor
dmrg.path = mol.path
dmrg.nsite = mol.sbas/2
dmrg.sbas  = mol.sbas
dmrg.isym = 2
dmrg.build()
dmrg.comm = mol.comm
dmrg.qsectors = {str([nelec,sz]):1} 
sc = mpo_dmrg_schedule.schedule()
sc.fixed(maxM=1,maxiter=0)
sc.prt()
dmrg.partition()
dmrg.loadInts(mol)
dmrg.dumpMPO()
dmrg.default(sc)

#-------------------------------------------------------
if rank == 0:
   flmps0 = dmrg.flmps
   flmps1 = h5py.File(dmrg.path+'/lmpsQt','w')
   qtensor_api.fmpsQt(flmps0,flmps1,'L')
   flmps0.close()
   flmps1.close()
   shutil.copy(dmrg.path+'/lmpsQt','./lmpsQ0')
#-------------------------------------------------------
mol.comm.Barrier()
