#
#    34   126   126     1   T       -5.1491130015      0.000E+00      0.754E-09
#    34   126   126     2   T       -5.1372384525      0.195E-13      0.814E-08
#    34   126   126     3   T       -5.1233644821      0.266E-14      0.304E-08
#    34   126   126     4   T       -5.1180021882      0.983E-12      0.281E-06
#
#  State =    1    -0.000  Proj  =    1.000
#  State =    2     2.000  Proj  =    0.986
#  State =    3     2.000  Proj  =    0.986
#  State =    4     0.000  Proj  =    1.000
#
import h5py
import shutil
import numpy
from mpi4py import MPI
from zmpo_dmrg.source.itools.molinfo import class_molinfo
from zmpo_dmrg.source import mpo_dmrg_class
from zmpo_dmrg.source import mpo_dmrg_schedule
from zmpo_dmrg.source.qtensor import qtensor_api

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

sval = 1.0
sz = 0.0

################################
# 0. Initialize an MPS(N,Sz) 
################################
dmrg = mpo_dmrg_class.mpo_dmrg()
dmrg.occun = numpy.array([1.,0.,0.,1.]*3) # AFM initial guess
dmrg.path = mol.path
dmrg.nsite = mol.sbas/2
dmrg.sbas  = mol.sbas
dmrg.isym = 2
dmrg.build()
dmrg.comm = mol.comm
dmrg.qsectors = {str([mol.nelec,sz]):1} 
sc = mpo_dmrg_schedule.schedule()
sc.fixed(maxM=1,maxiter=0)
sc.prt()
dmrg.ifIO = True
dmrg.ifQt = False
dmrg.partition()
dmrg.loadInts(mol)
dmrg.dumpMPO()
dmrg.default(sc)
dmrg.checkMPS()

flmps1 = h5py.File(dmrg.path+'/lmpsQ0','w')
qtensor_api.fmpsQt(dmrg.flmps,flmps1,'L')
flmps1.close()

dmrg.final()

################################
# 1. Using an MPS in Qt form
################################
flmps1 = h5py.File(dmrg.path+'/lmpsQ0','r')
dmrg2 = mpo_dmrg_class.mpo_dmrg()
dmrg2.nsite = mol.sbas/2
dmrg2.sbas  = mol.sbas
dmrg2.isym = 2
dmrg2.build()
dmrg2.comm = mol.comm
dmrg2.qsectors = {str([mol.nelec,sz]):1} 
sc2 = mpo_dmrg_schedule.schedule(tol=1.e-8)
sc2.maxM = 5
sc2.maxiter = 6
sc2.normal()
sc2.prt()
#---------------------------
sc2.Tols = [10.*tol for tol in sc2.Tols] 
dmrg2.ifs2proj = True
dmrg2.npts = 4
dmrg2.s2quad(sval,sz)
#---------------------------
mol.build()
dmrg2.path = mol.path
dmrg2.ifQt = True # KEY
dmrg2.partition()
dmrg2.loadInts(mol)
dmrg2.dumpMPO()
dmrg2.default(sc2,flmps1)
dmrg2.checkMPS()
# New L-MPS
dmrg2.final()
flmps1.close()

if rank == 0:
   shutil.copy(dmrg2.path+'/lmps','./lmpsQ1')
