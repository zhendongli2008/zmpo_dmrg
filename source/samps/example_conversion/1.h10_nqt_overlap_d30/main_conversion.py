import math
import time
import h5py
import shutil
import numpy
from mpi4py import MPI
from zmpo_dmrg.source.samps import mpo_dmrg_conversion,block_itrf

sval = 1.0
sz = 0.0
ifQt = False
ifs2proj = True #False

flmps0 = h5py.File('./lmps0','r')
nsite = flmps0['nsite'].value
print "Input qsym=",flmps0['qnum'+str(nsite)].value

flmps1 = h5py.File('./lmps1','w')

# Conversion
t0 = time.time()
mpo_dmrg_conversion.sweep_projection(flmps0,flmps1,ifQt,sval,thresh=1.e-8,\
			  	     ifcompress=True,ifBlockSingletEmbedding=False,\
				     ifBlockSymScreen=True)
path = './lmps_compact'
block_itrf.compact_rotL(flmps1,path)
flmps1.close()
exit()

# *** Without singlet embedding: 
#
# finalize left_sweep ...
#  ibond/dim(NS)/dim(NSM)= 0 1 1
#  ibond/dim(NS)/dim(NSM)= 1 3 4
#  ibond/dim(NS)/dim(NSM)= 2 8 14
#  ibond/dim(NS)/dim(NSM)= 3 13 27
#  ibond/dim(NS)/dim(NSM)= 4 14 32
#  ibond/dim(NS)/dim(NSM)= 5 16 39
#  ibond/dim(NS)/dim(NSM)= 6 20 51
#  ibond/dim(NS)/dim(NSM)= 7 16 42
#  ibond/dim(NS)/dim(NSM)= 8 18 49
#  ibond/dim(NS)/dim(NSM)= 9 12 36
#  ibond/dim(NS)/dim(NSM)= 10 1 1
#
# *** With singlet embedding:
#
# finalize left_sweep ...
#  ibond/dim(NS)/dim(NSM)= 0 1 3
#  ibond/dim(NS)/dim(NSM)= 1 4 12
#  ibond/dim(NS)/dim(NSM)= 2 14 42
#  ibond/dim(NS)/dim(NSM)= 3 24 77
#  ibond/dim(NS)/dim(NSM)= 4 24 76
#  ibond/dim(NS)/dim(NSM)= 5 24 80
#  ibond/dim(NS)/dim(NSM)= 6 25 81
#  ibond/dim(NS)/dim(NSM)= 7 20 54
#  ibond/dim(NS)/dim(NSM)= 8 15 35
#  ibond/dim(NS)/dim(NSM)= 9 8 12
#  ibond/dim(NS)/dim(NSM)= 10 1 1
#
t1 = time.time()
print 'dt=',(t1-t0)

#
# Check
#
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

P00 = dmrg2.checkMPS(flmps0)[-1]
# Check energy
dmrg2.checkMPS(flmps1)

#P11 = dmrg2.checkMPS(flmps1)[-1]
#P01 = dmrg2.checkMPS(flmps0,flmps1)[-1]
#print 'Overlap: <Psi|P|Psi0>*N0=',P01/math.sqrt(P00)
#
# The Remarkable fact is that P11=1 by construction,
# such that P01=O01, and <Psi|P|Psi0>*N0=<Psi|Psi0>*N0, 
# which is extremely easy to compute as no projector is involved.
#
from zmpo_dmrg.source import mpo_dmrg_init
pop = mpo_dmrg_init.genPops(dmrg2,flmps0,flmps0,'./tmp_sop','L')
pop = numpy.dot(dmrg2.qwts,pop)
dmrg2.ifs2proj = False
sop = mpo_dmrg_init.genSops(dmrg2,flmps0,flmps1,'./tmp_sop','L')
#
# pop= 0.618247725148
# sop= 0.770106557564
# Overlap: <Psi|P|Psi0>*N0= 0.979421330091
#
print
print 'pop=',pop
print 'sop=',sop
print 'Overlap: <Psi|P|Psi0>*N0=',sop/math.sqrt(pop)
exit()

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

flmps0.close()
flmps1.close()
