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
ifs2proj = False

flmps0 = h5py.File('./lmps0','r')
nsite = flmps0['nsite'].value
print "Input qsym=",flmps0['qnum'+str(nsite)].value

flmps1 = h5py.File('./lmps1','w')

# Conversion
t0 = time.time()
mpo_dmrg_conversion.sweep_projection(flmps0,flmps1,ifQt,sval,thresh=1.e-6,\
			  	     ifcompress=True,ifBlockSingletEmbedding=False,\
				     ifBlockSymScreen=False,ifpermute=False)
path = './lmps_compact'
block_itrf.compact_rotL(flmps1,path)
#flmps1.close()
#exit()

#> import shutil
#> import numpy
#> from mpi4py import MPI
#> from zmpo_dmrg.source.samps import mpo_dmrg_conversion,block_itrf
#> from zmpo_dmrg.source.itools.molinfo import class_molinfo
#> from zmpo_dmrg.source import mpo_dmrg_class
#> from zmpo_dmrg.source import mpo_dmrg_schedule
#> from zmpo_dmrg.source.properties import mpo_dmrg_propsItrf
#> 
#> dmrg = mpo_dmrg_class.mpo_dmrg()
#> dmrg.nsite = nsite
#> dmrg.path = './'
#> groups = [[i] for i in range(nsite)]
#> expect = mpo_dmrg_propsItrf.eval_Local(dmrg,flmps1,groups,'N',spinfo=None)
#> print 'expect_N=',expect
#> print '<N>=',numpy.sum(expect)
#> t1 = time.time()
#> print 'dt=',(t1-t0)

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

# CHECK
from zmpo_dmrg.source import mpo_dmrg_init
esum = mpo_dmrg_init.genHops(dmrg2,flmps1,flmps1,'./tmp','R')
print 'esum=',numpy.sum(esum)
exit()

P00 = dmrg2.checkMPS(flmps0)[-1]
# Check energy
dmrg2.checkMPS(flmps1)
exit()

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
