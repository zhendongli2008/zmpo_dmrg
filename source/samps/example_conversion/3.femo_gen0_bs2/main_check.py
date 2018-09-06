import h5py
nsite = 76
sval = 1.5
sz = 1.5
ifs2proj = True
fname = 'lmpsQs'
flmps0 = h5py.File('./mpsFiles/'+fname,'r')
flmps1 = h5py.File(fname+'_Qt1','r')

#=== Check the converted MPS ===#
import math
import shutil
import numpy
from mpi4py import MPI
from zmpo_dmrg.source.itools.molinfo import class_molinfo
from zmpo_dmrg.source import mpo_dmrg_class
from zmpo_dmrg.source import mpo_dmrg_io
from zmpo_dmrg.source import mpo_dmrg_init

comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank

dmrg2 = mpo_dmrg_class.mpo_dmrg()
dmrg2.nsite = nsite
dmrg2.sbas  = nsite*2
dmrg2.isym = 2
dmrg2.build()
dmrg2.comm = comm
#---------------------------
if ifs2proj:
   dmrg2.ifs2proj = True
   dmrg2.npts = 11
   dmrg2.s2quad(sval,sz)
#---------------------------
dmrg2.path = './'
dmrg2.ifQt = True
dmrg2.partition()

if fname == 'lmpsQs':
   pop = 0.280309490026
elif fname == 'lmpsQ0':
   mpo_dmrg_io.dumpMPO_R(dmrg2)
   pop = mpo_dmrg_init.genPops(dmrg2,flmps0,flmps0,'./tmp_pop','L')
   pop = numpy.dot(dmrg2.qwts,pop)
   
dmrg2.ifs2proj = False
sop = mpo_dmrg_init.genSops(dmrg2,flmps0,flmps1,'./tmp_sop','L')
print
print 'pop(<Psi0|P|Psi0>)=',pop
print 'sop(<Psi0|P|Psi1>)=',sop
print 'Overlap: <Psi|P|Psi0>*N0=',sop/math.sqrt(pop)
sop = mpo_dmrg_init.genSops(dmrg2,flmps1,flmps1,'./tmp_sop','L')
print 'sop(<Psi1|P|Psi1>)=',sop

#> # <S2>	 
#> if not ifs2proj:
#>    info=None
#> else:
#>    info=[dmrg2.npts,sval,sz]
#> from zmpo_dmrg.source.properties import mpo_dmrg_propsItrf
#> expect = mpo_dmrg_propsItrf.eval_S2Global(dmrg2,flmps1,spinfo=info)
#> print 'expect_S2=',expect

# New L-MPS
dmrg2.final()
flmps0.close()
flmps1.close()
