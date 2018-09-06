import math
import time
import h5py
import shutil
import numpy
from mpi4py import MPI
from zmpo_dmrg.source.qtensor import qtensor_api
from zmpo_dmrg.source.samps import mpo_dmrg_conversion,block_itrf

sval = 0.5
sz = 0.5
ifQt = False

fname = 'lmpsQs'#s' #0'
flmpsQ = h5py.File(fname,'r')
flmps0 = h5py.File(fname+'_NQt0','w')
flmps1 = h5py.File(fname+'_NQt1','w')

qtensor_api.fmpsQtReverse(flmpsQ,flmps0,'L')

# Conversion
t0 = time.time()
mpo_dmrg_conversion.sweep_projection(flmps0,flmps1,ifQt,sval,thresh=1.e-8,\
				     ifcompress=True,\
				     ifBlockSingletEmbedding=True,\
				     ifBlockSymScreen=True,\
				     ifpermute=True)
path = './lmps_compact'
block_itrf.compact_rotL(flmps1,path)
t1 = time.time()
print 'dt=',(t1-t0)

flmpsQ.close()
flmps0.close()
flmps1.close()
