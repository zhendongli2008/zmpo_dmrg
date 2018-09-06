import time
import h5py
import shutil
import numpy
from mpi4py import MPI
from zmpo_dmrg.source.samps import mpo_dmrg_conversion

sval = 0.0
sz = 0.0
ifQt = False
ifs2proj = False

flmps0 = h5py.File('./lmps0','r')
flmps1 = h5py.File('./lmps1','w')

# Conversion
t0 = time.time()
mpo_dmrg_conversion.sweep_projection(flmps0,flmps1,ifQt,sval,thresh=1.e-4)
t1 = time.time()
print 'dt=',(t1-t0)
