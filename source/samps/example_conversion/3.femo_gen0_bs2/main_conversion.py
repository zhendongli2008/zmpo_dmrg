import math
import time
import h5py
import shutil
import numpy
from mpi4py import MPI
from zmpo_dmrg.source.samps import mpo_dmrg_conversion
from zmpo_dmrg.source.qtensor import qtensor_api

sval = 1.5
sz = 1.5
ifQt = False

fname = 'lmpsQs'#s' #0'
flmpsQ = h5py.File('./mpsFiles/'+fname,'r')
flmps0 = h5py.File(fname+'_NQt0','w')
flmps1 = h5py.File(fname+'_NQt1','w')
flmps2 = h5py.File(fname+'_Qt1','w')

qtensor_api.fmpsQtReverse(flmpsQ,flmps0,'L')

# Conversion
t0 = time.time()
mpo_dmrg_conversion.sweep_projection(flmps0,flmps1,ifQt,sval,thresh=1.e-3)
t1 = time.time()
print 'dt=',(t1-t0)

qtensor_api.fmpsQt(flmps1,flmps2,'L')

flmpsQ.close()
flmps0.close()
flmps1.close()
flmps2.close()
