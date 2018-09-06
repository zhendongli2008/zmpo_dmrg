import numpy
from pyscf import gto,scf

#==================================================================
# MOLECULE
#==================================================================
mol = gto.Mole()
mol.verbose = 5 #6

#==================================================================
# Coordinates and basis
#==================================================================
molname = 'h' #be' #h2cluster'#h2o3' #c2'

if molname == 'h':
   R = 2.0
   natoms = 10 #10 #40 #,14,20,50
   mol.atom = [['H',(0.0,0.0,i*R)] for i in range(natoms)]
   mol.basis = 'sto-3g' #cc-pvdz' #sto-3g' #6-31g' 

#==================================================================
mol.symmetry = False #True
mol.charge = 0
mol.spin = 0
#==================================================================
mol.build()

#==================================================================
# SCF
#==================================================================
mf = scf.RHF(mol)
mf.init_guess = 'atom'
mf.level_shift = 0.0
mf.max_cycle = 100
mf.conv_tol=1.e-14
#mf.irrep_nelec = {'B2':2}
print(mf.scf())
mf.analyze()

#==================================================================
# Dump integrals
#==================================================================
import h5py
f = h5py.File('mole.h5')
mo = f['mo_coeff_spatialAll'].value
f.close()

from pyscf import mcscf
from pyscf.dmrgscf.dmrgci import *
import time

t0 = time.time()
mc = mcscf.CASCI(mf, 10, [5,5])
mc.mo_coeff = mo
mc.fcisolver = DMRGCI(mol)
mc.onlywriteIntegral = True
mc.casci()
