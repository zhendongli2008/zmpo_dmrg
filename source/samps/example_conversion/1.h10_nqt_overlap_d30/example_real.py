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
from zmpo_dmrg.source.itools import ipyscf_real
#mol.spin = 2 - triplet
iface = ipyscf_real.iface(mol,mf)
#iface.ccsd()
#iface.fci()
#iface.molden(iface.mo_coeff,'CMO')
#print iface.reorder(iface.mo_coeff)
#iface.local()
#iface.molden(iface.lmo_coeff,'LMO')
#print iface.reorder(iface.lmo_coeff)

iface.iflocal = True #False
iface.iflowdin = True #False
iface.ifreorder = False
iface.ifdual = False
iface.param1 = 0.0 #1.0
iface.param2 = 0.0 #1.0
iface.spin   = 0.0 #0.0

iface.dump(fname='mole.h5')
