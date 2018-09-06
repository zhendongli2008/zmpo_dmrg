#
# Optimized version of interface for dumping integrals
# The cutted mo_coeff must be inputed directly (maybe with nfrozen).
#
import h5py
import numpy
import scipy.linalg
from zmpo_dmrg.source.tools import fielder
from pyscf import ao2mo
from pyscf.scf import hf

# Provide the basic interface
class iface:
   def __init__(self,mol,mf):
      self.iflowdin = False
      self.iflocal = False
      self.ifreorder = False
      # Interface
      self.mol  = mol
      self.mf   = mf
      self.nelec= mol.nelectron
      self.spin = mol.spin
      self.nalpha = (mol.nelectron+mol.spin)/2
      self.nbeta  = (mol.nelectron-mol.spin)/2
      try: 
	 self.nbas = mf.mo_coeff[0].shape[0]
      except:
	 self.nbas = 0     
      self.mo_coeff = mf.mo_coeff
      self.lmo_coeff = None
      # frozen core
      self.nfrozen = 0
      self.ncut = 0

   # This is the central part
   def dump(self,fname='mole.h5'):
      # Effective
      nbas = self.nbas-self.nfrozen 
      sbas = nbas*2
      print '\n[iface.dump] (self.nbas,nbas)=',(self.nbas,nbas) 
      # Basic information
      f = h5py.File(fname, "w")
      cal = f.create_dataset("cal",(1,),dtype='i')
      enuc = self.mol.energy_nuc()
      nelecA = self.nelec - self.nfrozen*2
      cal.attrs["nelec"] = nelecA 
      cal.attrs["sbas"]  = sbas 
      cal.attrs["enuc"]  = enuc
      cal.attrs["escf"]  = 0. # Not useful at all
      # Intergrals
      flter = 'lzf'
      mcoeffC = self.mo_coeff[:,:self.nfrozen].copy()
      mcoeffA = self.mo_coeff[:,self.nfrozen:].copy()
      # Core part
      pCore = 2.0*mcoeffC.dot(mcoeffC.T)
      vj,vk = hf.get_jk(self.mol,pCore)
      h = self.mf.get_hcore()
      fock = h + vj - 0.5*vk  
      fmo = reduce(numpy.dot,(mcoeffA.T,fock,mcoeffA))
      ecore = 0.5*numpy.trace(pCore.dot(h+fock))
      # Active part
      nact = mcoeffA.shape[1]
      eri = ao2mo.general(self.mol,(mcoeffA,mcoeffA,mcoeffA,mcoeffA),compact=0)
      eri = eri.reshape(nact,nact,nact,nact)
      # Reorder
      if self.ifreorder:
         order = fielder.orbitalOrdering(eri,'kij')
      else:
	 order = range(mcoeffA.shape[1])	    
      # Sort
      mcoeffA = mcoeffA[:,numpy.array(order)].copy()
      fmo = fmo[numpy.ix_(order,order)].copy()
      eri = eri[numpy.ix_(order,order,order,order)].copy()
      #========================
      # Spin orbital integrals
      #========================
      gmo_coeff = numpy.hstack((mcoeffC,mcoeffA))
      print 'gmo_coeff.shape=',gmo_coeff.shape
      f.create_dataset("mo_coeff_spatialAll", data=gmo_coeff)
      # INT1e:
      h1e = numpy.zeros((sbas,sbas)) 
      h1e[0::2,0::2] = fmo # AA
      h1e[1::2,1::2] = fmo # BB
      # INT2e:
      h2e = numpy.zeros((sbas,sbas,sbas,sbas))
      h2e[0::2,0::2,0::2,0::2] = eri # AAAA
      h2e[1::2,1::2,1::2,1::2] = eri # BBBB
      h2e[0::2,0::2,1::2,1::2] = eri # AABB
      h2e[1::2,1::2,0::2,0::2] = eri # BBAA
      # <ij|kl> = [ik|jl]
      h2e = h2e.transpose(0,2,1,3)
      # Antisymmetrize V[pqrs]=-1/2*<pq||rs> - In MPO construnction, only r<s part is used. 
      h2e = -0.5*(h2e-h2e.transpose(0,1,3,2))
      print 'E[core]=',ecore
      cal.attrs["ecor"]  = ecore
      int1e = f.create_dataset("int1e", data=h1e, compression=flter)
      int2e = f.create_dataset("int2e", data=h2e, compression=flter)
      # Occupation
      occun = numpy.zeros(sbas)
      for i in range(self.nalpha-self.nfrozen):
	 occun[2*i] = 1.0
      for i in range(self.nbeta-self.nfrozen):
	 occun[2*i+1] = 1.0
      print
      print 'initial occun for',len(occun),' spin orbitals:\n',occun
      sorder = numpy.array([[2*i,2*i+1] for i in order]).flatten()
      occun = occun[sorder].copy()
      assert abs(numpy.sum(occun)-nelecA)<1.e-10
      print "sorder:",sorder
      print "occun :",occun
      orbsym = numpy.array([0]*sbas) 
      spinsym = numpy.array([[0,1] for i in range(nbas)]).flatten()
      print "orbsym :",orbsym
      print "spinsym:",spinsym
      f.create_dataset("occun",data=occun)
      f.create_dataset("orbsym",data=orbsym)
      f.create_dataset("spinsym",data=spinsym)
      f.close()
      print 'Successfully dump information for HS-DMRG calculations! fname=',fname
      self.check(fname)
      return 0

   def check(self,fname='mole.h5'):
      print '\n[iface.check]'	   
      f2 = h5py.File(fname, "r")
      print "nelec=",f2['cal'].attrs['nelec']
      print "sbas =",f2['cal'].attrs['sbas']
      print 'enuc =',f2['cal'].attrs['enuc']
      print 'ecor =',f2['cal'].attrs["ecor"]
      print 'escf =',f2['cal'].attrs['escf']
      print f2['int1e']
      print f2['int2e']
      f2.close()
      print "FINISH DUMPINFO into file =",fname
      return 0
