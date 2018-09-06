import h5py
import numpy
import scipy.linalg
import zreleri
from zmpo_dmrg.source.tools import fielder

#
# Provide the basic interface
#
class iface:
   def __init__(self,mol,mf):
      self.iflowdin = False
      self.iflocal = False
      self.ifreorder = False
      self.ifgaunt = False
      # Interface
      self.mol  = mol
      self.mf   = mf
      self.nelec= mol.nelectron
      self.nbas = mol.nao_nr()
      self.sbas = mol.nao_2c()
      self.mo_coeff = mf.mo_coeff
      self.lmo_coeff = None
      # frozen core
      self.core = None
      self.act = None
      # test
      self.ifHFtest = True 

   # CheckFile
   def check(self,fname='mole.h5'):
      print '\n[iface.check]'	   
      f2 = h5py.File(fname, "r")
      print " nelec=",f2['cal'].attrs['nelec']
      print " sbas =",f2['cal'].attrs['sbas']
      print ' enuc =',f2['cal'].attrs['enuc']
      print ' ecor =',f2['cal'].attrs["ecor"]
      print ' escf =',f2['cal'].attrs['escf']
      print f2['int1e']
      print f2['int2e']
      f2.close()
      print " FINISH DUMPINFO into file =",fname
      return 0 
   
   # 2016.10.12: 
   # The very first version of dump4C - no localization, no reorder!
   def dump4C(self,fname='mole4C.h5'):
      print '\n[iface.dump4C]'	 
      #
      # Define active space
      #	
      if self.core is None and self.act is None:
	 # The first N2C orbitals are negative energy states
	 mo_coeff = self.mo_coeff[:,self.sbas:]
	 ncore = 0
	 nact  = self.sbas
      elif self.core is not None and self.act is None:
	 mo_coeff = self.mo_coeff[:,self.sbas:]
	 ncore = len(self.core)
	 nact  = self.sbas-ncore
      elif self.core is not None and self.act is not None:
	 mo_coeff = self.mo_coeff[:,self.sbas:]
	 mo_coeff = numpy.hstack((mo_coeff[:,numpy.array(self.core)],\
			  	  mo_coeff[:,numpy.array(self.act)]))
	 ncore = len(self.core)
	 nact  = len(self.act)
      elif self.core is None and self.act is not None:
         print 'Core must be set, if act exists' 
         exit()
      #
      # Effective parameters (Keff,Neff)
      #
      norb  = ncore + nact
      nelec = self.nelec - ncore 
      orderAct = range(nact)
      order = numpy.array(range(ncore)+list(ncore+numpy.array(orderAct)))
      print ' self.core = ',self.core
      print ' self.act  = ',self.act
      print ' ncore/nact/norb = ',(ncore,nact,norb) 
      print ' nelec = ',nelec
      print ' order = ',order
      # 
      # Transform integrals with core+act: H1e
      #
      mo_coeff = mo_coeff[:,numpy.array(order)].copy()
      h1e = self.mf.get_hcore()
      hmo = reduce(numpy.dot,(mo_coeff.T.conj(),h1e,mo_coeff))
      print ' shape of mo  =',mo_coeff.shape
      print ' shape of hmo =',hmo.shape
      print ' deviation H1 =',numpy.linalg.norm(hmo-hmo.T.conj())
      # 
      # Transform integrals with core+act: H2e
      #
      erifile = 'tmperi'
      zreleri.ao2mo(self.mf, mo_coeff, erifile)
      if self.ifgaunt: zreleri.ao2mo_gaunt(self.mf, mo_coeff, erifile)
      with h5py.File(erifile) as f1:
	 eri = f1['ericas'].value # [ij|kl]
	 eri = eri.reshape(norb,norb,norb,norb)
	 eriC = eri[:ncore,:ncore,:ncore,:ncore]
      # 
      # ecore = hii + 1/2<ij||ij> = hij + 1/2*([ii|jj]-[ij|ji])
      # 
      ecore = numpy.einsum('ii',hmo[:ncore,:ncore])\
 	    + 0.5*(numpy.einsum('iijj',eriC)-numpy.einsum('ijji',eriC))
      ecore = ecore.real
      #
      # fock_ij = hij + <ik||jk> = hij + [ij|kk]-[ik|kj]; k in core
      #
      fock = hmo[ncore:,ncore:].copy()
      for i in range(nact):
         for j in range(nact):
 	    for k in range(ncore):
	       fock[i,j] +=  eri[ncore+i,ncore+j,k,k]-eri[ncore+i,k,k,ncore+j]
      hmo = fock.copy()
      #
      # Get antisymmetrized integrals
      #
      eri = eri[ncore:,ncore:,ncore:,ncore:].copy()
      # Reorder
      if self.ifreorder:
         # Kij is real symmetric 
         kij = numpy.einsum('ijji->ij',eri)
         kij_imag = numpy.linalg.norm(kij.imag)
         kij_real = kij.real
	 print '\nReorder of spinors:'
         print ' Norm of kij_imag =',kij_imag
         print ' Symm of kij_real =',numpy.linalg.norm(kij_real-kij_real.T)
         order = fielder.orbitalOrdering(kij_real,'kmat')
         print ' order =',order
	 hmo = hmo[numpy.ix_(order,order)].copy()
	 eri = eri[numpy.ix_(order,order,order,order)].copy()
      else:
         order = range(eri.shape[0])	
      # <ij|kl>=[ik|jl]
      eri = eri.transpose(0,2,1,3)
      # Antisymmetrize V[pqrs]=-1/2*<pq||rs> - In MPO construnction, only r<s part is used. 
      eri = -0.5*(eri-eri.transpose(0,1,3,2))
      # 
      # DUMP modified Integrals
      #
      enuc  = self.mol.energy_nuc()
      print '\nBasic information:'
      print ' enuc  = ',enuc
      print ' ecore = ',ecore
      print ' nelec = ',nelec
      print ' nact  = ',nact
      print ' hmo   = ',hmo.shape
      print ' eriA  = ',eri.shape
      print ' deviation H1 =',numpy.linalg.norm(hmo-hmo.T.conj())
      flter = 'lzf'
      f = h5py.File(fname, "w")
      f.create_dataset("int1e", data=hmo, compression=flter)
      f.create_dataset("int2e", data=eri, compression=flter)
      #
      # Basic information
      #
      cal = f.create_dataset("cal",(1,),dtype='i')
      cal.attrs["nelec"] = nelec
      cal.attrs["sbas"]  = nact
      cal.attrs["enuc"]  = enuc
      cal.attrs["escf"]  = 0. # Not useful at all: self.mf.energy_elec(self.mf.make_rdm1())[0]
      cal.attrs["ecor"]  = ecore
      #
      # Occupation & Symmetry
      #
      occun = numpy.zeros(nact)
      for i in range(self.nelec-ncore):
	 occun[i] = 1.0
      print
      print ' order:',order
      print ' initial occun for',len(occun),' spin orbitals:\n',occun
      occun = occun[order].copy()
      orbsym = numpy.array([0]*nact) 
      spinsym = numpy.array([[0,1] for i in range(nact/2)]).flatten()
      print " final occun:\n",occun
      print " orbsym :",orbsym
      print " spinsym:",spinsym
      f.create_dataset("occun",data=occun)
      f.create_dataset("orbsym",data=orbsym)
      f.create_dataset("spinsym",data=spinsym)
      # Finalize
      f.close()
      import os 
      os.remove(erifile)
      # Test
      if self.ifHFtest:
	 etot = ecore \
	      + numpy.einsum('ii',hmo[:nelec,:nelec])\
	      - numpy.einsum('ijij',eri[:nelec,:nelec,:nelec,:nelec])
	 etot = etot.real
	 escf = self.mf.energy_elec(self.mf.make_rdm1())[0]
	 print ' HFtest_etot',etot
         print ' HFtest_escf',escf
         print ' HFtest_edif',etot-escf
         assert abs(etot-escf)<1.e-8
      print '\nSuccessfully dump information for FS-DMRG calculations! fname=',fname
      self.check(fname)
      return 0
