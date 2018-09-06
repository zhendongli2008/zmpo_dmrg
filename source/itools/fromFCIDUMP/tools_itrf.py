import h5py
import numpy

def dump(info,ordering=None,fname='mole.h5'):
   ecore,int1e,int2e = info
   if ordering != None:
      int1e = int1e[numpy.ix_(ordering,ordering)].copy()
      int2e = int2e[numpy.ix_(ordering,ordering,ordering,ordering)].copy()
   # dump information
   nbas = int1e.shape[0]
   sbas = nbas*2
   print '\n[tools_itrf.dump] interface from FCIDUMP with nbas=',nbas
   f = h5py.File(fname, "w")
   cal = f.create_dataset("cal",(1,),dtype='i')
   cal.attrs["nelec"] = 0.
   cal.attrs["sbas"]  = sbas
   cal.attrs["enuc"]  = 0.
   cal.attrs["ecor"]  = ecore
   cal.attrs["escf"]  = 0. # Not useful at all
   # Intergrals
   flter = 'lzf'
   # INT1e:
   h1e = numpy.zeros((sbas,sbas))
   h1e[0::2,0::2] = int1e # AA
   h1e[1::2,1::2] = int1e # BB
   # INT2e:
   h2e = numpy.zeros((sbas,sbas,sbas,sbas))
   h2e[0::2,0::2,0::2,0::2] = int2e # AAAA
   h2e[1::2,1::2,1::2,1::2] = int2e # BBBB
   h2e[0::2,0::2,1::2,1::2] = int2e # AABB
   h2e[1::2,1::2,0::2,0::2] = int2e # BBAA
   # <ij|kl> = [ik|jl]
   h2e = h2e.transpose(0,2,1,3)
   # Antisymmetrize V[pqrs]=-1/2*<pq||rs> - In MPO construnction, only r<s part is used. 
   h2e = -0.5*(h2e-h2e.transpose(0,1,3,2))
   int1e = f.create_dataset("int1e", data=h1e, compression=flter)
   int2e = f.create_dataset("int2e", data=h2e, compression=flter)
   # Occupation
   occun = numpy.zeros(sbas)
   orbsym = numpy.array([0]*sbas)
   spinsym = numpy.array([[0,1] for i in range(nbas)]).flatten()
   f.create_dataset("occun",data=occun)
   f.create_dataset("orbsym",data=orbsym)
   f.create_dataset("spinsym",data=spinsym)
   f.close()
   print ' Successfully dump information for MPO-DMRG calculations! fname=',fname
   print ' with ordering',ordering
   return 0

if __name__ == '__main__':
   import tools_io
   info = tools_io.loadERIs()
   ordering = range(10)
   ordering = ordering[::-1]
   print 'ordering=',ordering
   dump(info,ordering)
