#!/usr/bin/env python
#
# Angular momentum
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
#
# def iftriangle(j1,j2,j3):
# def cgcoeff(j1,m1,j2,m2,j3,m3):
# def cgtensor_slow(j1,j2,j3):
# def cgtensor(j1,j2,j3):
# def cgtensor_fast1(j1,j2,j3):
# def cgtensor_fast2(j1,j2,j3):
# 
from sympy import N
from sympy.physics.quantum.cg import CG
import numpy
from zmpo_dmrg.libs import libangular

def iftriangle(j1,j2,j3):
   jmin = abs(j1-j2)
   ncase = int(j1+j2-jmin+1)
   ift = False
   for i in range(ncase):
      if abs(j3-jmin-i)<1.e-10: 
         ift = True
	 break
   return ift

def cgcoeff(j1,m1,j2,m2,j3,m3):
   return N(CG(j1,m1,j2,m2,j3,m3).doit())

# <j1m1j2m2|j3m3>
def cgtensor_slow(j1,j2,j3):
   d1 = int(2*j1+1)
   d2 = int(2*j2+1)
   d3 = int(2*j3+1)
   cgt = numpy.zeros((d1,d2,d3))
   ift = iftriangle(j1,j2,j3)
   if ift:
      for i1 in range(d1):
         m1 = -j1+i1
	 for i2 in range(d2):
            m2 = -j2+i2
            for i3 in range(d3):
               m3 = -j3+i3
               cgt[i1,i2,i3] = cgcoeff(j1,m1,j2,m2,j3,m3)
   return ift,cgt

# <j1m1j2m2|j3m3>
def cgtensor(j1,j2,j3):
   d1 = int(2*j1+1)
   d2 = int(2*j2+1)
   d3 = int(2*j3+1)
   cgt = numpy.zeros((d1,d2,d3))
   ift = iftriangle(j1,j2,j3)
   if ift:
      for i1 in range(d1):
         m1 = -j1+i1
	 for i2 in range(d2):
            m2 = -j2+i2
            for i3 in range(d3):
               m3 = -j3+i3
               cgt[i1,i2,i3] = libangular.anglib.cleb(int(2*j1),int(2*m1),\
			       			      int(2*j2),int(2*m2),\
						      int(2*j3),int(2*m3))
   return ift,cgt

# Return tensor version
def cgtensor_fast1(j1,j2,j3):
   ift = iftriangle(j1,j2,j3)
   tj1 = int(2*j1)
   tj2 = int(2*j2)
   tj3 = int(2*j3)
   cgt = libangular.anglib.cgtensor(tj1,tj2,tj3)
   return ift,cgt

# For testing purpose
def cgtensor_fast2(j1,j2,j3):
   ift = iftriangle(j1,j2,j3)
   tj1 = int(2*j1)
   tj2 = int(2*j2)
   tj3 = int(2*j3)
   import ctypes
   import os
   path = os.path.join("/Users/zhendongli2008/Desktop/work/pcodes/zmpo_dmrg",'libs/libangular.so')
   lib = ctypes.CDLL(path)
   ctj1 = ctypes.c_int(tj1)
   ctj2 = ctypes.c_int(tj2)
   ctj3 = ctypes.c_int(tj3)
   cgt = numpy.zeros((tj1+1,tj2+1,tj3+1),order='F')
   tmp = ctypes.c_int(0)
   lib.f2pywrap_anglib_cgtensor2_(ctypes.byref(tmp),\
		   	          ctypes.byref(ctj1),\
		   		  ctypes.byref(ctj2),\
		   		  ctypes.byref(ctj3),\
	 		          cgt.ctypes.data_as(ctypes.c_void_p))
   return ift,cgt

if __name__ == '__main__':
   print cgcoeff(1,0,1,0,1,0)
   print cgcoeff(0.5,0.5,0.5,-0.5,1,0)
   print cgcoeff(0.5,0.5,0.5,-0.5,0,0)
   print cgcoeff(0.5,-0.5,0.5,0.5,1,0)
   print cgcoeff(0.5,-0.5,0.5,0.5,0,0)
   print cgcoeff(2.5,2.5,1.5,0.5,4,3)
   import math
   print math.sqrt(3.0/8.0)

   print cgtensor(0.5,0.5,0.5)
   print cgtensor(0.5,0.5,0)
   print cgtensor(0.5,0.5,1)
   
   print
   import time
   j1 = 50
   j2 = 50
   j3 = 50
   t0 = time.time()
   ift,cgt1 = cgtensor(j1,j2,j3)
   t1 = time.time()
   ift,cgt2 = cgtensor_fast1(j1,j2,j3)
   t2 = time.time()
   #ift,cgt3 = cgtensor_fast2(j1,j2,j3)
   #t3 = time.time()
   print 'tA=',t1-t0
   print 'tB=',t2-t1
   #print 'tC=',t3-t2
   print numpy.sum(abs(cgt1)),numpy.sum(abs(cgt2)),numpy.linalg.norm(cgt1-cgt2)
   #print numpy.sum(abs(cgt1)),numpy.sum(abs(cgt3)),numpy.linalg.norm(cgt1-cgt3)
