#!/usr/bin/env python
#
# Some global settings: especially for data type used in DMRG!
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
#
import os
import ctypes
import numpy
from mpi4py import MPI

pth = os.path.dirname(os.path.abspath(__file__))
pth = os.path.split(pth)[0] 
path = os.path.join(pth,'libs/libqsym.so')
libqsym = ctypes.CDLL(path)
path = os.path.join(pth,'libs/libangular.so')
libangular = ctypes.CDLL(path)

dmrg_type = 'real'
if dmrg_type == 'real':
   dmrg_dtype = numpy.float_
   dmrg_mtype = MPI.DOUBLE
elif dmrg_type == 'complex': 
   dmrg_dtype = numpy.complex_
   dmrg_mtype = MPI.DOUBLE_COMPLEX
