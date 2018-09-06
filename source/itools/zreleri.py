#!/usr/bin/env python
# $Id$
# -*- coding: utf-8

import tempfile
import numpy
import h5py
from pyscf import lib
from pyscf.lib import logger
import pyscf.ao2mo
from pyscf.ao2mo import r_outcore

# JCP, 139, 014108

#-------
# DC
#-------
def ao2mo(mf, mo_coeff, erifile):
    n4c, nmo = mo_coeff.shape
    n2c = n4c//2
    c = mf.mol.light_speed
    ol = mo_coeff[:n2c]
    os = mo_coeff[n2c:] * (.5/c)

    def run(mos, intor, dst, bound, blksize):
        logger.debug(mf, 'blksize = %d', blksize)
        r_outcore.general(mf.mol, mos, erifile,
                          dataname='tmp', intor=intor,
                          verbose=mf.verbose)
        with h5py.File(erifile) as feri:
            for i0, i1 in prange(0, bound, blksize):
                logger.debug(mf, 'load %s %d', intor, i0)
                buf = feri[dst][i0:i1]
                # Add the contribution from 'tmp' to erifile
		buf += feri['tmp'][i0:i1]
                feri[dst][i0:i1] = buf
                buf = None

    r_outcore.general(mf.mol, (ol,ol,ol,ol), erifile,
                      dataname='ericas', intor='cint2e',
                      verbose=mf.verbose)
    blksize = max(1, int(512e6/16/(nmo**3))) * nmo
    run((os,os,os,os), 'cint2e_spsp1spsp2', 'ericas', nmo*nmo, blksize)
    run((os,os,ol,ol), 'cint2e_spsp1'     , 'ericas', nmo*nmo, blksize)
    run((ol,ol,os,os), 'cint2e_spsp2'     , 'ericas', nmo*nmo, blksize)
    return 0

#-------
# DCG
#-------
def ao2mo_gaunt(mf, mo_coeff, erifile):
    ao2mo(mf, mo_coeff, erifile)

    n4c, nmo = mo_coeff.shape
    n2c = n4c//2
    c = mf.mol.light_speed
    ol = mo_coeff[:n2c]
    os = mo_coeff[n2c:] * (.5/c)

    def run(mos, intor, dst, bound, blksize):
        logger.debug(mf, 'blksize = %d', blksize)
        r_outcore.general(mf.mol, mos, erifile,
                          dataname='tmp', intor=intor,
                          aosym='s1', verbose=mf.verbose)
        with h5py.File(erifile) as feri:
            for i0, i1 in prange(0, bound, blksize):
                logger.debug(mf, 'load %s %d', intor, i0)
                buf = feri[dst][i0:i1]
                buf -= feri['tmp'][i0:i1]
                feri[dst][i0:i1] = buf
                buf = None

    blksize = max(1, int(512e6/16/(nmo**3))) * nmo
    run((ol,os,ol,os), 'cint2e_ssp1ssp2', 'ericas', nmo*nmo, blksize)
    run((ol,os,os,ol), 'cint2e_ssp1sps2', 'ericas', nmo*nmo, blksize)
    run((os,ol,ol,os), 'cint2e_sps1ssp2', 'ericas', nmo*nmo, blksize)
    run((os,ol,os,ol), 'cint2e_sps1sps2', 'ericas', nmo*nmo, blksize)
    return 0

#-------
# DCB
#-------
def ao2mo_breit(mf, mo_coeff, erifile):
    ao2mo(mf, mo_coeff, erifile)

    n4c, nmo = mo_coeff.shape
    n2c = n4c // 2
    c = mf.mol.light_speed
    ol = mo_coeff[:n2c]
    os = mo_coeff[n2c:] * (.5/c)

    def run(mos, intor, dst, bound, blksize):
        logger.debug(mf, 'blksize = %d', blksize)
        r_outcore.general(mf.mol, mos, erifile,
                          dataname='tmp', intor=intor,
                          aosym='s1', verbose=mf.verbose)
        with h5py.File(erifile) as feri:
            for i0, i1 in prange(0, bound, blksize):
                logger.debug(mf, 'load %s %d', intor, i0)
                buf = feri[dst][i0:i1]
                buf += feri['tmp'][i0:i1]
                feri[dst][i0:i1] = buf
                buf = None

    blksize = max(1, int(512e6/16/(nmo**3))) * nmo
    run((ol,os,ol,os), 'cint2e_breit_ssp1ssp2', 'ericas', nmo*nmo, blksize)
    run((ol,os,os,ol), 'cint2e_breit_ssp1sps2', 'ericas', nmo*nmo, blksize)
    run((os,ol,ol,os), 'cint2e_breit_sps1ssp2', 'ericas', nmo*nmo, blksize)
    run((os,ol,os,ol), 'cint2e_breit_sps1sps2', 'ericas', nmo*nmo, blksize)
    return 0

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)

if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    from pyscf import mp
    mol = gto.M(
        verbose = 1,
        #output = 'out_h2o',
        atom = '''
        o    0   0   0
        o    0   0   1.20752
        ''',
        basis = {'H': gto.uncontract_basis(gto.basis.load('321g', 'H')),
                 'O': gto.uncontract_basis(gto.basis.load('321g', 'O')),},
        #light_speed = 10
    )

#----------------------------------------------
# DC
#----------------------------------------------
    mf = scf.DHF(mol)
    mf.conv_tol = 1e-14
    print(mf.scf())  # -75.640153125

    _tmpfile = tempfile.NamedTemporaryFile()
    erifile = _tmpfile.name
    nmo = mf.mo_coeff.shape[1]
    
    casorb = mf.mo_coeff[:,nmo//2+4:nmo//2+8]
    ao2mo(mf, casorb, erifile)
    with h5py.File(erifile) as f1:
        print f1['ericas'].shape

#----------------------------------------------
# DCG
#----------------------------------------------
#    mf = scf.DHF(mol)
#    mf.conv_tol = 1e-12
#    mf.with_gaunt = True
#    print(mf.scf()) # -75.6324876552
#
#    _tmpfile = tempfile.NamedTemporaryFile()
#    erifile = _tmpfile.name
#    nmo = mf.mo_coeff.shape[1]
#    casorb = mf.mo_coeff[:,nmo//2+4:nmo//2+8]
#    ao2mo_gaunt(mf, casorb, erifile)
#    with h5py.File(erifile) as f1:
#        print f1['ericas'].shape
#
#----------------------------------------------
# DCB
#----------------------------------------------
#    mf = scf.DHF(mol)
#    mf.conv_tol = 1e-10
#    mf.with_breit = True
#    print(mf.scf())
#
#    _tmpfile = tempfile.NamedTemporaryFile()
#    erifile = _tmpfile.name
#    nmo = mf.mo_coeff.shape[1]
#    casorb = mf.mo_coeff[:,nmo//2+4:nmo//2+8]
#    ao2mo_breit(mf, casorb, erifile)
#    with h5py.File(erifile) as f1:
#        print f1['ericas'].shape
#
