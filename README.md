# Parallel Fock-Space DMRG algorithm based on MPO #

Basic ideas:                                        

 1. The central quantity is the three indexed operators in MPO format.

 2. Parallelization is achieved at the operator level.

Supports:
 
 1. Ab inito DMRG with/without particle number and Sz symmetry

 2. Spin-projected DMRG with S^2 symmetry

 3. Conversion form spin-projected MPS to spin-adapted MPS used in the BLOCK code

Reference:

 Spin-Projected Matrix Product States: Versatile Tool for Strongly Correlated Systems
 
 Z. Li and G.K.-L. Chan, J. Chem. Theory Comput., 2017, 13 (6), pp 2681–2695
