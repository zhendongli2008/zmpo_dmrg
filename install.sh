#
# The installation is very simple
#

echo '* Start compling qtensor utils ...'

mkdir -p libs
cp __init__.py libs

cd ctypes
gcc -fPIC -shared -g -O2 -o libqsym.so qsym.c
mv libqsym.so ../libs

cd ../f90subs
f2py -c -m libangular angular.F90
mv libangular.so ../libs

echo '* Start generation of codes for RDMs ...'

echo '* ZMPO_DMRG is successfully implemented!'
