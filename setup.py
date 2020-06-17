# advice command : python3 setup.py build_ext --inplace
from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
# This line only needed if building with NumPy in Cython file.
from numpy import get_include
from os import system, mkdir # only for creating temp folder


build_path = "./mvnorm/fortran_interface/build/"
try:
      mkdir(build_path)
except OSError:
      pass


# compile the fortran modules without linking
fortran_mod_comp = 'gfortran ./mvnorm/fortran_interface/external_source/mvtdst.f -c -o ./mvnorm/fortran_interface/build/gfunc.o -O3 -fPIC -ffixed-form'
print (fortran_mod_comp)
system(fortran_mod_comp)
shared_obj_comp = 'gfortran ./mvnorm/fortran_interface/compilation/pygfunc.f90 -c -o ./mvnorm/fortran_interface/build/pygfunc.o -O3 -fPIC'
print (shared_obj_comp)
system(shared_obj_comp)


ext_modules = [Extension(# module name:
                         'mvnorm.fortran_interface.pygfunc',
                         # source file:
                         ['./mvnorm/fortran_interface/compilation/pygfunc.pyx'],
                         # other compile args for gcc
                         extra_compile_args=['-fPIC', '-O3'],
                         # other files to link to
                         extra_link_args=['./mvnorm/fortran_interface/build/gfunc.o',
                                 './mvnorm/fortran_interface/build/pygfunc.o'])]

setup(
    name = 'mvnorm',
    version="0.1",
    packages=find_packages(),
    cmdclass = {'build_ext': build_ext},
    # Needed if building with NumPy.
    # This includes the NumPy headers when compiling.
    include_dirs = [get_include()],
    ext_modules = ext_modules)
