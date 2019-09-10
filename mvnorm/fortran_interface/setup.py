# advice command : python3 setup.py build_ext --inplace

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
# This line only needed if building with NumPy in Cython file.
from numpy import get_include
from os import system, mkdir # only for creating temp folder


temp_path = "./temp/"
try:
      mkdir(temp_path)
except OSError:
      pass
      
  
  

# compile the fortran modules without linking
fortran_mod_comp = 'gfortran ./external_source/mvtdst.f -c -o ./temp/gfunc.o -O3 -fPIC -ffixed-form'
print (fortran_mod_comp)
system(fortran_mod_comp)
shared_obj_comp = 'gfortran ./compilation/pygfunc.f90 -c -o ./temp/pygfunc.o -O3 -fPIC'
print (shared_obj_comp)
system(shared_obj_comp)



ext_modules = [Extension(# module name:
                         'pygfunc',
                         # source file:
                         ['./compilation/pygfunc.pyx'],
                         # other compile args for gcc
                         extra_compile_args=['-fPIC', '-O3'],
                         # other files to link to
                         extra_link_args=['./temp/gfunc.o', './temp/pygfunc.o'])]

setup(name = 'pygfunc',
      cmdclass = {'build_ext': build_ext},
      # Needed if building with NumPy.
      # This includes the NumPy headers when compiling.
      include_dirs = [get_include()],
      ext_modules = ext_modules)
