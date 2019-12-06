Torch-MvNorm
--------------------------------------------------------------------------------

Torch-MvNorm is a small Python package that allows

- Multivariate normal density integration, in particular computing cumulative distribution functions (CDFs)
- Partial differentiaton of CDFs through implementation of closed-form formulas (see [Marmin et al. 2019](https://hal.archives-ouvertes.fr/hal-01133220v4/document), appendix 6)
- Quantities manipulation within PyTorch tensor-based framework

---

- [About Torch-MvNorm](#about-torch-mvnorm)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Communication and contribution](#communication-and-contribution)



## About Torch-MvNorm

Torch-MvNorm is a library that consists of the two following components:

- **mvnorm.fotran_interface** -- PyTorch-Fortan bridge for [Alan Genz's routine](http://www.math.wsu.edu/faculty/genz/software/fort77/mvndstpack.f)
- **mvnorm.autograd** -- implementation of the formula of the multivariate normal CDF gradient

Torch-MvNorm can be used as an engineering or research tool for optimizing functions that requires multivariate normal CDF, for example in machine learning.

Torch-MvNorm has minimal overhead as it integrate PyTorch autodiff framework.


## Installation


### Dependencies

- [Install PyTorch](https://pytorch.org/get-started/locally/)

- Install gfortran and python-dev
```
sudo apt-get install gfortran
sudo apt-get install python-dev
```

- Install joblib and Cython python modules
```
python3 -m pip install joblib Cython
```

### Get the Torch-MvNorm source
```bash
git clone --recursive https://github.com/SebastienMarmin/torch-mvnorm
cd torch-mvnorm
```


### Install Torch-MvNorm

Compile Fortran and build the interface:
```
cd mvnorm/fortran_interface/
python3 setup.py build_ext --inplace
```

### Test the code
```
python3 tests/test_general.py
```


## Getting Started

- [Tests: run the code on small examples](https://github.com/SebastienMarmin/torch-mvnorm/blob/master/tests)
- Documentation is [here](https://sebastienmarmin.github.io/torch-mvnorm/).

## Communication and contribution

I welcome all contributions. Please let me know if you encounter a bug by [filing an issue](https://github.com/SebastienMarmin/torch-mvnorm/issues).
Feel free to request a feature, make suggestions, share thoughts, etc, using the GitHub plateform or [contacting me](mailto:marmin-public@mailbox.org).

If you came across this work for a publication, please considere citing me for the code or [for the mathematical derivation](https://github.com/SebastienMarmin/torch-mvnorm/blob/master/bib.bib).

## License

Torch-MvNorm is under GNU General Public License. See the LICENSE file.
