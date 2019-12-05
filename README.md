Torch-MvNorm
--------------------------------------------------------------------------------

Torch-MvNorm is a small Python package that allows:

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

- **mvnorm.fotran_interface** -- bridge PyTorch-Fortan for [Alan Genz's routine](http://www.math.wsu.edu/faculty/genz/software/fort77/mvndstpack.f)
- **mvnorm.autograd** -- implementation of the formula of the multivariate normal CDF gradient

Torch-MvNorm can be used as an engineering or research tool for optimizing functions that requires multivariate normal CDF, for example in machine learning.

Torch-MvNorm has minimal overhead as it integrate PyTorch autodiff framework.


## Installation

### Dependencies


```
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing
```


### Get the Torch-MvNorm source
```bash
git clone --recursive https://github.com/SebastienMarmin/torch-mvnorm
cd torch-mvnorm
```

### Install Torch-MvNorm
On Linux
```bash
TODO
```


## Getting Started

- [Tests: run the code on small examples](https://github.com/SebastienMarmin/torch-mvnorm/tests)
- Documentation is [here](https://sebastienmarmin.github.io/torch-mvnorm/).

## Communication and contribution

I welcome all contributions. Please let me know if you encounter a bug by [filing an issue](https://github.com/SebastienMarmin/torch-mvnorm/issues).
Feel free to request a feature, make suggestions, share thoughts, etc, using the GitHub plateform or [contacting me](mailto:marmin-public@mailbox.org).

If you came across this work for a publication, please considere citing me for the code or [for the mathematical derivation](https://github.com/SebastienMarmin/torch-mvnorm/bib.bib).

## License

Torch-MvNorm is under GNU General Public License. See the LICENSE file.
