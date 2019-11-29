[comment]: <>  (![PyTorch Logo](https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png))

Torch-MvNorm
--------------------------------------------------------------------------------

Torch-MvNorm is a small Python package that allows:

- Multivariate normal density integration, in particular computing cumulative distribution functions (CDFs).
- Partial differentiaton of CDFs through implementation of closed-form formulas (see [Marmin et al. 2019](https://hal.archives-ouvertes.fr/hal-01133220v4/document), appendix 6)
- Quantities manipulation within PyTorch tensor-based framework.

- [About Torch-MvNorm](#about-torch-mvnorm)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Communication and contribution](#communication-and-contribution)



## About Torch-MvNorm

Torch-MvNorm is a library that consists of the following components:

| Component | Description |
| ---- | --- |
| **mvnorm.fotran_interface** | an interface pyTorch-fortan for using [Alan Genz's routine](http://www.math.wsu.edu/faculty/genz/software/fort77/mvndstpack.f) |
| **mvnorm.autograd** | an implementation of the closed-form formula of the gradient of a multivariate normal CDF |

Torch-MvNorm can be used as a machine learning research tool for optimizing functions that requires multivariate normal CDF.

Torch-MvNorm has minimal overhead as it integrate PyTorch autodiff framework.


## Installation

### Install Dependencies

TODO
```
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing
```


### Get the PyTorch Source
```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive
```

### Install Torch-MvNorm
On Linux
```bash
TODO
```


## Getting Started

- TODO some code example
- [Tests: run the code on small examples](https://github.com/TODO)
[comment]: <>  (- [The API Reference](https://TODO))


## Communication and contribution

We welcome all contributions. Please let us know if you encounter a bug by [filing an issue](https://github.com/TODO).
Feel free to request a feature, make suggestions, share thoughts, etc, using the GitHub plateform or [contacting me](mailto:marmin-public@mailbox.org).


## License

Torch-MvNorm is under GNU General Public License. See the LICENSE file.
