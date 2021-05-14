NEW: differentiation w.r.t. covariance is now supported!

Installation is also simplified using SciPy.

Torch-MvNorm
--------------------------------------------------------------------------------

Torch-MvNorm is a small Python package that allows

- Multivariate normal density integration, in particular computing cumulative distribution functions (CDFs)
- Partial differentiaton of CDFs through implementation of closed-form formulas (see e.g. [Marmin et al. 2019](https://hal.archives-ouvertes.fr/hal-01133220v4/document), appendix 6)
- Quantities manipulation within PyTorch tensor-based framework


---

- [About Torch-MvNorm](#about-torch-mvnorm)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Communication and contribution](#communication-and-contribution)



## About Torch-MvNorm

Torch-MvNorm is a library that consists of the two following components:

- **integration** -- PyTorch-Fortan bridge for [Alan Genz's routine](http://www.math.wsu.edu/faculty/genz/software/fort77/mvndstpack.f) using SciPy.
- **multivariate_normal_cdf** -- implementation of the formula of the multivariate normal CDF gradient, with respect to location and covariance.


## Installation


### Dependencies

- [Install PyTorch](https://pytorch.org/get-started/locally/)

- Install joblib python module, e.g.
```
sudo apt-get install -y python3-joblib
```

- Install SciPy python module.

### Get the Torch-MvNorm source
```bash
git clone --recursive https://github.com/SebastienMarmin/torch-mvnorm
cd torch-mvnorm
```

### Test the code
```
python3 tests/test_parallele.py
```

## Getting Started

- Run the code on [small examples](https://github.com/SebastienMarmin/torch-mvnorm/blob/master/tests).
- Have a look at [the documentation](https://sebastienmarmin.github.io/torch-mvnorm/). TODO updates.

## Communication and contribution

I welcome all contributions. Please let me know if you encounter a bug by [filing an issue](https://github.com/SebastienMarmin/torch-mvnorm/issues).
Feel free to request a feature, make suggestions, share thoughts, etc, using the GitHub plateform or [contacting me](mailto:marmin-public@mailbox.org).

If you came across this work for a publication, please considere citing me for the code or [for the mathematical derivation](https://github.com/SebastienMarmin/torch-mvnorm/blob/master/bib.bib).

## License

Torch-MvNorm is under GNU General Public License. See the LICENSE file.
