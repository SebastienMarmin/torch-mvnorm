from itertools import zip_longest
from torch import broadcast_to, ones, erfc
import torch
from .Phi import Phi

def broadcast_shape(a,b):
    res = reversed(tuple(i if j==1 else (j if i==1 else (i if i==j else -1)) for i,j in zip_longest(reversed(a),reversed(b),fillvalue=1)))
    return list(res)

def Phi1D(z):
    return erfc(-z*0.70710678118654746171500846685)/2
    #                  ^ sqrt(2)/2

def multivariate_normal_cdf(value,loc=0.0,covariance_matrix=None,cov_diagonal=False):
    """Compute orthant probabilities for a multivariate normal random vector Z 
     ``P(Z_i < value_i, i = 1,...,d)``. 
    Probability values can be returned with closed-form backward derivatives.
    
    Parameters
    ----------
    value : torch.Tensor,
        upper integration limits. Can have batch shape.
        The last dimension must be equal to d, the dimension of the
        Gaussian vector.
    loc : torch.Tensor, optional
        Mean of the Gaussian vector. Default is zeros. Can have batch
        shape. Last dimension must be equal to d, the dimension of the
        Gaussian vector.
    covariance_matrix : torch.Tensor, optional
        Covariance matrix of the Gaussian vector.
        Can have batch shape. The two last dimensions must be equal to d,
        the dimension of the Gaussian vector. Identity matrix by default.
        If the covariance is diagonal, `cov_diagonal` can be set to
        ``True`` and `covariance_matrix` must have the same shape as 
        `value` (or be broadcastable to `value`).
    cov_diagonal=False : boolean, optional
        See `covariance_matrix`. Avoid expensive numerical integration.
    Returns
    -------
    value : torch.Tensor
        The probability of the event ``Y < value``, with
        ``Y`` a Gaussian vector defined by `loc` and `covariance_matrix`.
        Closed form derivative are implemented if `value`  `loc`,
        `covariance_matrix` require a gradient.
    Notes
    -------
    Parameters `value` and `covariance_matrix`, as 
    well as the returned probability tensor are broadcasted to their
    common batch shape. See PyTo    rch' `broadcasting semantics
    <https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics>`_.
    See the integration parameters. TODO
    Partial derivative are computed using closed form formula, see e.g. Marmin et al. [2]_, p 13.
    References
    ----------
    .. [1] Alan Genz and Frank Bretz, "Comparison of Methods for the Computation of Multivariate 
       t-Probabilities", Journal of Computational and Graphical Statistics 11, pp. 950-971, 2002. `Source code <http://www.math.wsu.edu/faculty/genz/software/fort77/mvtdstpack.f>`_.
    .. [2] Sébastien Marmin, Clément Chevalier and David Ginsbourger, "Differentiating the multipoint Expected Improvement for optimal batch design", International Workshop on Machine learning, Optimization and big Data, Taormina, Italy, 2015. `PDF <https://hal.archives-ouvertes.fr/hal-01133220v4/document>`_.
    Examples
    --------
    >>> import torch
    >>> from torch.autograd import grad
    >>> n = 4
    >>> x = 1 + torch.randn(n)
    >>> x.requires_grad = True
    >>> # Make a positive semi-definite matrix
    >>> A = torch.randn(n,n)
    >>> C = 1/n*torch.matmul(A,A.t())
    >>> p = mvnorm.multivariate_normal_cdf(x,covariance_matrix=C)
    >>> p
    tensor(0.3721, grad_fn=<MultivariateNormalCDFBackward>)
    >>> grad(p,(x,))
    >>> (tensor([0.0085, 0.2510, 0.1272, 0.0332]),)
    """
    x_shape = value.shape
    d = x_shape[-1]
    if covariance_matrix is None:
        covariance_matrix = ones_like(value)
        cov_diagonal = True
    if cov_diagonal:
        z = (value-loc)/covariance_matrix.sqrt()
        return Phi1D(z).prod(-1)
    cov_shape = covariance_matrix.shape[-2:]
    if len(cov_shape) < 2:
        raise ValueError("covariance_matrix must have at least \
                          two dimensions.")
    d = cov_shape[-1]
    if cov_shape[-2] != d:
        raise ValueError("Covariance matrix must have the last two \
            dimension equal to d. Here its "+str(cov_shape[-2:]))
    m = loc-value # actually do P(Y-value<0)
    batch_shape = broadcast_shape(m.shape[:-1],cov_shape[:-2])
    vector_shape = batch_shape + [d]
    matrix_shape = batch_shape + [d,d]
    m_b = broadcast_to(m,vector_shape)
    c_b = broadcast_to(covariance_matrix,matrix_shape)
    return Phi(m_b,c_b)