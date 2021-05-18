from itertools import zip_longest
from torch import broadcast_to
from .Phi import Phi

def broadcast_shape(a,b):
    res = reversed(tuple(i if j==1 else (j if i==1 else (i if i==j else -1)) for i,j in zip_longest(reversed(a),reversed(b),fillvalue=1)))
    return list(res)

def multivariate_normal_cdf(value,loc=None,covariance_matrix=None):
    """Compute rectangle probabilities for a multivariate normal random vector Z 
     ``P(Z_i < values_i, i = 1,...,d)``. 
    Probability values can be returned with closed-form backward derivatives.
    
    Parameters
    ----------
    value : torch.Tensor, optional
        upper integration limits.  Can have batch shape. The
        last dimension is the dimension of the random vector.
        Default is ``None`` which is understood as minus infinity
        for all components. Values ``- numpy.Inf`` are supported, e.g.
        if only few components have an infinite boundary. The last 
        dimension must be equal to d, the dimension of the Gaussian
        vector.
    loc : torch.Tensor, optional
        Mean of the Gaussian vector. Default is zeros. Can have batch
        shape. last dimension must be equal to d, the dimension of the
        Gaussian vector.
    covariance_matrix : torch.Tensor, optional
        Covariance matrix of the Gaussian vector. Must be provided.
        Can have batch shape. The two last dimensions must be equal to d,
        the dimension of the Gaussian vector.
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
    well as the returns `value`, `error` and `info` are broadcasted to their
    common batch shape. See PyTorch' `broadcasting semantics
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

    if loc is None:
        loc = 0.0
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