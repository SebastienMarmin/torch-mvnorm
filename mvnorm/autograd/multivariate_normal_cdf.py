# -*- coding: utf-8 -*-
from .conditioning import one_component_conditioning, two_components_conditioning
from numpy import array, zeros as np_zeros, int32, tril_indices, float64, broadcast_to
from torch import tensor, diagonal, cat, unbind, ones as torch_ones, zeros as torch_zeros, ones_like, tril_indices as torch_tril_indices, triu_indices as torch_triu_indices,tril,triu,diag_embed # onely for dev
from torch.autograd import Function, grad # only for test

from mvnorm import genz_bretz
from mvnorm.hyperrectangle_integral import hyperrectangle_integral

from torch import tensor, diagonal, cat, unbind, zeros as torch_zeros, Tensor, Size, matmul,int32
from torch.distributions import MultivariateNormal

from torch.autograd import Function
from numpy import Inf
from ..parallel.joblib import _hyperrectangle_integration




sqrt5 = 2.2360679774997898050514777424
def phi(z,s):
    return 0.39894228040143270286321808271/s*(-z**2/(2*s**2)).exp()
#                 ^ oneOverSqrt2pi     
 
def phi2_sub(z,C): # compute pairs of bivariate densities and organize them in a matrix
    V = diagonal(C,dim1=-2,dim2=-1)
    a,c = V.unsqueeze(-1),V.unsqueeze(-2)
    det = a*c-C**2
    x1 = z.unsqueeze(-1)
    x2 = z.unsqueeze(-2)
    exponent = -0.5/det*(c*x1**2+a*x2**2-2*C*x1*x2)
    res = 0.15915494309189534560822210096/det.sqrt()*(exponent).exp()
    #           ^ oneOver2pi     
    return tril(res,-1) + triu(res,1)
 

def _cov2cor(a,b,mat,stds):
    a_r = None if a is None else a/stds
    b_r = None if b is None else b/stds
    mat_r   = ((mat/stds.unsqueeze(-1))/stds.unsqueeze(-2))
    return a_r, b_r, mat_r


class MultivariateNormalCDF(Function):
    """
    Array with associated photographic information.

    ...

    Attributes
    ----------
    exposure : float
        Exposure in seconds.

    Methods
    -------
    colorspace(c='rgb')
        Represent the photo in the given colorspace.
    gamma(n=1.0)
        Change the photo's gamma exposure.

    """
    @staticmethod
    def forward(ctx, val,c,maxpts,abseps,releps):
        # input infos
        ctx.maxpts   = maxpts
        ctx.abseps   = abseps
        ctx.releps   = releps
        ctx.save_for_backward(val,c) # m.data is not needed for gradient computation
        stds = diagonal(c,dim1=-2,dim2=-1).sqrt()
        ctx.stds = stds
        val_rescaled,_, corr = _cov2cor(val,None,c,stds)
        out = _hyperrectangle_integration(None,val_rescaled,corr,maxpts,abseps,releps,False)
        return out 

    @staticmethod
    def backward(ctx, grad_output):
        val,c = ctx.saved_tensors
        d = val.size(-1)
        grad_x = grad_c = None
        need_x, need_c = ctx.needs_input_grad[:2]
        p = phi(val,ctx.stds)
        var = ctx.stds**2
        if need_c:
            m_cond,c_cond,m_cond2,c_cond2= two_components_conditioning(c,x=None,m=val,var = var,cov2cor=True)
        else:
            m_cond,c_cond = one_component_conditioning(c,m = val, var = var,cov2cor=True)
        #m_c_l = unbind(m_cond,-2)
        #c_cond = cat(tuple(c_c_l[i].unsqueeze(-3) for i in range(d)),-3)
        #P_l= (_parallel_CDF(m_c_l[i],c_c_l[i],ctx.maxpts,ctx.abseps,ctx.releps).unsqueeze(-1) for i in range(d))
        P = _hyperrectangle_integration(None,m_cond,c_cond,ctx.maxpts,ctx.abseps,ctx.releps,False)
        #P_l= (CDFapp(m_c_l[i],c_c_l[i],ctx.maxpts,ctx.abseps,ctx.releps).unsqueeze(-1) for i in range(d))
        #P = cat(tuple(P_l),-1)
        res = grad_output.unsqueeze(-1)*P*p
        if need_x:
            grad_x = res
        if need_c:
            raise NotImplementedError("Differentiation w.r.t. cova is not yet implemented")
        return grad_x, grad_c, None, None, None
       
# Create functional linked with autograd machinerie.
CDFapp = MultivariateNormalCDF.apply

def multivariate_normal_cdf(lower=None,upper=None,loc=None,covariance_matrix=None,scale_tril=None,method="GenzBretz",nmc=200, maxpts = 25000, abseps = 0.001, releps = 0, error_info = False):
    """Compute probabilities by integrating the multivariate normal density. Probability
    values can be returned with closed-form backward derivatives.
    

    Parameters
    ----------
    lower : torch.Tensor, optional
        Lower integration limits.  Can have batch shape. with the
        last dimension is the dimension of the random vector.
        Default is ``None`` which is understood as minus infinity
        for all components. Values ``numpy.Inf`` are supported.
    upper : torch.Tensor, optional
        Upper integration limits. See `lower`.
    loc : torch.Tensor, optional
        Mean of the Gaussian vector. Default is zeros.
    covariance_matrix : torch.Tensor, optional
        Covariance matrix of the Gaussian vector. Must be provided if 
        `scale_tril` is not.
    scale_tril : torch.Tensor, optional
        A lower triangular root of the covariance matrix of the Gaussian 
        vector. Must be provided if `scale_tril` is not.
    method : :obj: str, optional
        Method deployed for the integration. Either ``'MonteCarlo'`` or
        ``'GenzBretz'``.
    nmc : :obj: int, optional
        Number of Monte Carlo samples.
    maxpts :obj: int, optional
        Maximum number of integration points in the Fortran routine.
    abseps :obj: float, optional
        Absolute error tolerance.
    releps :obj: float, optional
        Relative error tolerance.
    error_info :obj: bool, optional
        Should an estimation of the integration error be returned.
        Not compatible with autograd.

    Returns
    -------
    value : torch.Tensor
        The probability of the event ``lower < Y < upper``, with
        ``Y`` a Gaussian vector defined by `loc` and `covariance_matrix`
        (or `scale_tril`).
        
    error : torch.Tensor
        The estimated error for each component of `value`. **Returned only 
        if** `error_info` is ``True``.
    info : torch.Tensor
        Tensor of type ``int32`` informing on the execution for each 
        component.
            - If ``0``, normal completion with ``error < abseps``
            - If ``1``, completion with ``error > abseps`` and (for 
              ``method =  'GenzBretz'``) all maxpts evaluation budget is
              depleted.
            - If ``2``, N > 1000 or N < 1 (only for ``method =  'GenzBretz'``)
            - If ``3``, `covariance_matrix` is not positive semi-definite (only for ``method = 'GenzBretz'``)
        **Returned only if** `error_info` is ``True``.


    Notes
    -------
    Parameters `lower`, `upper` and `covariance_matrix` (or `scale_tril`), as 
    well as the returns `value`, `error` and `info` are broadcasted to their
    common batch shape. See PyTorch' `broadcasting semantics
    <https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics>`_.

    If any component of ``lower - upper`` is negative, the function returns a null
    tensor with consistent shape.

    Method ``MonteCarlo`` uses Monte Carlo sampling for estimating `value`, whereas
    ``method = 'GenzBretz'`` call a Fortran routine [1]_.

    If `method` is ``MonteCarlo`` a Cholesky decomposition of the covariance matrix
    will be performed. Else if ``GenzBretz``, only the correlation matrix is 
    computed and passed to the Fortran routine.

    The parameter `maxpts` can be used to limit the time. A suggested calibration
    strategy is to start with 1000 times the integration dimension, and then
    increase it if the returned `error` is too large.

    Partial derivative are computed using non-trivial closed form formula.
    See [2]_ for a derivation.

    Using autograd requires that `upper` or `lower` have all its components infinite
    (or unspecified).

    References
    ----------

    .. [1] Alan Genz and Frank Bretz, "Comparison of Methods for the Computation of Multivariate 
       t-Probabilities", Journal of Computational and Graphical Statistics 11, pp. 950-971, 2002. `Source code <http://www.math.wsu.edu/faculty/genz/software/fort77/mvtdstpack.f>`_.

    .. [2] Sébastien Marmin, Clément Chevalier and David Ginsbourger, "Differentiating the multipoint Expected Improvement for optimal batch design", International Workshop on Machine learning, Optimization and big Data, Taormina, Italy, 2015. `Link <https://hal.archives-ouvertes.fr/hal-01133220v4/document>`_.


    Examples
    --------

    >>> import torch
    >>> n = 4
    >>> x = 1 + torch.randn(n)
    >>> # Make a positive semi-definite matrix
    >>> A = torch.randn(n,n)
    >>> C = 1/n*torch.matmul(A,A.t())
    >>> mvnorm.multivariate_normal_cdf(upper=x,covariance_matrix=C)
    tensor(0.1349)
    
    """
    if (covariance_matrix is not None) + (scale_tril is not None) != 1:
        raise ValueError("Exactly one of covariance_matrix or scale_tril may be specified.")
    mat  = scale_tril if covariance_matrix is None else covariance_matrix
    device, dtype = mat.device, mat.dtype
    d = mat.size(-1)  
    if isinstance(lower,(int,float)):
        lower = Tensor.new_full((d,), float(lower), dtype=dtype, device=device)
    if isinstance(upper,(int,float)):
        upper = Tensor.new_full((d,), float(upper), dtype=dtype, device=device)
    lnone = lower is None
    unone = upper is None
    if not lnone and lower.max()==-Inf:
        lower=None
        lnone = True
    if not unone and upper.min()== Inf:
        upper=None
        unone = True

    if method=="MonteCarlo": # Monte Carlo estimation
        if loc is None:
            loc = torch_zeros(d,device=device,dtype=dtype)
        p = MultivariateNormal(loc=loc,scale_tril=scale_tril,covariance_matrix=covariance_matrix)
        r = nmc%5
        N = nmc if r==0 else nmc + 5 - r # rounded to the upper multiple of 5
        Y = p.sample(Size([N]))
        if lnone and unone:
            error = torch_zeros(p.batch_shape,device=device,dtype=dtype) if error_info else -1
            info  = torch_zeros(p.batch_shape,device=device,dtype=int32) if error_info else -1
            value = torch_ones(p.batch_shape,device=device,dtype=dtype)
            
        else:
            if lnone:
                Z = (Y<upper).prod(-1)
            else:
                Z = (Y>lower).prod(-1) if unone else (Y<upper).prod(-1)*(Y>lower).prod(-1)
            if error_info: # Does NOT slow down significatively
                booleans = Z.view(N//5,5,*Z.shape[1:]) # divide in 5 groups to have an idea of the precision 
                values   = ((booleans.sum(0).type(dtype)))/N*5
                value = values.mean(0)
                std = values.var(0).sqrt()
                error = 1.96 * std / sqrt5 # at 95 %
                info = (error > abseps).type(int32)
            else:
                value = Z.sum(0).type(dtype)/N
                error = info = -1
    elif method == "GenzBretz": # Fortran routine
        if (d > 1000):
            raise ValueError("Only dimensions below 1000 are allowed. Got "+str(d)+".")
        # centralize the problem
        uppe = upper if loc is None else None if unone else upper - loc
        lowe = lower if loc is None else None if lnone else lower - loc
        
        c = matmul(scale_tril,scale_tril.transpose(-1,-2)) if covariance_matrix is None else covariance_matrix
        if (not unone and uppe.requires_grad) or (not lnone and lowe.requires_grad) or mat.requires_grad:
            if error_info:
                raise ValueError("Option 'error_info' is True, and one of x, loc, covariance_matrix or scale_tril requires gradient. With option 'GenzBretz', the estimation of CDF error is not compatible with autograd.")
            error = info = -1
            if lnone:
                upp = uppe
            elif unone:
                upp = -lowe
            else:    
                raise ValueError("For autograd with option 'GenzBretz', at least lower or upper should be None (or with all components infinite).")
            value =  CDFapp(upp,c,maxpts,abseps,releps)
        else:
            if lnone and unone:
                value = torch_ones(c.shape[:-2],device=device,dtype=dtype)
                error = torch_zeros(c.shape[:-2],device=device,dtype=dtype) if error_info else -1
                info = torch_zeros(c.shape[:-2],device=device,dtype=int32) if error_info else -1
            else:
                stds = diagonal(c,dim1=-2,dim2=-1).sqrt()
                low,upp, corr = _cov2cor(lowe,uppe,c,stds)
                res = _hyperrectangle_integration(low,upp,corr,maxpts,abseps,releps,info=error_info)
                value, error, info = (res if error_info else (res,-1,-1))
    else:
        raise ValueError("The 'method=' should be either 'GenzBretz' or 'MonteCarlo'.")

    #if error_info and error > abseps:
    #        warn("Estimated error is higher than abseps. Consider raising the computation budget (nmc for method='MonteCarlo' or maxpts for 'GenzBretz'). Switch 'error_info' to False to ignore.")
    if error_info:
        return value, error, info
    else:
        return value



"""
Q_l = _parallel_CDF(m_cond2,c_cond2,ctx.maxpts,ctx.abseps,ctx.releps)
Q = torch_zeros(*Q_l.shape[:-1],d,d)
trilind = torch_tril_indices(d,d,offset=-1)
triuind = torch_triu_indices(d,d,offset =1)
Q[...,trilind[0],trilind[1]] = Q_l
Q[...,triuind[0],triuind[1]] = Q_l
q = phi2_sub(val,c)
hess = q*Q
D = -(val*res+(hess*c).sum(-1))/var
grad_c = hess + diag_embed(D)
"""