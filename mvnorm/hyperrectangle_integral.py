from torch import Tensor, diag_embed, ones, zeros,broadcast_tensors, diagonal,float64, no_grad
import numpy as np
from numpy import Inf, int32, tril_indices
from mvnorm.fortran_interface.pygfunc import f as genz_bretz

def _broadcast(lower, upper, mean, mat):
    for param, name in zip((lower,upper,mean),("lower","upper","mean")):
        if param.dim() < 1:
            raise ValueError("Input "+name+" must be at least one-dimensional.")
    if mat.dim() < 2:
        raise ValueError("Input corr or sigma must be at least two-dimensional.")
    lower_,upper_,mean_,mat= broadcast_tensors(lower.unsqueeze(-1), upper.unsqueeze(-1), mean.unsqueeze(-1), mat)
    lower = lower_[..., 0]
    upper = upper_[..., 0]
    mean = mean_[..., 0]
    return lower,upper,mean,mat


def hyperrectangle_integral(lower = None, upper = None, mean = None, corr = None, sigma = None, scale_tril=None,
                                        maxpts = 25000, abseps = 0.001, releps = 0,info=True):
    # Note: the Fortran algorithm is based on the correlation matrix.
    # If the user provide "scale_tril", this function will calculate the corresponding correlation matrix.
    with no_grad():
        if (corr is not None) + (sigma is not None) + (scale_tril is not None) != 1:
            raise ValueError("Exactly one of corr, sigma or scale_tril may be specified.")
        if scale_tril is None:
            mat,is_corr  = (corr, True) if sigma is None else (sigma,False)
        else:
            mat  = torch.matmul(scale_tril,scale_tril.transpose(-1,-2))
            is_corr = False
        device, dtype = mat.device, mat.dtype
        if lower is None:
            lower = Tensor([-Inf],device=device).type(dtype)
        else:
            if isinstance(lower,(int,float)):
                lower = Tensor([float(lower)],device=device).type(dtype)
        if upper is None:
            upper = Tensor([ Inf],device=device).type(dtype)
        else:
            if isinstance(upper,(int,float)):
                upper = Tensor([float(upper)],device=device).type(dtype)
        if mean is None:
            mean = zeros(1,device=device,dtype=dtype)
        lower,upper,mean,mat = _broadcast(lower, upper, mean, mat)
        d = lower.size(-1)
        if (d > 1000):
            raise ValueError("Only dimensions below 1000 are allowed. Got "+str(d)+".")
        # keeping mean is unecessary, centralize the problem
        lower_c = lower - mean
        upper_c = upper - mean
        if mat.size(-2)==1: # diagonal (or univariate)
            pass
        else:
            if is_corr:
                l = lower_c
                u = upper_c
                c = mat
            else: # operate standardization
                stds = diagonal(sigma,dim1=-2,dim2=-1).sqrt()
                l = lower_c/stds
                u = upper_c/stds
                c = (mat/stds.unsqueeze(-1))/stds.unsqueeze(-2)
        infin = np.array(tuple(2 for i in range(d))).astype(int32) 
        # infin is a int vector to pass to the fortran code controlling the integral limits
        # infin is a int vector to pass to the fortran code controlling the integral limits
        #            if INFIN(I) < 0, Ith limits are (-infinity, infinity);
        #            if INFIN(I) = 0, Ith limits are (-infinity, UPPER(I)];
        #            if INFIN(I) = 1, Ith limits are [LOWER(I), infinity);
        #            if INFIN(I) = 2, Ith limits are [LOWER(I), UPPER(I)].
        infl = lower==-Inf 
        infu = upper==Inf
        infin[infl] = 0
        infin[infu] = 1
        infin[infl*infu] = -1 # basically ignores these componenents
        # now that this info is stored, we get rid of the numpy.Inf's (they are user-friendly but not understood in Fortran)
        l[infl] = 0
        u[infu] = 0
        # let Cython finish the pass to the Fortran routine.
        if l.dim()>1:
            raise NotImplementedError("Batch dimensions are not supported. For batch computation, use multivariate_normal_cdf instead.")
        return genz_bretz(d,l.type(float64).numpy(),u.type(float64).numpy(),infin,c.type(float64).numpy()[tril_indices(d,-1)],maxpts,abseps,releps,1 if info else 0)



