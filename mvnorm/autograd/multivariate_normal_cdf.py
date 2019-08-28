
from .conditioning import one_component_conditioning

from torch import tensor, diagonal, cat, unbind, zeros as torch_zeros, new_full
from torch.autograd import Function
from numpy import Inf
from ..parallel.joblib import _hyperrectangle_integration




def phi(z,s):
    return 0.39894228040143270286321808271/s*(-z**2/(2*s**2)).exp()
#                 ^ oneOverSqrt2pi     

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
        m_cond,c_c_l = one_component_conditioning(c,m = val, var = ctx.stds**2,cov2cor=True)
        m_c_l = unbind(m_cond,-2)
        P_l= (_hyperrectangle_integration(None,m_c_l[i],c_c_l[i],ctx.maxpts,ctx.abseps,ctx.releps,False).unsqueeze(-1) for i in range(d)) # TODO
        P = cat(tuple(P_l),-1)
        res = grad_output.unsqueeze(-1)*P*p
        if need_x:
            grad_x = res
        if need_c:
            raise NotImplementedError("Deriv w.r.t. cov is not implemented yet.")
        #if bias is not None and ctx.needs_input_grad[2]:
        return grad_x, grad_c, None, None, None
       

CDFapp = MultivariateNormalCDF.apply

def multivariate_normal_cdf(lower=None,upper=None,loc=None,covariance_matrix=None,scale_tril=None,method="GenzBretz",nmc=200, maxpts = 25000, abseps = 0.001, releps = 0, error_info = False):
    """Gets and prints the spreadsheet's header columns

    Parameters
    ----------
    file_loc : str
        The file location of the spreadsheet
    print_cols : bool, optional
        A flag used to print the columns to the console (default is False)

    Returns
    -------
    list
        a list of strings representing the header columns
    """
    if (covariance_matrix is not None) + (scale_tril is not None) != 1:
        raise ValueError("Exactly one of sigma or scale_tril may be specified.")
    mat  = scale_tril if covariance_matrix is None else covariance_matrix
    device, dtype = mat.device, mat.dtype
    d = mat.size(-1)  
    if isinstance(lower,(int,float)):
        lower = new_full((d,), float(lower), dtype=dtype, device=device)
    if isinstance(upper,(int,float)):
        upper = new_full((d,), float(upper), dtype=dtype, device=device)
    lnone = lower==None
    unone = upper==None
    if not lnone and lower.max()==-Inf:
        lower=None
        lnone = True
    if not unone and upper.min()== Inf:
        upper=None
        unone = True


    if method=="MonteCarlo": # Monte Carlo estimation
        if loc is None:
            loc = torch_zeros(d,device=device,dtype=dtype)
        info  = -1 # for consistancy with GenzBretz
        p = MultivariateNormal(loc=loc,scale_tril=scale_tril,covariance_matrix=covariance_matrix)
        r = nmc%5
        N = nmc if r==0 else nmc + 5 - r # rounded to the upper multiple of 5
        Y = p.sample(torch.Size([N]))
        if lnone and unone:
            error = torch_zeros(Y.shape[:-2],device=device,dtype=dtype) if error_info else -1
            value = torch_ones(Y.shape[:-2],device=device,dtype=dtype)
        else:
            if lnone:
                Z = (Y<upper).prod(-1)
            else:
                Z = (Y>lower).prod(-1) if unone else (Y<upper).prod(-1)*(Y>lower).prod(-1)
            if error_info: # Does NOT slow down significatively
                booleans = Z.view(*Z.shape[:-1],5,N//5) # divide in 5 groups to have an idea of the precision 
                values   = ((booleans.sum(-1).type(torch.float)))/N*5
                value = values.mean(-1)
                std = values.var(-1).sqrt()
                error = 1.96 * std / sqrt(5) # at 95 %
            else:
                value = Z.mean(-1)
                error = -1
    elif method == "GenzBretz": # Fortran routine
        if (d > 1000):
            raise ValueError("Only dimensions below 1000 are allowed. Got "+str(d)+".")
        if loc is not None: # centralize the problem
            uppe = None if unone else upper - loc
            lowe = None if lnone else lower - loc
        c = matmul(scale_tril,scale_tril.transose(-1,-2)) if covariance_matrix is None else covariance_matrix
        if error_info:
            if (not unone and uppe.requires_grad) or (not lnone and lowe.requires_grad) or mat.requires_grad:
                raise ValueError("Option 'error_info' is True, and one of x, loc, covariance_matrix or scale_tril requires gradient. With option 'GenzBretz', the estimation of CDF error is not compatible with autograd.")
            stds = diagonal(c,dim1=-2,dim2=-1).sqrt()
            low,upp, corr = _cov2cor(lowe,uppe,c,stds)
            value, error, info = _hyperrectangle_integration(low,upp,corr,maxpts,abseps,releps,info=True)
        else:
            error = -1
            info = -1
            if lnone and unone:
                raise ValueError("For autograd with option 'GenzBretz', at least lower or upper must have one finite component.") # TODO deal with this assigning zero grad for the covariance
            elif lnone:
                upp = uppe
            elif unone:
                upp = -lowe
            else:
                raise ValueError("For autograd with option 'GenzBretz', at least lower or upper should be None (or with all components inifinite).")
            value =  CDFapp(upp,c,maxpts,abseps,releps)

    else:
        raise ValueError("The 'method=' should be either 'GenzBretz' or 'MonteCarlo'.")

    #if error_info and error > abseps:
    #        warn("Estimated error is higher than abseps. Consider raising the computation budget (nmc for method='MonteCarlo' or maxpts for 'GenzBretz'). Switch 'error_info' to False to ignore.")
    return value, error, info

