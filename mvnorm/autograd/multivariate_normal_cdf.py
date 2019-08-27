
from .conditioning import one_component_conditioning, two_components_conditioning
from numpy import array, zeros as np_zeros, int32, tril_indices, float64, broadcast_to
from torch import tensor, diagonal, cat, unbind, ones as torch_ones, zeros as torch_zeros, ones_like, tril_indices as torch_tril_indices, triu_indices as torch_triu_indices,tril,triu,diag_embed # onely for dev
from torch.autograd import Function, grad # only for test

from mvnorm import genz_bretz
from mvnorm.hyperrectangle_integral import hyperrectangle_integral

from operator import mul
from itertools import zip_longest

# number of jobs > 1 necessary for high d. Else faster with n_jobs=1.
N_JOBS = 1
if N_JOBS > 1:
    from joblib import Parallel, delayed

# Default computation-budget parameters that can be changed by the user (see TODO).
MAXPTS = 25000
ABSEPS = 0.001
RELEPS = 0



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
 

def broadcast_shape(a,b):
    res = reversed(tuple(i if j==1 else (j if i==1 else (i if i==j else -1)) for i,j in zip_longest(reversed(a),reversed(b),fillvalue=1)))
    return list(res)

def _parallel_CDF(x,correlation,maxpts,abseps,releps):
    d = correlation.size(-1)
    ### Convert to numpy
    lower = np_zeros(d).astype(float64)
    upper = x.numpy().astype(float64)
    infin = array(tuple(0 for i in range(d))).astype(int32) 
    trind = tril_indices(d,-1)
    corre = correlation.numpy()[...,trind[0],trind[1]].astype(float64)
    #### Broadcast  
    batch_shape = broadcast_shape(upper.shape[:-1],corre.shape[:-1])
    # equivalent to broadcast(upper[...,0],corre[...,0]).shape .
    dd = d*(d-1)//2
    shape1 = batch_shape+[d]
    shape2 = batch_shape+[dd]
    u = broadcast_to(upper,shape1).reshape(-1,d)
    N = u.shape[0]
    l = broadcast_to(lower,[N,d])
    i = broadcast_to(infin,[N,d])
    c = broadcast_to(corre,shape2).reshape(N,dd)
    # Note: need to make repetitions for l and i even if they are constant to handle parallel
    # runs of Fortran code. We don't want several pointers to the same memory location.
    print(" of dim "+str(d))
    if N_JOBS>1:
        p = Parallel(n_jobs=N_JOBS,prefer="threads")(
            delayed(genz_bretz)(d, l[j,:], u[j,:], i[j,:], c[j,:], maxpts,abseps,releps,0) for j in range(N))
    else:
        p = (genz_bretz(d, l[j,:], u[j,:], i[j,:], c[j,:], maxpts,abseps,releps,0) for j in range(N))
    out = tensor(tuple(p),dtype = x.dtype,device = x.device).view(batch_shape)    
    return out

class MultivariateNormalCDF(Function):

    @staticmethod
    def forward(ctx, val,c,maxpts,abseps,releps):
        # input infos
        ctx.maxpts   = maxpts
        ctx.abseps   = abseps
        ctx.releps   = releps
        ctx.save_for_backward(val,c) # m.data is not needed for gradient computation
        stds = diagonal(c,dim1=-2,dim2=-1).sqrt()
        ctx.stds = stds
        val_rescaled = (val/ctx.stds)
        corr = ((c/stds.unsqueeze(-1))/stds.unsqueeze(-2))
        out = _parallel_CDF(val_rescaled,corr,maxpts,abseps,releps)
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
        P = _parallel_CDF(m_cond,c_cond,ctx.maxpts,ctx.abseps,ctx.releps)
        #P_l= (CDFapp(m_c_l[i],c_c_l[i],ctx.maxpts,ctx.abseps,ctx.releps).unsqueeze(-1) for i in range(d))
        #P = cat(tuple(P_l),-1)
        res = grad_output.unsqueeze(-1)*P*p
        if need_x:
            grad_x = res
        if need_c:
            #dd = d*(d-1)//2
            #m_cond2 = cat(tuple(m_c_l2[i].unsqueeze(-2) for i in range(dd)),-2)
            #c_cond2 = cat(tuple(c_c_l2[i].unsqueeze(-3) for i in range(dd)),-3)
            
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
        #if bias is not None and ctx.needs_input_grad[2]:
        return grad_x, grad_c, None, None, None
       
# Create functional linked with autograd machinerie.
CDFapp = MultivariateNormalCDF.apply

def multivariate_normal_cdf(x,loc=None,covariance_matrix=None,scale_tril=None,method="GenzBretz",nmc=200, maxpts = 25000, abseps = 0.001, releps = 0, error_info = False):
    if (covariance_matrix is not None) + (scale_tril is not None) != 1:
        raise ValueError("Exactly one of sigma or scale_tril may be specified.")
    mat  = scale_tril if covariance_matrix is None else covariance_matrix
    device, dtype = mat.device, mat.dtype
    d = mat.size(-1)     
    if loc is None:
        loc = torch_zeros(d,device=device,dtype=dtype)
    if isinstance(x,(int,float)):
        x = Tensor([float(upper)],device=device).type(dtype)
    if method=="MonteCarlo": # Monte Carlo estimation
        p = MultivariateNormal(loc=loc,scale_tril=scale_tril,covariance_matrix=covariance_matrix)
        r = nmc%5
        N = nmc if r==0 else nmc + 5 - r # rounded to the upper multiple of 5
        Z = (p.sample(torch.Size([N]))<x).prod(-1)
        if error_info: # Does NOT slow down significatively
            booleans = Z.view(*Z.shape[:-1],5,N//5) # divide in 5 groups to have an idea of the precision 
            values   = ((booleans.sum(-1).type(torch.float)))/N*5
            value = values.mean(-1).item()
            std = values.var(-1).sqrt().item()
            error = 1.96 * std / sqrt(5) # at 95 %
        else:
            value = Z.mean(-1).item()
            error = -1
    elif method == "GenzBretz": # Fortran routine
        if d!=x.size(-1):
            raise ValueError("The covariance matrix does not have the same number of dimensions (" +str(d)+ ") as 'x' ("+str(x.size(-1))+").")
        if (d > 1000):
            raise ValueError("Only dimensions below 1000 are allowed. Got "+str(d)+".")
        if not error_info:
                if maxpts is None:
                    maxpts = MAXPTS
                if abseps is None:
                    abseps = ABSEPS
                if releps is None:
                    releps = RELEPS
                c =  matmul(scale_tril,scale_tril.transose(-1,-2)) if covariance_matrix is None else covariance_matrix
                value =  CDFapp(x-loc,c,maxpts,abseps,releps)
                error = -1
        else:
            if x.requires_grad or loc.requires_grad or mat.requires_grad:
                raise ValueError("Option 'error_info' is True, and one of x, loc, covariance_matrix or scale_tril requires gradient. With option 'GenzBretz', the estimation of CDF error is not compatible with autograd.")
            if x.dim()>1 or loc.dim()>1 or mat.dim()>2:
                raise ValueError("Option 'error_info' is True, and one of x, loc, covariance_matrix or scale_tril requires gradient have a number of dim > 1 (or > 2 for matrices). With option 'GenzBretz', the estimation of CDF error is not compatible with batch tensor shape.")
            value, error, info = hyperrectangle_integral(upper = x, mean = loc, sigma = covariance_matrix,scale_tril=scale_tril, maxpts = maxpts, abseps = abseps, releps = releps)
    else:
        raise ValueError("The 'method=' should be either 'GenzBretz' or 'MonteCarlo'.")
    if error_info and error > abseps:
            warn("Estimated error is higher than abseps. Consider raising the computation budget (nmc for method='MonteCarlo' or maxpts for 'GenzBretz'). Switch 'error_info' to False to ignore.")
    return value, error

