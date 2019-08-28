# number of jobs > 1 necessary for high d. Else faster with n_jobs=1.

from operator import mul
from itertools import zip_longest

from mvnorm import genz_bretz
from numpy import array, zeros as np_zeros, int32, tril_indices, float64,broadcast, broadcast_to,Inf,empty as np_empty,full, ascontiguousarray
from torch import tensor, int32 as torch_int32, erfc, ones as torch_ones


N_JOBS = 1
if N_JOBS > 1:
    from joblib import Parallel, delayed

def broadcast_shape(a,b):
    res = reversed(tuple(i if j==1 else (j if i==1 else (i if i==j else -1)) for i,j in zip_longest(reversed(a),reversed(b),fillvalue=1)))
    return list(res)


def _parallel_genz_bretz(l,u,i,c,maxpts,abseps,releps,info=False):
    # Note: need to make repetitions for l and i even if they are constant to handle parallel
    # runs of Fortran code. We don't want several pointers to the same memory location.
    infint = 1 if info else 0
    d = l.shape[1]
    N = l.shape[0]
    if N_JOBS>1:
        p = Parallel(n_jobs=N_JOBS,prefer="threads")(
            delayed(genz_bretz)(d, l[j,:], u[j,:], i[j,:], c[j,:], maxpts,abseps,releps,infint) for j in range(N))
    else:
        p = tuple(genz_bretz(d, l[j,:], u[j,:], i[j,:], c[j,:], maxpts,abseps,releps,infint) for j in range(N))
    if info:
        return zip(*p) # i.e. values, errors, infos
    else:
        return p #


sqrt2M1 = 0.70710678118654746171500846685
def Phi(z):
    return erfc(-z*sqrt2M1)/2

def _hyperrectangle_integration(lower,upper,correlation,maxpts,abseps,releps,info=False):
    # main differences with _parallel_CDF is that it handles complex lower/upper bounds
    # with infinite components and it returns information on the completion.
    d = correlation.size(-1)
    trind = tril_indices(d,-1)
    ### Infere batch_shape
    lnone = lower is None
    unone = upper is None
    bothNone = lnone and unone
    if bothNone:
        pre_batch_shape = []
    # broadcast lower and upper to get pre_batch_shape
    elif not lnone and not unone:
        pre_batch_shape = broadcast(lower,upper).shape[:-1]
        cdf = False
    else: # case were we compute P(Y<x): lower is [-inf, ..., -inf] and upper = x.
          # Invert lower and upper if it is upper=None
        cdf = True
        if unone:
            upper = -lower
        pre_batch_shape = upper.shape[:-1]
    cor = ascontiguousarray(correlation.numpy()[...,trind[0],trind[1]],dtype=float64)

    # broadcast all lower, upper, correlation
    batch_shape = broadcast_shape(pre_batch_shape,cor.shape[:-1])
    dtype  = correlation.dtype
    device = correlation.device
    if bothNone: # trivial case
        if info:
            return (torch_ones(*batch_shape,dtype = dtype,device = device),
                    torch_zeros(*batch_shape,dtype = dtype,device = device),
                    torch_zeros(*batch_shape, dtype = torch_int32,device = device))
        else:
            return torch_ones(*batch_shape,dtype = dtype,device = device)
    else:
        if d==1:
            val = Phi(upper.squeeze(-1)) if cdf else  Phi(upper.squeeze(-1)) - Phi(lower.squeeze(-1))
            if info:
                return (val,
                    torch_zeros(*batch_shape,dtype = dtype,device = device),
                    torch_zeros(*batch_shape, dtype = torch_int32,device = device))
            else:
                return val
        dd = d*(d-1)//2 # size of flatten correlation matrix
        # Broadcast:
        c = broadcast_to(cor,batch_shape+[dd]).reshape(-1,dd)
        N = c.shape[0] # batch number
        upp =  upper.numpy().astype(float64)
        shape1 = batch_shape+[d]
        u = broadcast_to(upp,shape1).reshape(N,d)
        infu = u==Inf
        if cdf:
            l = np_empty((N,d),dtype=float64) # never used but required by Fortran code
            i = np_zeros((N,d),dtype=int32)
            i.setflags(write=1)
            i[infu] = -1 # basically ignores these componenents
        else:
            low = lower.numpy().astype(float64)
            l = broadcast_to(low,shape1).reshape(N,d)
            i = full((N, d), 2,dtype=int32)
            infl = l==-Inf
            i.setflags(write=1)
            i[infl] = 0
            i[infu] = 1
            i[infl*infu] = -1 # basically ignores these componenents
        
        # infin is a int vector to pass to the fortran code controlling the integral limits
        #            if INFIN(I) < 0, Ith limits are (-infinity, infinity);
        #            if INFIN(I) = 0, Ith limits are (-infinity, UPPER(I)];
        #            if INFIN(I) = 1, Ith limits are [LOWER(I), infinity);
        #            if INFIN(I) = 2, Ith limits are [LOWER(I), UPPER(I)].  
            
        # TODO better to build res and assign or build-reshap?
        res = _parallel_genz_bretz(l,u,i,c,maxpts,abseps,releps,info)
    if info :
        values, errors, infos = res
        return (tensor(values,dtype = dtype,device = device).view(batch_shape),
                tensor(errors,dtype = dtype,device = device).view(batch_shape),
                tensor(infos, dtype = torch_int32,device = device).view(batch_shape))
    else:
        return tensor(res,dtype = dtype,device = device).view(batch_shape)


    """     l.setflags(write=1)
    l[infl] = 0
    u.setflags(write=1)
    u[infu] = 0 

    # now that this info is stored, we get rid of the numpy.Inf's
    #  (they are user-friendly but not understood in Fortran)

    """

""" 

def _parallel_CDF(x,correlation,maxpts,abseps,releps):
    d = correlation.size(-1)
    ### Convert to numpy
    lower = np_zeros(d).astype(float64)
    # lower is irrelevant but requested by the Fortran code when infi is filled with zeros
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
    res = _parallel_genz_bretz(l,u,i,c,maxpts,abseps,releps,info=False) 
    # for `info` use _parallel_hyperrectangle_integration
    return tensor(res,dtype = x.dtype,device = x.device).view(batch_shape)
 """