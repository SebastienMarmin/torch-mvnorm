# number of jobs > 1 necessary for high d. Else faster with n_jobs=1.

from operator import mul
from itertools import zip_longest

from mvnorm import genz_bretz
from numpy import array, zeros as np_zeros, int32, tril_indices, float64, broadcast_to,Inf
from torch import tensor, int32 as torch_int32, float32 as torch_float32


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
    N = l.shape[0]
    if N_JOBS>1:
        p = Parallel(n_jobs=N_JOBS,prefer="threads")(
            delayed(genz_bretz)(d, l[j,:], u[j,:], i[j,:], c[j,:], maxpts,abseps,releps,infint) for j in range(N))
    else:
        p = (genz_bretz(d, l[j,:], u[j,:], i[j,:], c[j,:], maxpts,abseps,releps,infint) for j in range(N))
    if info:
        return zip(*tuple(p)) # i.e. values, errors, infos
    else:
        return tuple(p) #



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

def _parallel_hyperrectangle_integration(lower,upper,correlation,maxpts,abseps,releps):
    # main differences with _parallel_CDF is that it handles complex lower/upper bounds
    # with infinite components and it returns information on the completion.
    d = correlation.size(-1)
    trind = tril_indices(d,-1)
    ### Convert to numpy
    low = lower.numpy().astype(float64)
    upp = upper.numpy().astype(float64)
    cor = correlation.numpy()[...,trind[0],trind[1]].astype(float64)
    batch_shape = broadcast(low[...,0],upp[...,0],cor[...,0,0]).shape
    dd = d*(d-1)//2
    shape1 = batch_shape+[d]
    shape2 = batch_shape+[dd]
    l = broadcast_to(low,shape1).reshape(-1,d)
    N = u.shape[0]
    u = broadcast_to(upp,shape1).reshape(N,d)
    c = broadcast_to(corre,shape2).reshape(N,dd)
    infin = np.array(tuple(2 for i in range(d))).astype(int32)
    i = broadcast_to(infin,[N,d])
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
    # now that this info is stored, we get rid of the numpy.Inf's
    #  (they are user-friendly but not understood in Fortran)
    lower[infl] = 0
    upper[infu] = 0
    # TODO better to build res and assign or build-reshap?
    values, errors, infos = _parallel_genz_bretz(l,u,i,c,maxpts,abseps,releps,info=True)
    # for skipping `info` use _parallel_CDF
    return (tensor(values,dtype = x.dtype,device = x.device).view(batch_shape),
            tensor(errors,dtype = torch_float32,device = x.device).view(batch_shape),
            tensor(infos, dtype = torch_int32,device = x.device).view(batch_shape))

