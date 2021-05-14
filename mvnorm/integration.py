from itertools import zip_longest
from scipy.stats import mvn
from numpy import array, zeros, int32,broadcast_to, Inf,full
from joblib import Parallel, delayed

def prod(tup):
    a = 1
    for i in tup:
        a *= i
    return a

def broadcast_shape(a,b):
    res = reversed(tuple(i if j==1 else (j if i==1 else (i if i==j else -1)) for i,j in zip_longest(reversed(a),reversed(b),fillvalue=1)))
    return list(res)

class ParameterBox:
    pass

integration = ParameterBox() # Will contain all module-level values.
integration.maxpts = None # global in module # "maxpts=None" means maxpts = d*1000
integration.abseps = 1e-6 # global in module
integration.releps = 1e-6
integration.n_jobs = 1
# TODO : allow more joblib control
# TODO : forbid typos! like "integration.relesp"

def integrate(l,u,m,c):
    return mvn.mvnun(l, u, m, c, integration.maxpts, integration.abseps, integration.releps)

def parallel_integration(l,u,m,c):
    N = c.shape[0]
    if N==0:
        return tuple(), tuple()
    p = Parallel(n_jobs=integration.n_jobs)(
        delayed(integrate)(l[j,...], u[j,...], m[j,...], c[j,...]) for j in range(N)) 
    v, i = zip(*p)
    return v, i


def hyperrectangle_integration(lower,upper,mean,covariance,info=False):
    # parallel batch version of scipy.stats.mvn
    # no pytorch here
    d = covariance.shape[-1]
    if mean is None:
        mean = zeros(d)
    if lower is None:
        lower = full(d,-Inf)
    if upper is None:
        upper = full(d,Inf)
    batch_shape = broadcast_shape(lower.shape[:-1],broadcast_shape(upper.shape[:-1],covariance.shape[:-2]))
    vector_shape = batch_shape + [d]
    matrix_shape = batch_shape + [d,d]
    N = prod(batch_shape)
    l = broadcast_to(lower,vector_shape).reshape(N,d)
    u = broadcast_to(upper,vector_shape).reshape(N,d)
    m  = broadcast_to(mean ,vector_shape).reshape(N,d)
    c = broadcast_to(covariance,matrix_shape).reshape(N,d,d)

    v, i = parallel_integration(l,u,m,c)
    values = array(v).reshape(batch_shape)
    if info :
        infos  = array(i, dtype = int32).reshape(batch_shape)
        return (values,infos)
    else:
        return values

