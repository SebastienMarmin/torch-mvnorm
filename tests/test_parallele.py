import torch
from torch import randn, matmul
from torch.autograd import grad
import sys
sys.path.append(".")
from mvnorm import Phi, integration
from time import time

# Computation of P(Y<0), Y ~ N(m,C), by density integration.
# NOTE For P(Y<x), replace "m" by "m+x".
# Future releases may have P(x,m,c), or even P(l,u,m,c).

batch_dim = [23,12] # do 276 probability computations in one tensor, can be "[]"

d = 15 # dimension of the random vector
m = randn(*batch_dim,d)
# Compute an arbitrary covariance
A = randn(batch_dim[-1],d,d) if len(batch_dim)>0 else randn(d,d)
C = torch.diag(torch.ones(d))+ 1/d*matmul(A,A.transpose(-1,-2))

####        mean shape: [*batch_shape,d] (can be just "[d]")
####  covariance shape: [*batch_shape,d,d]
#### result prob shape: [*batch_shape] (tensor scalar if batch_shape is empty)
# Batch_shape of tensors can broadcast:
# for exemple m.shape == [3,4,6] and C.shape == [4,6,6]

# parameters of "from scipy.stats import mvn"
integration.maxpts=1000*d # default
integration.abseps=1e-6 # default
integration.releps=1e-6 # default
integration.n_jobs = 1 # joblib parameter

t1 = time()
print(Phi(m,C))
t2 = time()

print("With "+str(integration.n_jobs)+" job(s):"+str(round(t2-t1,3))+" s.")


integration.n_jobs = 10
t1 = time()
print(Phi(m,C))
t2 = time()

print("With "+str(integration.n_jobs)+" job(s):"+str(round(t2-t1,3))+" s.")
