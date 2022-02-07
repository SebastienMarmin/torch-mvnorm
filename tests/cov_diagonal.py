import torch
from torch import randn, matmul
from torch.autograd import grad
import sys
sys.path.append(".")
from mvnorm import multivariate_normal_cdf as Phi, integration
from time import time
from numpy import sqrt

# Computation of P(Y<0), Y ~ N(m,C), by density integration.
batch_dim = [2,8] # do 16 probability computations in one tensor, can be a scalar ("[]")

d = 23 # dimension of the random vector
x = randn(*batch_dim,d)+0.5*sqrt(d)
# Compute an arbitrary covariance
D = randn(*batch_dim,d)**2
C = torch.diag_embed(D,dim1=-2,dim2=-1)

# parameters of "from scipy.stats import mvn"
integration.maxpts=1000*d # default
integration.abseps=1e-6 # default
integration.releps=1e-6 # default
integration.n_jobs = 1 # joblib parameter

t1 = time()
print(Phi(x,covariance_matrix= C,diagonality_tolerance=-1))
t2 = time()

print("With full matrix and "+str(integration.n_jobs)+" job(s):"+str(round(t2-t1,3))+" s.")



t1 = time()
print(Phi(x,covariance_matrix=C,diagonality_tolerance=0))
t2 = time()

print("With knowing it's actually diagonal and 1 job:"+str(round(t2-t1,3))+" s.")


t1 = time()
print(Phi(x,covariance_matrix=C+1e-4,diagonality_tolerance=2e-4))
t2 = time()

print("With detecting it's almost diagonal:"+str(round(t2-t1,3))+" s.")

