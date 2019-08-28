# A script to verify correctness
import sys
sys.path.append(".") 
#sys.path.append("./..") 
from mvnorm import hyperrectangle_integral
from mvnorm import multivariate_normal_cdf
import torch
from torch.distributions import MultivariateNormal
from torch import randn, matmul, rand
import numpy as np
from numpy import sqrt, Inf
import matplotlib
import matplotlib.pylab as plt
import time
from joblib import Parallel, delayed
from warnings import warn


n = 1
error_info = False
nmc=20000
maxpts = 100000
abseps = 0.000003
releps = 0
bd= [2]
lower_list = [randn(*bd,n),randn(*bd,n,requires_grad=True),randn(*bd,n)]
upper_list = [l+rand(*bd,n) if l is not None else None for l in lower_list]
lower_list[2].fill_(-Inf)
upper_list[2].fill_(Inf)
lower_list[1][...,-1] = -Inf
upper_list[1][...,0] = Inf
upper_list[1][...,-1] = Inf

m_list = [randn(*bd,n),None,randn(*bd,n)]
A = randn(*[],n,n)
C_list = [torch.diag_embed(torch.ones(*bd,n),dim1=-1,dim2=-2)+ 1/n*matmul(A,A.transpose(-1,-2))]
C_list[-1].requires_grad = True
S_list = [None]#[None,torch.cholesky(C_list[0])]



for lower in lower_list:
    for upper in upper_list:
        for loc in m_list:
            for covariance_matrix in C_list:
                for scale_tril in S_list:
                    print("lower")
                    print(lower)
                    print("upper")
                    print(upper)
                    print("loc")
                    print(loc)
                    try:
                        a1 = multivariate_normal_cdf(lower,upper,loc,covariance_matrix,scale_tril,"GenzBretz",nmc, maxpts, abseps, releps, error_info)
                    except ValueError as e:
                        a1 = e
                    #try:
                    a1bis = multivariate_normal_cdf(lower.detach() if lower is not None else None,upper.detach() if upper is not None else None,loc.detach() if loc is not None else None,covariance_matrix.detach() if covariance_matrix is not None else None,scale_tril.detach() if scale_tril is not None else None,"GenzBretz",nmc, maxpts, abseps, releps, error_info)
                    #except ValueError as e:
                    #    a1bis = e
                    try:
                        a2 =multivariate_normal_cdf(lower,upper,loc,covariance_matrix,scale_tril,"MonteCarlo",nmc, maxpts, abseps, releps, error_info)
                    except ValueError as e:
                        a2 = e
                    print("RESULTS")
                    print(a1)
                    print(a1bis)
                    print(a2)
                    print("-----------------------------------------------------------")
    