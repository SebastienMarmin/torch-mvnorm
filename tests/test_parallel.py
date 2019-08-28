
import sys
sys.path.append(".") 
#sys.path.append("./..") # if this script runs in test folder
from mvnorm.parallel.joblib import _hyperrectangle_integration
import torch
import numpy as np


bs = []
d = 3
val = d*(3+torch.randn(*bs,d))
A = torch.randn(*bs,d,d)
C = torch.matmul(A,A.transpose(-2,-1))+d**2*torch.diag(torch.ones(d))
stds = torch.diagonal(C,dim1=-2,dim2=-1).sqrt()
val_rs = val/stds
c = (C/stds.unsqueeze(-1))/stds.unsqueeze(-2)
maxpts,abseps,releps = 1000,.001,0
print(_parallel_CDF(val_rs,c,maxpts,abseps,releps))


u = val
l = u.clone()
l -= 1000
l[...,-1] -= np.Inf #

u_rs = u/stds
l_rs = l/stds
print("l")
print(l_rs)
print("u")
print(u_rs)
print(_hyperrectangle_integration(l_rs,u_rs,c,maxpts,abseps,releps))
l2 = l_rs
l2[2] = -1
print("l")
print(l2)
print("u")
print(u_rs)
print(_parallel_hyperrectangle_integration(l2,u_rs,c,maxpts,abseps,releps))


