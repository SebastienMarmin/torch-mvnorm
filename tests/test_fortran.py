# A script to verify correctness
from joblib import Parallel, delayed


import sys
sys.path.append(".") 
#sys.path.append("./..") # if this script runs in test folder
from mvnormpy import genz_benz
import torch
import numpy as np
import time

def g(i):
    return genz_benz(d,irr,u_list[i],infin,c_list[i],maxpts,abseps,releps,0)
    

rep = 200
u_list = []
c_list = []
d = 20
maxpts = 3000
abseps = 0.001
releps = 0
irr = np.zeros(d)
infin = np.array(tuple(0 for i in range(d))).astype(np.int32)
for i in range(rep):
    val = d*(3+torch.randn(d))
    A = torch.randn(d,d)
    C = torch.matmul(A,A.transpose(-2,-1))+d**2*torch.diag(torch.ones(d))
    stds = torch.diagonal(C,dim1=-2,dim2=-1).sqrt()
    u = val/stds
    u_n = u.type(torch.float64).numpy()
    c = (C/stds.unsqueeze(-1))/stds.unsqueeze(-2)
    c_n = c.type(torch.float64).numpy()[np.tril_indices(d,-1)]
    u_list +=  [u_n]
    c_list +=  [c_n]
print("Start...")
t = time.time()
a = Parallel(n_jobs=3,prefer="threads")(delayed(g)(i) for i in range(rep))
print("Time: "+str(time.time()-t)+" s.")