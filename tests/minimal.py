import sys
sys.path.append(".")

import torch
from torch.autograd import grad
from mvnorm import multivariate_normal_cdf as Phi

d = 3
x = torch.randn(d)
m = torch.zeros(d)
Croot = torch.randn(d,d)
C = Croot.mm(Croot.t())+torch.diag(torch.ones(d))

x.requires_grad = True
m.requires_grad = True
C.requires_grad = True

P = Phi(x,m,C)
dPdx, dPdm, dPdC = grad(P,(x,m,C))

print(dPdx)
print(dPdm)
print(dPdC)
