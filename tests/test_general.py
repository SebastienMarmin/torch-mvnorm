import sys
sys.path.append(".")

from torch import randn, matmul
from torch.autograd import grad
from mvnorm import multivariate_normal_cdf as P


n = 4
x = randn(n,requires_grad=True)
A = randn(n,n)
C = matmul(A,A.t())

print("For Y~N(0,C), P(Y<x)=")
p = P(x,covariance_matrix=C)[0]
print(p)
print("dP(Y<x)/dx = ")
print(grad(p,(x,))[0])




