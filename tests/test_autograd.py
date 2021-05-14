import torch
from torch import zeros
from torch.autograd import grad
import sys
sys.path.append(".")
from mvnorm import Phi, integration




def dPdC_num(m,c):
    d = c.size(-1)
    res = torch.empty(d)
    p = Phi(m,c)
    h = 0.005
    res = torch.empty(d,d)
    for i in range(d):
        for j in range(d):
            ch = c.clone()
            ch[i,j] = ch[i,j]+h/2
            ch[j,i] = ch[j,i]+h/2
            ph = Phi(m,ch)
            res[i,j] = (ph-p)/h
    return res



def dPdx_num(m,c, maxpts = 25000, abseps = 0.000001, releps = 0):
    d = c.size(-1)
    res = torch.empty(d)
    p = Phi(m,c)
    h = 0.005
    for i in range(d):
        mh = m.clone()
        mh[i] = mh[i]+h
        ph = Phi(mh,c)
        res[i] = (ph-p)/h
    return res

def d2Pdx2_num(m,c):
    h = 0.005
    d = c.shape[-1]
    res = torch.zeros(d,d)
    for i in range(d):
        zz = zeros(d)
        zz[i] = 1
        for j in range(d):
            yy = zeros(d)
            yy[j] = 1
            res[i,j] = (1/h**2*(Phi(m + .5*h*(zz+yy), c) - Phi(m + .5*h*(-zz+yy), c)-(Phi(m + .5*h*(zz-yy), c) - Phi(m + .5*h*(-zz-yy), c)))).detach()
    return 0.5*(res.transpose(-2,-1)+res)

def jacobian(y, x, create_graph=False):                                                               
    jac = []                                                                                          
    flat_y = (y*1).reshape(-1) # does someone know why it doesn't work without this "*1"? tx
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)):                                                                      
        grad_y[i] = 1.                                                                                
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))                                                           
        grad_y[i] = 0.                                                                                
    return torch.stack(jac).reshape(y.shape + x.shape)                                                
                                                                                                    
def hessian(y, x):                                                                                    
    return jacobian(jacobian(y, x, create_graph=True), x)           


d = 3
A = torch.randn(d,d)
C = torch.matmul(A,A.transpose(-2,-1))+torch.diag_embed(torch.ones(d),dim1=-1,dim2=-2)
integration.maxpts=100000000
integration.abseps=1e-8
integration.releps=0
# Usually, maxpts doesn't need to be so high (and abseps&releps so small). It is set
# high here to check derivatives using finite difference.
# Finite difference requires extremely high precision (and computation time).
mean = torch.zeros(d)
mean.requires_grad = True
C.requires_grad = True
P = Phi(mean,C)
dPdm, dPdC = grad(P,(mean,C),create_graph=True)

print("___Gradient___   dPhi/dm")
print("ANALYTICAL:")
print(dPdm)
print("NUMERICAL:")
print(dPdx_num(mean,C))

print("___Packett's formula___   1/2*d2Phi/dm2 == dPhi/dC")
print("ANALYTICAL dPhi/dC:")
print(dPdC)
print("ANALYTICAL 1/2*d2Phi/dm2:")
print(.5*hessian(P,mean))
print("NUMERICAL dPhi/dC:")
print(dPdC_num(mean,C))
print("NUMERICAL 1/2*d2Phi/dm2:")
print(0.5*d2Pdx2_num(mean,C))

