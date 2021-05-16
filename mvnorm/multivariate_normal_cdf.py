from numpy import zeros_like
from torch import tensor, diagonal, cat, unbind, ones as torch_ones, zeros as torch_zeros, ones_like, tril_indices as torch_tril_indices, triu_indices as torch_triu_indices,tril,triu,diag_embed, erfc # onely for dev
from torch.autograd import Function, grad # only for test

from torch import tensor, diagonal, cat, unbind, zeros as torch_zeros, Tensor, Size, matmul,int32
from torch.distributions import MultivariateNormal

from torch.autograd import Function
from torch import randn, from_numpy
from numpy import Inf

from .integration import hyperrectangle_integration
from .conditioning import make_condition


sqrt5 = 2.2360679774997898050514777424
def phi(z,v):
    return 0.39894228040143270286321808271/v.sqrt()*(-z**2/(2*v)).exp()
#                 ^ oneOverSqrt2pi     

sqrt2M1 = 0.70710678118654746171500846685
def Phi1D(x,m,c):
    z = (x-m)/c.squeeze(-1).sqrt()
    return (erfc(-z*sqrt2M1)/2).squeeze(-1)

def phi2_sub(z,C): # compute pairs of bivariate densities and organize them in a matrix
    V = diagonal(C,dim1=-2,dim2=-1)
    a,c = V.unsqueeze(-1),V.unsqueeze(-2)
    det = a*c-C**2
    x1 = z.unsqueeze(-1)
    x2 = z.unsqueeze(-2)
    exponent = -0.5/det*(c*x1**2+a*x2**2-2*C*x1*x2)
    res = 0.15915494309189534560822210096/det.sqrt()*(exponent).exp()
    #           ^ oneOver2pi     
    return tril(res,-1) + triu(res,1)


def to_torch(x):
    if len(x.shape) == 0:
        return tensor(float(x))
    else:
        return from_numpy(x)



class PhiHighDim(Function):

    @staticmethod
    def forward(ctx, m, c):
        m_np = m.numpy()
        c_np = .5*(c+c.transpose(-1,-2)).numpy()
        ctx.save_for_backward(m,c)
        return to_torch(hyperrectangle_integration(None,zeros_like(m_np),m_np,c_np))

    @staticmethod
    def backward(ctx,grad_output):
        if grad_output is None:
            return None, None
        res_m = res_c = None
        need_m, need_c = ctx.needs_input_grad[0:2]
        if need_c or need_m:
            m,c = ctx.saved_tensors
            m_cond,c_cond = make_condition(0,m,c)
            v = diagonal(c,dim1=-2,dim2=-1)
            p = phi(m,v)
            P = Phi(m_cond,c_cond)
            grad_m = -P*p
            grad_output_u1 = grad_output.unsqueeze(-1)
            res_m = grad_output_u1*grad_m
            if need_c:
                d = c.shape[-1] # d==1 should never happen here
                if d==2:
                    P2 = 1
                else:
                    trilind = torch_tril_indices(d,d-1,offset=-1)
                    m_cond2,c_cond2 = make_condition(0,m_cond,c_cond)
                    Q_l = Phi(m_cond2[...,trilind[0],trilind[1],:],c_cond2[...,trilind[0],trilind[1],:,:])
                    P2 = torch_zeros(*Q_l.shape[:-1],d,d,dtype=Q_l.dtype)
                    P2[...,trilind[0],trilind[1]] = Q_l
                    P2[...,trilind[1],trilind[0]] = Q_l
                p2 = phi2_sub(m,c)
                hess = p2*P2
                D = -(m*grad_m+(hess*c).sum(-1))/v
                grad_c = .5*(hess + diag_embed(D))
                res_c = grad_output_u1.unsqueeze(-1)*grad_c
        return res_m, res_c


Phinception = PhiHighDim.apply

def Phi(m,c):
    d = c.shape[-1]
    if d == 1:
        return Phi1D(0,m,c)
    else:
        return Phinception(m,c)

if __name__ == "__main__":
    import torch                                                                                          
                                                                                                      
    def jacobian(y, x, create_graph=False):                                                               
        jac = []                                                                                          
        flat_y = (y*1).reshape(-1) # does someone know why it doesn't work without this one?
        grad_y = torch.zeros_like(flat_y)                                                                 
        for i in range(len(flat_y)):                                                                      
            grad_y[i] = 1.                                                                                
            grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
            jac.append(grad_x.reshape(x.shape))                                                           
            grad_y[i] = 0.                                                                                
        return torch.stack(jac).reshape(y.shape + x.shape)                                                
                                                                                                        
    def hessian(y, x):                                                                                    
        return jacobian(jacobian(y, x, create_graph=True), x)                                             
                                                                                                        
    def f(x):                                                                                             
        return x**2 +x[0]*x[1]*1
                                                                                                        
    x = torch.ones(4, requires_grad=True)                                                                 

    d = 20

    batch_shape = []

    m = torch.rand( d,requires_grad=True)#,dtype=torch.float64)
    a = torch.rand(*batch_shape,d,d)#,dtype=torch.float64)
    c = torch.matmul(a,a.transpose(-1,-2))
    
    
    #coef = torch.tensor([10*i for i in range(int(np.prod(batch_shape)))]).reshape(batch_shape)
    #np.save("m",m.detach().numpy())
    #np.save("c",c.detach().numpy())
    #print("coef=",end="")
    #print(coef)
    P = (Phi(m,c))
    print("P =",end=" ")
    print(P)
    import time
    t1 = time.time()
    print(.5*hessian(P,m))
    t2 = time.time()
    print(t2-t1)
    c.requires_grad = True
    P = (Phi(m,c))
    t1 = time.time()
    print(grad(P,[c])[0])
    t2 = time.time()
    print(t2-t1)
    #dP = grad(P,[m,c])
    #print("dPdm")
    #print("dP =",end=" ")
    #print(dP)

    #res = torch.autograd.gradcheck(Phi.apply, (m, c),atol=0.001,rtol=0.01, nondet_tol=0.02, raise_exception=True)
    #print(res) 

"""
    import torch
    # Create functional linked with autograd machinerie.
    Phi_app = Phi.apply
    d = 4
    m = randn(d,requires_grad=True)
    a = randn(d,d)
    c = torch.matmul(a,a.t())
    P = Phi_app(m,c)
    dPdm = grad(P,[m], create_graph=True)[0]
    print("dPdm")
    print(dPdm)
    d2Pdm2 = grad(dPdm[0],[m], create_graph=True)[0]
    print("d2Pdm2")
    print(d2Pdm2)
    d3Pdm3 = grad(d2Pdm2[0],[m])[0]
    print("d3Pdm3")
    print(d3Pdm3)
    #print((c**3*(x*c).sin()).detach())
"""

"""
Q_l = _parallel_CDF(m_cond2,c_cond2,ctx.maxpts,ctx.abseps,ctx.releps)
Q = torch_zeros(*Q_l.shape[:-1],d,d)
trilind = torch_tril_indices(d,d,offset=-1)
triuind = torch_triu_indices(d,d,offset =1)
Q[...,trilind[0],trilind[1]] = Q_l
Q[...,triuind[0],triuind[1]] = Q_l
q = phi2_sub(val,c)
hess = q*Q
D = -(val*res+(hess*c).sum(-1))/var
grad_c = hess + diag_embed(D)
"""


"""


class MultivariateNormalCDF(Function):
    @staticmethod
    def forward(ctx, x, c):
        ctx.save_for_backward(x,c)
        out_np = f(x.numpy(),c.numpy())
        out = to_torch(out_np)# .float()#.to(device)
        return out 

    @staticmethod
    def backward(ctx, grad_output):
        x,c = ctx.saved_tensors
        need_x, need_c = ctx.needs_input_grad[:2]
        grad_x = grad_c = None
        if need_x:
            grad_x_np = dfdx(x.numpy(),c.numpy())
            grad_x = to_torch(grad_x_np)# .float()#.to(device)
        if need_c:
            grad_c_np = dfdc(x.numpy(),c.numpy())
            grad_c = to_torch(grad_c_np)
        return grad_x, grad_c



"""