


if __name__ == "__main__":
    
    import torch
    from torch.autograd import grad
    import sys
    sys.path.append(".")
    from mvnorm.hyperrectangle_integral import hyperrectangle_integral
    from mvnorm.autograd.multivariate_normal_cdf import multivariate_normal_cdf as CDF
    def dPdx_num(x,c, maxpts = 25000, abseps = 0.001, releps = 0):
        d = c.size(-1)
        res = torch.empty(d)
        p = CDF(x,covariance_matrix= c,maxpts = maxpts, abseps = abseps,releps=releps)
        h = 0.005
        for i in range(d):
            xh = x.clone()
            xh[i] = xh[i]+h
            ph = CDF(xh,covariance_matrix=c,maxpts = maxpts, abseps = abseps,releps=releps)
            res[i] = (ph-p)/h
        return res



    d = 3
    bd = [3,2]
    values = torch.randn(*bd,d)
    values.requires_grad = True
    A = torch.randn(*bd,d,d)
    C = torch.matmul(A,A.transpose(-2,-1))+torch.diag_embed(torch.ones(*bd,d),dim1=-1,dim2=-2)
    maxpts = 2500000
    abseps = 0.00001
    releps = 0
    mean = torch.zeros(*bd,d)
    mean.requires_grad = True
    x = (values-mean)
    p = CDF(x,covariance_matrix=C,maxpts=maxpts,abseps=abseps)
    print(p)
    indi = (2,1)
    print("ANALYTICAL: d(CDF(X))/dx_"+str(indi[0])+str(indi[1])+" = ")
    print(grad(p[indi],(values))[0])
    print("NUMERICAL:  d(CDF(X_"+str(indi[0])+str(indi[1])+"))/dx_"+str(indi[0])+str(indi[1])+" = ")
    print(dPdx_num(x[indi[0],indi[1],:],C[indi[0],indi[1],...],maxpts=maxpts,abseps=abseps))

