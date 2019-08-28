


if __name__ == "__main__":
    
    import torch
    from torch.autograd import grad
    import sys
    sys.path.append("..")
    sys.path.append(".")
    from mvnorm.hyperrectangle_integral import hyperrectangle_integral
    from mvnorm.autograd.multivariate_normal_cdf import multivariate_normal_cdf as CDF
    def dPdx_num(x,c, maxpts = 25000, abseps = 0.001, releps = 0):
        d = c.size(-1)
        res = torch.empty(d)
        p,err,_ = hyperrectangle_integral(upper= x, sigma = c,maxpts = maxpts, abseps = abseps,releps=releps)
        h = 0.005
        for i in range(d):
            xh = x.clone()
            xh[i] = xh[i]+h
            ph,_,_ = hyperrectangle_integral(upper= xh, sigma = c,maxpts = maxpts, abseps = abseps,releps=releps)
            res[i] = (ph-p)/h
        return res
    def d2Pdx2_num(x,c, maxpts = 25000, abseps = 0.001, releps = 0):
        d = c.size(-1)
        res = torch.empty(d,d)
        h = 0.05
        for i in range(d):
            zz = torch.zeros(d)
            zz[i] = 1
            for j in range(d):
                yy = torch.zeros(d)
                yy[j] = 1
                p1=hyperrectangle_integral(upper=x+h*(zz+yy),sigma= c,maxpts = maxpts, abseps = abseps,releps=releps)[0]
                p2=hyperrectangle_integral(upper=x+h*(yy), sigma = c,maxpts = maxpts, abseps = abseps,releps=releps)[0]
                p3=hyperrectangle_integral(upper=x + h*zz, sigma = c,maxpts = maxpts, abseps = abseps,releps=releps)[0]
                p4=hyperrectangle_integral(upper=x,sigma = c,maxpts = maxpts, abseps = abseps,releps=releps)[0]
                res[i,j] = 1/h**2*(p1 - p2 -(p3 - p4))
        return res
    def dPdC_num(x,c, maxpts = 25000, abseps = 0.001, releps = 0):
        d = c.size(-1)
        res = torch.empty(d,d)
        h = 0.005
        for i in range(d):
            for j in range(d):
                zz = torch.zeros(d,d)
                zz[i,j] = .5
                zz[j,i] = .5
                if i==j:
                    zz[j,i] = 1
                p1=hyperrectangle_integral(upper=x,sigma= c+h*zz,maxpts = maxpts, abseps = abseps,releps=releps)[0]
                p2=hyperrectangle_integral(upper=x, sigma = c,maxpts = maxpts, abseps = abseps,releps=releps)[0]
                res[i,j] = 1/(h)*(p1 - p2)
        return res



    d = 4
    bd = [1,2]
    x = torch.zeros(*bd,d)
    x.requires_grad = True
    A = torch.randn(*bd,d,d)
    C = (torch.matmul(A,A.transpose(-2,-1))+1*torch.diag_embed(torch.ones(*bd,d),dim1=-1,dim2=-2))
    C.requires_grad = True
    maxpts = 250000
    abseps = 0.000001
    releps = 0

    p = CDF(x,covariance_matrix=C,maxpts=maxpts,abseps=abseps)[0]
    print(p)
    indi = (0,0)
    print("ANALYTICAL: d(CDF(X))/dx_"+str(indi[0])+str(indi[1])+" = ")
    G = grad(p[indi],(x,C),create_graph=False)
    print(G[0])
    print("NUMERICAL:  d(CDF(X_"+str(indi[0])+str(indi[1])+"))/dx_"+str(indi[0])+str(indi[1])+" = ")
    print(dPdx_num(x[indi[0],indi[1],:],C[indi[0],indi[1],...],maxpts=maxpts,abseps=abseps))

    # Check of Paciorek formula:
    #print(d2Pdx2_num(x,C,maxpts=maxpts,abseps=abseps))
    #print(2*dPdC_num(x,C ,maxpts=maxpts,abseps=abseps))
    print(d2Pdx2_num(x[indi[0],indi[1],:],C[indi[0],indi[1],...],maxpts=maxpts,abseps=abseps))
    print(2*dPdC_num(x[indi[0],indi[1],:],C[indi[0],indi[1],...] ,maxpts=maxpts,abseps=abseps))
    print(G[1])
