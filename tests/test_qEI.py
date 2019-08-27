import sys
sys.path.append(".") 
sys.path.append("./..") 
sys.path.append("./../torchdesigns") 
sys.path.append("./../splinetorch/splinetorch")

import torch
from torch.distributions import MultivariateNormal
from torch import randn, matmul
import numpy as np
from numpy import sqrt
import matplotlib
import matplotlib.pylab as plt
import time
from joblib import Parallel, delayed
from mvnorm import multivariate_normal_cdf as CDF
from splinetorch import Interpolant
from mvnorm.hyperrectangle_integral import hyperrectangle_integral

sqrt2M1 = 0.70710678118654746171500846685

#def Phi(z):
#    return torch.erfc(-z*sqrt2M1)/2
#def ImprovementProbaUB(I,m,c,T):
#    s = torch.diagonal(c,dim1=-1,dim2=-2).sqrt()
#    p = Phi((T-I-m)/s)
#    return p.sum(-1)


def EI(distribution,T,method="GenzBretz",nmc=200,nintegr=40,  maxpts = 100, abseps = 0.001, releps = 0,plo=False):
    if method=="MonteCarlo": # Monte Carlo estimation
        r = nmc%5
        N = nmc if r==0 else nmc + 5 - r
        # rounded to the upper multiple of 5
        Z = distribution.sample(torch.Size([N])).min(-1).values
        improvement = ((Z<T).type(torch.float)*(T-Z)).view(*Z.shape[:-1],5,N//5)
        # divide in 5 groups to have 5 predictions an idea of the variability
        eis   = ((improvement.sum(-1)))/N*5 # compute 5 EIs
        ei = (eis.sum(-1))/5
        std = eis.var(-1).sqrt()
        error = 1.96 * std / sqrt(5) # at 95 %
    elif method == "GenzBretz": # Fortran routine
        p = distribution
        if not issubclass(type(p),(torch.distributions.MultivariateNormal)):
            raise ValueError("Method GenzBretz is only implemented for MultivariateNormal distributions.")
        m = p.loc
        C = p.covariance_matrix
        s = p.stddev
        # compute the integration bounds
        quantiles = m-3.09*s ## compute 0.001 quantiles
        y_min = torch.min(quantiles) # lowest quantile
        improvement_max = T-y_min  # probability of higher improvement than that is neglected
        
        #yy1 = torch.empty(nintegr)
        t1 = (.5*(torch.linspace(0,1,nintegr)+(torch.linspace(0,1,nintegr))**(2)))*improvement_max
        # integration variable, points more dense for low improvement

        d_instru =  MultivariateNormal(loc=-m,scale_tril=distribution.scale_tril) # this or integrating from lower to plus infinity # TODO # also implement a loop deeper
        """
        for i in range(nintegr):
            gb = hyperrectangle_integral(lower = torch.ones(m.size(-1))*(T-t1[i]), mean = m, sigma = C,maxpts = maxpts, abseps = abseps, releps = releps)
            yy1[i] = 1-gb[0]
        """
        yy1 = 1-CDF(torch.ones(nintegr,m.size(-1))*(t1.unsqueeze(-1)-T),loc = -m,covariance_matrix=C,maxpts = maxpts, abseps = abseps, releps = releps)[0]

        x = torch.cat((t1,1.2*improvement_max.unsqueeze(-1)),-1)
        y = torch.cat((yy1,torch.zeros(1)),-1)
        spl = Interpolant(x,y,degree=2,init_right=True)
        ei = spl.integrate()
        if plo:
            xx = torch.linspace(x.min(),x.max(),100)
            yy = spl.predict(xx)
            plt.plot(xx,yy)
            plt.scatter(x,y)
            plt.show()
        #print(spline_integrate(t1,yy1,degree=2,plot=False))
        error = torch.tensor([0]) # TODO
    else:
        raise ValueError("The 'method=' should be either 'GenzBretz' or 'MonteCarlo'.")
    return ei, error

if __name__ == "__main__":


    for kk in range(1):
        print("ITE"+str(kk))
        #mu,cov = gpr._predict_f(Xc,noCov=False)
        n = 10
        m = randn(n)
        A = randn(n,n)
        C = matmul(A,A.t()) + n*torch.diag(torch.rand(n))
        s = torch.diagonal(C).sqrt()
        T = m.min().detach()
        p = MultivariateNormal(m,covariance_matrix=C) 
        # TODO make useless cholesky for GenzBretz


        rep = 10
        df = 2000000
        precisions = range(2000000,10000000,4000000)
        resu_ei_mc = np.empty((rep,len(precisions)))
        resu_tc_mc = np.empty((rep,len(precisions)))
        resu_er_mc = np.empty((rep,len(precisions)))
        resu_ei_ex = np.empty((rep,len(precisions)))
        resu_tc_ex = np.empty((rep,len(precisions)))
        resu_er_ex = np.empty((rep,len(precisions)))
        for j in range(len(precisions)):
            print(round(j/len(precisions)*100)/100)
            def foo(i):
                nintegr=(precisions[j])//df+5
                if i==0 :
                    print("nint:"+str(nintegr))
                time_start2 = time.time()
                qei_exa  = EI(p,T,nintegr=nintegr,maxpts=n*1000,abseps = 10**(-4)/nintegr,plo=i==0)
                rt2 = (time.time() - time_start2)
                resu_tc_ex[i,j] = rt2
                time_start3 = time.time()
                qei_MC  = EI(p,T,nmc=precisions[j],method="MonteCarlo")
                rt3 = (time.time() - time_start3)
                resu_tc_mc[i,j] =  rt3
                resu_ei_ex[i,j] = qei_exa[0]
                resu_ei_mc[i,j] = qei_MC[0]
                resu_er_ex[i,j] = qei_exa[1]
                #print("er"+str(qei_exa[1]))
                resu_er_mc[i,j] = qei_MC[1]
            #Parallel(n_jobs=3)(delayed(foo)(i) for i in range(rep))
            for i in range(rep):
                foo(i)

        ei_mc_m = np.nanmean(resu_ei_mc,0)
        ei_mc_s = np.nanstd(resu_ei_mc,0)
        ei_ex_m = np.nanmean(resu_ei_ex,0)
        ei_ex_s = np.nanstd(resu_ei_ex,0)

        ei_mc_er_m = np.nanmean(resu_er_mc,0)
        ei_ex_er_m = np.nanmean(resu_er_ex,0)
        
        plt.plot(precisions,ei_mc_m+2*ei_mc_s,c="blue")    
        plt.plot(precisions,ei_mc_m-2*ei_mc_s,c="blue")    
        plt.plot(precisions,ei_mc_m+ei_mc_er_m,c="blue",linestyle=":")    
        plt.plot(precisions,ei_mc_m-ei_mc_er_m,c="blue",linestyle=":")    
        plt.plot(precisions,ei_ex_m+2*ei_ex_s,c="red")    
        plt.plot(precisions,ei_ex_m-2*ei_ex_s,c="red")    
        plt.plot(precisions,ei_ex_m+ei_ex_er_m,c="red",linestyle=":")    
        plt.plot(precisions,ei_ex_m-ei_ex_er_m,c="red",linestyle=":")    
        plt.show()
        plt.plot(precisions,np.mean(resu_tc_mc,0),c="blue")
        plt.plot(precisions,np.mean(resu_tc_ex,0),c="red")    
        plt.show()

        plt.plot(np.mean(resu_tc_mc,0),ei_mc_s,c="blue")    
        plt.plot(np.mean(resu_tc_ex,0),ei_ex_s,c="red")   
        plt.show()



