# A script to verify correctness
import sys
sys.path.append(".") 
#sys.path.append("./..") 
from mvnormpy import hyperrectangle_integral
from mvnormpy import multivariate_normal_cdf
import torch
from torch.distributions import MultivariateNormal
from torch import randn, matmul
import numpy as np
from numpy import sqrt
import matplotlib
import matplotlib.pylab as plt
import time
from joblib import Parallel, delayed
from warnings import warn

def multivariate_normal_CDF(x,loc,covariance_matrix=None,scale_tril=None,method="GenzBretz",nmc=200,  maxpts = 25000, abseps = 0.001, releps = 0, error_warning = True):
    if loc.size(-1)!=x.size(-1):
        raise ValueError("'x' does not have the same number of dimensions (" +str(x.size(-1))+ ") as 'loc' ("+str(loc.size(-1))+").")
    if method=="MonteCarlo": # Monte Carlo estimation
        p = MultivariateNormal(loc=loc,scale_tril=scale_tril,covariance_matrix=covariance_matrix)
        r = nmc%5
        N = nmc if r==0 else nmc + 5 - r # rounded to the upper multiple of 5
        Z = (p.sample(torch.Size([N]))<x).prod(-1)
        if error_warning: # Does not slow down significatively
            booleans = Z.view(*Z.shape[:-1],5,N//5) # divide in 5 groups to have an idea of the precision 
            values   = ((booleans.sum(-1).type(torch.float)))/N*5
            value = values.mean(-1).item()
            std = values.var(-1).sqrt().item()
            error = 1.96 * std / sqrt(5) # at 95 %
        else:
            value = Z.mean(-1).item()
            error = -1
    elif method == "GenzBretz": # Fortran routine
        value, error, info = hyperrectangle_integral(upper = x, mean = loc, sigma = covariance_matrix,scale_tril=scale_tril, maxpts = maxpts, abseps = abseps, releps = releps)
        # TODO warnings when error is higher than desired
    else:
        raise ValueError("The 'method=' should be either 'GenzBretz' or 'MonteCarlo'.")
    if error_warning and error > abseps:
            warn("Estimated error is higher than abseps. Consider raising the computation budget (nmc for method='MonteCarlo' or maxpts for 'GenzBretz'). Switch 'error_warning' to ignore.")
    return value, error



if __name__ == "__main__":
        
    n = 15

    for kk in range(1):
        print("ITE"+str(kk))
        #mu,cov = gpr._predict_f(Xc,noCov=False)
        x = randn(n)
        m = x.clone()-.1*np.sqrt(n)
        A = randn(n,n)
        C = torch.diag(torch.ones(n))+ 1/n*matmul(A,A.t())
        dist = MultivariateNormal(m, covariance_matrix=C)
        rep = 10


        df = 2000000
        precisions = range(2000000,10000000,500000)
        resu_ei_mc = np.empty((rep,len(precisions)))
        resu_tc_mc = np.empty((rep,len(precisions)))
        resu_er_mc = np.empty((rep,len(precisions)))
        resu_ei_ex = np.empty((rep,len(precisions)))
        resu_tc_ex = np.empty((rep,len(precisions)))
        resu_er_ex = np.empty((rep,len(precisions)))
        for j in range(len(precisions)):
            print(round(j/len(precisions)*100)/100)
            def foo(i):
                time_start2 = time.time()
                qei_exa = multivariate_normal_cdf(x,loc=m,covariance_matrix=C,maxpts=precisions[j]//df+10,error_info=False)
                rt2 = (time.time() - time_start2)
                resu_tc_ex[i,j] = rt2
                time_start3 = time.time()
                qei_MC = multivariate_normal_cdf(x,loc=m,covariance_matrix=C,maxpts=precisions[j]//df+10,error_info=True)#qei_MC  = multivariate_normal_CDF(x,loc=m,covariance_matrix=C,nmc=precisions[j],method="MonteCarlo")
                rt3 = (time.time() - time_start3)
                resu_tc_mc[i,j] =  rt3
                resu_ei_ex[i,j] = qei_exa[0]
                resu_ei_mc[i,j] = qei_MC[0]
                resu_er_ex[i,j] = qei_exa[1]
                print("er"+str(qei_exa[1]))
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

