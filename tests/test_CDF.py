# A script to verify correctness
import sys
sys.path.append(".") 
#sys.path.append("./..") 
from mvnorm import hyperrectangle_integral
from mvnorm import multivariate_normal_cdf
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



if __name__ == "__main__":
    # Computation of P(Y<x), Y ~ N(m,C), by density integration.
    # Validation by Monte Carlo sampling.

    n = 15 # dimension of the random vector
    x = randn(n)
    m = x.clone()-.5*np.sqrt(n)
    # Compute an arbitrary covariance:
    A = randn(n,n)
    C = torch.diag(torch.ones(n))+ 1/n*matmul(A,A.t())

    # Number of times to repeat the experiment to check if the errors
    # are consistent with the provided error estimate.
    rep = 10

    # To experiment different levels of precision (nmc for MonteCarlo or maxpts for GenzBretz)
    precisions = range(150000,750000,50000)
    df = 150000 # ratio nmc/maxpts tested

    resu_ei_mc = np.empty((rep,len(precisions))) # store obtained proba for Monte Carlo method
    resu_tc_mc = np.empty((rep,len(precisions))) # store measured time
    resu_er_mc = np.empty((rep,len(precisions))) # store obtained error estimate
    resu_ei_ex = np.empty((rep,len(precisions))) # for GenzBretz method
    resu_tc_ex = np.empty((rep,len(precisions)))
    resu_er_ex = np.empty((rep,len(precisions)))
    for j in range(len(precisions)):
        if j%2==0:
            print("{:3}".format(int(round(j/len(precisions)*100)))+" % done.")
        for i in range(rep):
            time_start2 = time.time()
            qei_exa = multivariate_normal_cdf(upper=x,loc=m,covariance_matrix=C,maxpts=precisions[j]//df+10,error_info=True)
            rt2 = (time.time() - time_start2)
            resu_tc_ex[i,j] = rt2
            time_start3 = time.time()
            qei_MC = multivariate_normal_cdf(upper=x,loc=m,covariance_matrix=C,nmc=precisions[j],error_info=True,method="MonteCarlo")#qei_MC  = multivariate_normal_CDF(x,loc=m,covariance_matrix=C,nmc=precisions[j],method="MonteCarlo")
            rt3 = (time.time() - time_start3)
            resu_tc_mc[i,j] =  rt3
            resu_ei_ex[i,j] = qei_exa[0]
            resu_ei_mc[i,j] = qei_MC[0]
            resu_er_ex[i,j] = qei_exa[1]
            resu_er_mc[i,j] = qei_MC[1]



    # Compute mean, and standard deviations for values and error estimates
    ei_mc_m = np.nanmean(resu_ei_mc,0)
    ei_mc_s = np.nanstd(resu_ei_mc,0)
    ei_ex_m = np.nanmean(resu_ei_ex,0)
    ei_ex_s = np.nanstd(resu_ei_ex,0)

    ei_mc_er_m = np.nanmean(resu_er_mc,0)
    ei_ex_er_m = np.nanmean(resu_er_ex,0)
    
    

    plt.plot(precisions,ei_mc_m+2*ei_mc_s,c="blue",label="Monte Carlo (measured interval)")    
    plt.plot(precisions,ei_mc_m-2*ei_mc_s,c="blue")    
    plt.plot(precisions,ei_mc_m+ei_mc_er_m,c="blue",linestyle=":",label="Monte Carlo (code estimate)")    
    plt.plot(precisions,ei_mc_m-ei_mc_er_m,c="blue",linestyle=":")    
    plt.plot(precisions,ei_ex_m+2*ei_ex_s,c="red",label="Genz-Bretz (measured interval)")    
    plt.plot(precisions,ei_ex_m-2*ei_ex_s,c="red")    
    plt.plot(precisions,ei_ex_m+ei_ex_er_m,c="red",linestyle=":",label="Genz-Bretz (code estimate)")    
    plt.plot(precisions,ei_ex_m-ei_ex_er_m,c="red",linestyle=":")
    plt.legend()
    plt.xlabel("Number of MC samples (or GenzBretz's maxpts × "+str(df)+")")
    plt.ylabel("P(Y<x)")
    plt.title("Prediction intervals of CDF values (95 %)\n (in dimension "+str(n)+")")
    plt.show()

    plt.plot(precisions,np.mean(resu_tc_mc,0),c="blue",label="Monte Carlo")
    plt.plot(precisions,np.mean(resu_tc_ex,0),c="red",label="Genz-Bretz")
    plt.xlabel("Number of MC samples (or GenzBretz's maxpts × "+str(df)+")")
    plt.ylabel("Time (s)")
    plt.title("Average computation time")
    plt.legend()
    plt.show()

    plt.plot(np.mean(resu_tc_mc,0),ei_mc_s,c="blue",label="Monte Carlo")
    plt.plot(np.mean(resu_tc_ex,0),ei_ex_s,c="red",label="Genz-Bretz")  
    plt.title("Computation time versus precision\n(efficiency at bottom left)")
    plt.ylabel("Time (s)")
    plt.xlabel("Precision")
    plt.legend()
    plt.show()

