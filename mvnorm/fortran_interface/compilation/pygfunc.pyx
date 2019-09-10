from numpy import zeros, ones, int32, empty
from numpy cimport ndarray as ar

cdef extern from "pygfunc.h" nogil:
    void c_gfunc(int* n,int* nn, double* lower,double* upper,int* infin, double* correl, double* delta, int* maxpts, double* abseps, double* releps,double* error,double* value,int* inform)




def f(int d,ar[double] lower, ar[double] upper, ar[int] infin,ar[double] correl,int maxpts,double abseps, double releps, int info):
    cdef:
        int dd = d*(d-1)/2
        ar[double] delta = zeros(d)
        double error  = 42
        double value  = 42
        int inform = 4
    
    with nogil:
        c_gfunc(&d,&dd, <double*> lower.data, <double*> upper.data, <int*> infin.data, <double*> correl.data, <double*> delta.data,&maxpts, &abseps, &releps,&error,&value,&inform)
    if info==1:
        return value, error, inform
    else:
        return value




    #for i in range(d):
    #    for j in range(i):
    #        correl[j + i * (i-1) / 2] = c[i, j]
    ####
