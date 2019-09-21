import pandas as pd
import numpy as np
from scipy import linalg

A = np.genfromtxt('D:\\Courses\\Fall 19\\ELEG 815 Statistical Learning\\Homeworks\\ratingsData.csv',delimiter=',')
A_t = A.transpose()

def Singular_Value_Decomposition(A):
    res = np.linalg.svd(A)
    U = res[0]
    Sigma = res[1]
    Vh = res[2]
    return(U,Sigma,Vh)

def K_Rank_Approximation(A,k):
    SVD = Singular_Value_Decomposition(A)
    U = SVD[0]
    Sigma_Array  =SVD[1]
    Vh = SVD[2]
    Sigma_Matrix = linalg.diagsvd(Sigma_Array,A.shape[0],A.shape[1])
    for i in range(min(Sigma_Matrix.shape)-k):
        Sigma_Matrix[k+i][k+i] = 0                      # Choose k top eigen values or singular values(Or concepts/categories in this case)
    A_hat = np.linalg.multi_dot([U,Sigma_Matrix,Vh])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i][j] != 0:
                A_hat[i][j] = A[i][j]              # Put originial non zero ratings from Matrix A into A_hat

    #---------------------- Mean Squared Error After Prediction --------------------------
    testData = np.genfromtxt('D:\\Courses\\Fall 19\\ELEG 815 Statistical Learning\\Homeworks\\ratingsTest.csv',delimiter=',')
    SE_Sum = 0
    n = 0
    for index,value in np.ndenumerate(testData):    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndenumerate.html
        if value !=0:
            i = index[0]
            j = index[1]
            SE_Sum += (A_hat[i][j] - value)**2
            n += 1
    MSE = SE_Sum/n
    #--------------------- Mean Squared Error before prediction --------------------------
    SE_Sum_Initial = 0
    n_b = 0
    for index,value in np.ndenumerate(testData):
        if value !=0:
            i = index[0]
            j = index[1]
            SE_Sum_Initial += (A[i][j] - value)**2
            n_b += 1
    MSE_Initial = SE_Sum_Initial/n_b
    return (MSE,MSE_Initial)

#print(K_Rank_Approximation(A,4))
K_list =[4,6,8,100]
Errors = []

def Compute_SVD(A,K_list):
    for i in K_list:
        errs = K_Rank_Approximation(A,i)
        Errors.append(errs)
Compute_SVD(A,K_list)
print(Errors)



