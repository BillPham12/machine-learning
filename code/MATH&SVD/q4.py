import numpy as np
from numpy import array
from scipy.linalg import svd
from numpy import linalg as Nor

A = np.array([
[3,2,-1,4],
[1,0,2,3],
[-2,-2,3,-1]])

# non square matrix doesn't have the inverse
# calculating the rank of the matrix A and its transpose
print("A has ", np.linalg.matrix_rank(A), " linearly indepedent columns")
print("A has ", np.linalg.matrix_rank(A.T), " linearly independent rows")

# get the pseudo inverse since the matrix is not square
A_Inverse = np.linalg.pinv(A)
print("the pseudo inverse of A:\n", A_Inverse)
