import numpy as np
from scipy.linalg import svd
# define the matrix
a = np.array([[1,2,3],[2,3,4],
             [4,5,6],[1,1,1]])

# setting up print options
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

print("---orthogonal-matrix-U---diagonal-matrix---orthogonal-matrix-V---")
# compute the SVD
U, s, V = svd(a, full_matrices=True)
# print the value of U,S,V
print ("U:\n {}".format(U))
print ("S:\n {}".format(s))
print ("V:\n {}".format(V))
