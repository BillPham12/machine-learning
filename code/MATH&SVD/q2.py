import numpy as np
from scipy.linalg import svd
from numpy import linalg as Nor
# define the matrix
denta = 0.001
list = []
num = 1402
# store the array
for x in range (1,num):
    ar = [];
    x_value = -0.7 + denta*(x-1)
    for y in range(1,num):
        y_value = -0.7 + denta*(y-1)
        ar.append((1-x_value**2-y_value**2)**(1/2))
    list.append(ar)

# put into numpy array
a = np.array(list)
# setting up print options
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

# caculating SVD
print("---orthogonal-matrix-U---diagonal-matrix---orthogonal-matrix-V---")
# compute the SVD
U, s, VT = np.linalg.svd(a, full_matrices=False)
# print the value of U,S,V
#print ("U:\n {}".format(U))
#print ("S:\n {}".format(s))
#print ("VT:\n {}".format(VT))

# calculating the rank 2 approximation by using SVD's value
Ar = np.zeros((len(U), len(VT)))
for i in range(0,2):
    Ar += s[i] * np.outer(U.T[i], VT[i])
# calculating || A2 - A||
print(Ar)
print(Nor.norm(a-Ar))
