import numpy as np
from numpy import linalg as Nor
import random
import math

# define the matrix
a = np.array([[1,2,3],[2,3,4],
             [4,5,6],[1,1,1]])
b = np.array([1,1,1,1]).T

gama = 0.01

epsilion = [0.01, 0.05, 0.1,0.15,0.2,0.25,0.5]

table = [];

# ||A-T Ax - A-T b ||2  > gama
def function(x):
    return np.dot(a.T,np.dot(a,x)) - np.dot(a.T,b)

x = np.array([random.randrange(-2, 2), random.randrange(-2, 2), random.randrange(-2, 2)])
np.seterr(all='raise')
for eps in epsilion:
    iterations = 0
    current_x = x
    while Nor.norm(function(current_x),2) > gama:
        iterations = iterations +1
        # x = x - e (A-T Ax - A-T b)
        try:
            current_x = current_x - np.dot(eps,function(current_x))
        except FloatingPointError:
            break;
    table.append([eps, iterations, current_x])
for row in table:
    print(row)
