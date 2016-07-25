import numpy as np
from scipy.sparse.linalg import LinearOperator

N = 1e8

def mv(v):
    return np.concatenate((v,v,v),axis=0)

v = np.ones(N)
A = LinearOperator((3*N,N), matvec=mv)
A

# print A.matvec(v)

# print A * v
