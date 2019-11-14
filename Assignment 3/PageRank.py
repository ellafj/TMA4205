import numpy as np
import scipy.io
import scipy.sparse as sparse

#A = scipy.sparse.coo_matrix([[0,1,0,0,1,0], [1,0,1,0,0,0], [0,0,0,0,0,1], [1,0,0,0,1,0], [1,0,0,1,0,0], [1,1,1,1,1,1]])

def power_iteration(x0, d, tol):
    # Initializing variables
    A = scipy.io.mmread('stanford.mtx')
    N = A.get_shape()

    # Calculating B-matrix
    k = 1/sparse.csc_matrix.getnnz(A.tocsc(),axis=0)
    B = A.multiply(k)

    print('hello')
    x0 = np.ones((N[0],1))

    # Mx = M1 + M2
    M1 = (1-d)*sum(x0)*np.ones((N[0],1))
    M2 = 0.25 * B.multiply(x0)
    Mx = M1 + M2

    r = 1
    print(Mx)

    #while r > tol:
        #w = Mx

#print(np.ones((5,1)))
x0 = np.ones((5,1))

power_iteration(x0, 0.85, 10)
