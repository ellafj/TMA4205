import numpy as np
import scipy.io
import scipy.sparse as sparse

#A = scipy.sparse.coo_matrix([[0,1,0,0,1,0], [1,0,1,0,0,0], [0,0,0,0,0,1], [1,0,0,0,1,0], [1,0,0,1,0,0], [1,1,1,1,1,1]])

def power_iteration(d, tol):
    # Initializing variables
    A = scipy.io.mmread('stanford.mtx')
    N = A.get_shape()
    x0 = np.array(np.ones((N[0],1))) / (N[0])
    print('x0', x0.shape)

    # Calculating B-matrix
    k = 1/sparse.csc_matrix.getnnz(A.tocsc(),axis=0)
    B = A.multiply(k)

    #print('hello')
    #x = np.array(np.ones((N[0],1)))

    r = 1

    while r > tol:
        # Mx = M1 + M2
        M1 = (1-d)*sum(x0)*np.ones((N[0],1))
        #print(M1.get_shape())
        M2 = 0.25 * B.dot(x0)
        #print(M2.get_shape())
        Mx = M1 + M2
        #print(Mx.get_shape())
        x = Mx / np.linalg.norm(Mx)
        print('x', x.shape)
        r = np.linalg.norm(x - x0)
        #print(r)
        x0 = x

    return x0

#print(np.ones((5,1)))

x = power_iteration(0.85, 1E-8)
print(x.shape)
print(np.max(x))
indexes = np.where(x == np.max(x))
print(indexes[0])
