import scipy as sc
import numpy as np
from matplotlib import pyplot as plt
import scipy.sparse.linalg
from scipy import io
import scipy.sparse as sp

"""
A = np.zeros((5,5))
for i in range(5):
    if i == 0:
        A[0,-1] = 1
    else:
        A[i, i-1] = 1

print(A)

b = np.zeros(5)
b[0] = 1"""

A = scipy.io.mmread('add32.mtx')
b = np.reshape(scipy.io.mmread('add32_rhs1.mtx'), A.shape[0])
x0 = np.zeros(A.shape[0])
print(x0)

def givens_rotations(H, m, r0, beta):
    #m = scipy.sparse.csr_matrix.getnnz(A) #len(A)
    #r0 = b - A.dot(x0)
    #beta = np.linalg.norm(r0,2)

    submatrix = np.zeros((2,2))

    g = np.zeros(m+1)
    g[0] = beta

    H_list = [H]
    g_list = [g]

    for i in range(m-1):
        s1_under = np.sqrt(H[i,i]**2+H[i+1,i]**2)
        if s1_under == 0:
            s1 = 0
        else:
            s1 = H[i+1,i]/np.sqrt(H[i,i]**2+H[i+1,i]**2)
        c1_under = np.sqrt(H[i,i]**2+H[i+1,i]**2)
        if c1_under == 0:
            c1 = 0
        else:
            c1 = H[i,i]/np.sqrt(H[i,i]**2+H[i+1,i]**2)
        submatrix[0,1] = s1
        submatrix[1,0] = -s1
        submatrix[0,0] = c1
        submatrix[1,1] = c1

        omega = np.eye(m+1)
        omega[i:i+2,i:i+2] = submatrix
        print('H[i]', H[i])
        H_list.append(omega.dot(H_list[i]))
        g_list.append(omega.dot(g_list[i]))

    R_m = H_list[-1]
    g_m = g_list[-1]

    return R_m[:-1], g_m[:-1]

def GMRES(A, b, x0):
    m = A.shape[0] #scipy.sparse.csr_matrix.getnnz(A) #len(A)
    print('m', m)
    j = 0

    print('hello')
    r0 = b - A *x0
    print('r0', r0)
    beta = np.linalg.norm(r0,2)
    print('beta', beta)
    end_not_reached = True

    print('hello1')
    V = np.zeros((m+1,m))
    V[0] = r0/beta
    print('hello2')
    w = []
    H = np.zeros((m+1,m))
    print('hello3')

    while end_not_reached:
        print('j', j)
        w.append(A.dot(V[j]))

        for i in range(j+1):
            H[i,j] = np.inner(w[j], V[i])
            w[j] = w[j] - H[i,j]*V[i]

        h = np.linalg.norm(w[j], 2)

        if int(h) == 0:
            print('Found H')
            print('m', m)
            H[H < 10**(-10)] = 0
            print('H', H)
            #plt.spy(H)
            #plt.show()
            R_m, g_m = givens_rotations(H, m, r0, beta)
            y_m = np.linalg.solve(R_m, g_m)
            x_m = x0 + V[:-1].dot(y_m)
            return x_m, y_m

        if j != j-1:
            H[j+1,j] = h
            V[j+1] = w[j]/H[j+1,j]

        print('H',H)
        j += 1

x_m, y_m = GMRES(A,b,x0)
print('xm', x_m)
#print('ym', y_m)

print('x', sc.sparse.linalg.gmres(A, b, x0))

#givens_rotations(H, beta, 5)

