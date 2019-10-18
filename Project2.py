import scipy as sc
import numpy as np
from matplotlib import pyplot as plt
import scipy.sparse.linalg
from scipy import io
import scipy.sparse as sp


A = np.zeros((5,5))
for i in range(5):
    if i == 0:
        A[0,-1] = 1
    else:
        A[i, i-1] = 1

print(A)

b = np.zeros(5)
b[0] = 1
"""

A = scipy.io.mmread('add32.mtx')
b = np.reshape(scipy.io.mmread('add32_rhs1.mtx'), A.shape[0])
"""
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
    m = A.shape[0]
    j = 0

    r0 = b - A *x0
    beta = np.linalg.norm(r0,2)
    end_not_reached = True

    V = np.zeros((m+1,m))
    V[0] = r0/beta
    w = []
    H = np.zeros((m+1,m))

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
            y_m = np.linalg.solve(R_m, g_m)
            x_m = x0 + V[:-1].dot(y_m)
            return x_m, y_m

        if j != j-1:
            H[j+1,j] = h
            V[j+1] = w[j]/H[j+1,j]

        print('H',H)
        j += 1

#x_m, y_m = GMRES(A,b,x0)
#print('xm', x_m)
#print('ym', y_m)

print('x', sc.sparse.linalg.gmres(A, b, x0))

def GMRES_with_rotations(A,b,x0):
    m = A.shape[0]
    iter = 0

    r0 = b - A.dot(x0)
    beta = np.linalg.norm(r0,2)
    end_not_reached = True

    V = np.zeros((m+1,m))
    V[0] = r0/beta
    w = []
    H = np.zeros((m+1,m))

    x_m = 0
    y_m = 0

    while end_not_reached:
        print('iter', iter)

        for j in range(m):
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
                return x_m, y_m

            if j != j-1:
                H[j+1,j] = h
                V[j+1] = w[j]/H[j+1,j]

            print('j', j)
            if j > 1:
                omega = np.eye(j+1)
                print('omega', omega)
                s = H[j+1,j]/np.sqrt(H[j,j]**2+H[j+1,j]**2)
                c = H[j,j]/np.sqrt(H[j,j]**2+H[j+1,j]**2)
                omega[j,j+1] = s
                omega[j+1,j] = -s
                omega[j,j] = c
                omega[j+1,j+1] = c

                H[:j+1, :j] = omega.dot(H[:j+1,:j])
                g[:j+1] = omega.dot(g[:j+1])

                y_m = scipy.sparse.linalg.spsolve(H[:j,:j], g[:j])
                x_m = x0 + V[:-1].dot(y_m)

            print('H',H)

        iter += 1


def GMRES_with_rotations2(A,b,x0,m,iterations):

    r0 = b - A.dot(x0)
    r_norm = np.zeros(A.shape[0])
    r_norm[0] = np.linalg.norm(r0,2)
    iter = 0
    end_reached = False

    for k in range(iterations):
        V = np.zeros((b.shape[0],m+1))
        H = np.zeros((m+1,m))
        r = b - A.dot(x0)
        beta = np.linalg.norm(r,2)
        V[:,0] = r/beta
        g = np.zeros((m+1, 1))
        g[0] = beta

        for j in range(1,m):
            w = A.dot(V[:,j])

            for i in range(j):
                H[i,j] = w.dot(V[:,i])
                w = w - H[i,j]* V[:,i]

            H[j+1,j] = np.linalg.norm(w,2)

            if H[j+1,j] == 0:
                end_reached = True

            V[:, j+1] = w / H[j+1, j]
            print('H[j+1, j]', H[j+1, j])

            ## Adding Givens rotations

            if end_reached:
                H[j,j] = 1

            else:
                omega = np.eye(j)
                print('omega', omega)
                s = H[j+1,j]/np.sqrt(H[j,j]**2+H[j+1,j]**2)
                c = H[j,j]/np.sqrt(H[j,j]**2+H[j+1,j]**2)
                omega[j,j+1] = s
                omega[j+1,j] = -s
                omega[j,j] = c
                omega[j+1,j+1] = c

                H[:j+1, :j] = omega.dot(H[:j+1,:j])
                g[:j+1] = omega.dot(g[:j+1])

            y_m = scipy.sparse.linalg.spsolve(H[:j, :j], b[:j])
            x_m = V[:, j].dot(y_m)

            r_norm[(k-1)*m+j] = np.abs(g[j+1])
            iter += 1
            if r_norm[(k-1)*m+j]/r_norm[0] < tol or end_reached:
                return x_m, r_norm, iter



x_m, y_m = GMRES_with_rotations(A, b, x0)
print(x_m)

