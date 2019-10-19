import scipy
import numpy as np
from matplotlib import pyplot as plt
from scipy import io
import scipy.sparse as sp

A = scipy.io.mmread('add32.mtx')
b = np.reshape(scipy.io.mmread('add32_rhs1.mtx'), A.shape[0])

# Function that applies givens rotations to an upper Hessenberg matrix, H
def givens_rotations(H, m, beta):
    # Initializing variables
    submatrix = np.zeros((2,2))
    g = np.zeros(m+1)
    g[0] = beta
    H_list = [H]
    g_list = [g]

    # Calculating first rotation
    s1 = H[1,0]/np.sqrt(H[0,0]**2+H[1,0]**2)
    c1 = H[0,0]/np.sqrt(H[0,0]**2+H[1,0]**2)

    submatrix[0,1] = s1
    submatrix[1,0] = -s1
    submatrix[0,0] = c1
    submatrix[1,1] = c1

    omega = np.eye(m+1)
    omega[0:2,0:2] = submatrix
    H_list.append(omega.dot(H))
    g_list.append(omega.dot(g))

    # Applies the rest of the rotations
    for i in range(2,m):
        # Initializing variables
        H_new = H_list[i-1]
        H_old = H_list[i-2]
        s = H_new[i,i-1]/np.sqrt(H_old[i,i]**2+H_new[i,i-1]**2)
        c = H_old[i,i]/np.sqrt(H_old[i,i]**2+H_new[i,i-1]**2)

        submatrix[0,1] = s
        submatrix[1,0] = -s
        submatrix[0,0] = c
        submatrix[1,1] = c

        # Applying rotations
        omega = np.eye(m+1)
        omega[i-1:i+1,i-1:i+1] = submatrix
        H_list.append(omega.dot(H_list[i-1]))
        g_list.append(omega.dot(g_list[i-1]))
    
    Rm = H_list[-1]
    gm = g_list[-1]

    return Rm[:-1], gm[:-1]


def testing():
    A = np.zeros((6,5))
    for i in range(5):
        if i == 0:
            A[i,-1] = 1
        else:
            A[i,i-1] = 1
    b = np.zeros(A.shape[0])
    b[0] = 1
    x0 = np.zeros(5)
    r0 = b - A.dot(x0)
    beta = np.linalg.norm(r0,2)
    betae = np.zeros(5)
    betae[0] = beta

    H, g = old_givens_rotations(A, 5, beta)
    print(H)
    print(g)

#testing()

def GMRES_with_rotations(A,b,m,tol):
    n = A.shape[0]
    x0 = np.zeros(n)
    r0 = b - A.dot(x0)
    beta = np.linalg.norm(r0,2)
    print('initial residual was', beta)

    betae = np.zeros(n)
    betae[0] = beta

    H = np.zeros((m+1,m))
    V = np.zeros((n,m+1))
    V[:,0] = r0/beta
    end_not_reached = True

    for j in range(m):
        w = A.dot(V[:,j])

        for i in range(j+1):
            H[i, j] = w.dot(V[:, i])
            w -= H[i, j]*V[:, i]

        H[j+1,j] = np.linalg.norm(w,2)

        if H[j+1,j] < tol:
            print('Found a solution')
            m = j
            end_not_reached = False
            break

        V[:,j+1] = w/H[j+1,j]

    if end_not_reached:
        R_m, g_m = givens_rotations(H, m, beta)
        y_m = np.linalg.solve(R_m, g_m)
        x_m = V[:,:m].dot(y_m)
        return x_m, y_m, m
    else:
        y_m = np.linalg.solve(H[:m,:m], betae[:m])
        x_m = V[:,:m].dot(y_m)
        return x_m, y_m, m


x_m, y_m, m = GMRES_with_rotations(A, b, 500, 1E-18)
print('residual was', np.linalg.norm((A.todense()).dot(x_m)-b))

