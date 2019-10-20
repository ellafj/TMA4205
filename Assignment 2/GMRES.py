import scipy
import numpy as np
from matplotlib import pyplot as plt
from scipy import io
import scipy.sparse as sp
from scipy.sparse.linalg import gmres

# Function that applies Givens rotations to an upper Hessenberg matrix, H
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

# Function that solves a linear system Ax = b with GMRES
def GMRES_with_rotations(A,b,m,tol):
    # Initializing variables
    n = A.shape[0]
    x0 = np.zeros(n)
    r0 = b - A.dot(x0)
    beta = np.linalg.norm(r0,2)
    betae = np.zeros(n)                     # = beta * e_1
    betae[0] = beta
    H = np.zeros((m+1,m))
    V = np.zeros((n,m+1))
    V[:,0] = r0/beta
    end_reached = False

    for j in range(m):
        w = A.dot(V[:,j])

        for i in range(j+1):
            H[i, j] = w.dot(V[:, i])
            w -= H[i, j]*V[:, i]

        H[j+1,j] = np.linalg.norm(w,2)

        if H[j+1,j] < tol:
            print('Found a solution')
            m = j
            end_reached = True
            break

        V[:,j+1] = w/H[j+1,j]

    # Checks if found exact solution
    if end_reached:
        # Solves exact solution
        y_m = np.linalg.solve(H[:m,:m], betae[:m])
        x_m = V[:,:m].dot(y_m)
        return x_m, y_m
    else:
        # If not, applying Given's rotations
        R_m, g_m = givens_rotations(H, m, beta)
        y_m = np.linalg.solve(R_m, g_m)
        x_m = V[:,:m].dot(y_m)
        return x_m, y_m


A = scipy.io.mmread('add32.mtx')
b = np.reshape(scipy.io.mmread('add32_rhs1.mtx'), A.shape[0])

m = [10, 50, 100, 150]
tol = 1E-18

plt.spy(A)
plt.show()

for i in m:
    x_m, y_m = GMRES_with_rotations(A, b, i, tol)
    print('The residual for m =',i,'is', np.linalg.norm((A.todense()).dot(x_m)-b))

