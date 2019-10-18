import scipy as sc
import numpy as np
from matplotlib import pyplot as plt

A = np.zeros((5,5))
for i in range(5):
    if i == 0:
        A[0,-1] = 1
    else:
        A[i, i-1] = 1

print(A)

x0 = np.zeros(5)

b = np.zeros(5)
b[0] = 1

def GMRES(A, b, x0):
    m = len(A)
    j = 0

    r0 = b - A.dot(x0)
    beta = np.linalg.norm(r0,2)
    end_not_reached = True

    v = np.zeros((m+1,m))
    v[0] = r0/beta
    w = []
    H = np.zeros((m+1,m))

    while end_not_reached:
        w.append(A.dot(v[j]))

        for i in range(j+1):
            H[i,j] = np.inner(w[j], v[i])
            w[j] = w[j] - H[i,j]*v[i]

        h = np.linalg.norm(w[j], 2)

        if int(h) == 0:
            print('Found H')
            print(m)
            print(H)
            plt.spy(H)
            #plt.show()
            #break
            return H, beta

        if j != j-1:
            H[j+1,j] = h
            v[j+1] = w[j]/H[j+1,j]

        print('H',H)
        j += 1

H, beta = GMRES(A,b,x0)

def givens_rotations(H, beta, m):
    m = len(A)
    r0 = b - A.dot(x0)
    beta = np.linalg.norm(r0,2)

    submatrix = np.zeros((2,2))
    s1 = H[1,0]/np.sqrt(H[0,0]**2+H[1,0]**2)
    c1 = H[0,0]/np.sqrt(H[0,0]**2+H[1,0]**2)
    submatrix[0,1] = s1
    submatrix[1,0] = -s1
    submatrix[0,0] = c1
    submatrix[1,1] = c1

    g = np.zeros(m+1)
    g[0] = beta

    H_list = [H]
    g_list = [g]

    for i in range(m):
        omega = np.eye(m+1)
        omega[i:i+2,i:i+2] = submatrix
        print(H[i])
        H_list.append(omega.dot(H_list[i]))
        g_list.append(omega.dot(g_list[i]))
    print('H_list', H_list)
    print('g_list', g_list)




givens_rotations(H, beta, 5)

