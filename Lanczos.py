import numpy as np
import scipy.io
import cmath
import scipy.linalg
import matplotlib.pyplot as plt

def lanczos(A, v1, m, tol):
    # Initializing variables
    n = A.shape[0]
    alpha = np.zeros(m+1,dtype=complex)
    beta = np.zeros(m+2)

    v1 = v1/np.linalg.norm(v1,2)
    V = np.zeros((n,m+2), dtype=complex)
    V[:,1] = v1

    orthogonal_rows = 0
    tot_rows = 0

    # Constructing V
    for j in range(1,m+1):
        w = A.dot(V[:,j])-beta[j]*V[:,j-1]
        alpha[j] = w.dot(V[:,j])
        w -= alpha[j]*V[:,j]
        beta[j+1] = np.linalg.norm(w,2)
        if beta[j+1] <= tol:
            print('breaking')
            break
        else:
            V[:,j+1] = w/beta[j+1]

    # Constructing T
    T = np.zeros((m,m), dtype=complex)
    for j in range(m):
        T[j,j] = alpha[j+1]
        if j != m-1:
            T[j+1,j] = beta[j+2]
            T[j,j+1] = beta[j+2]

    T = scipy.sparse.csc_matrix(T)

    # Counting orthogonal vectors in V
    for i in range(m):
        for j in range(i):
            dotprod = np.abs(V[:,i].dot(V[:,j]))
            tot_rows += 1
            if dotprod < tol:
                orthogonal_rows += 1

    return T, V[:,1:-1], orthogonal_rows, tot_rows


A = scipy.io.mmread('mhd1280b.mtx')
n = A.shape[0]

m_list = [10, 50, 100, 150]
v1 = np.ones(n)
tau = 1

for m in m_list:
    T, V, orthogonal_rows, tot_rows = lanczos(A, v1, m, 1E-3)

    betae = np.zeros(m)
    betae[0] = 1/np.sqrt(1280)
    u_tau = V.dot(scipy.linalg.expm(-1j*T.multiply(tau)).dot(betae))

    plt.plot(u_tau.real, u_tau.imag, '.', label='m = ' + str(m))

    print('For m =', m, 'we get:')
    print('- Number of orthogonal rows out of total amount of rows:', orthogonal_rows, '/', tot_rows)
    print('- All values of u(tau) are in the interval [', min(u_tau), ',', max(u_tau), ']')

plt.xlabel('Re(u)')
plt.ylabel('Im(u)')
plt.legend()
plt.show()
