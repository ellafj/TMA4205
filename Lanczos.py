import numpy as np
import scipy.io
import cmath
import scipy.linalg

def lanczos(A, v1, m):
    beta = np.zeros(m,dtype=complex)
    alpha = np.zeros(m,dtype=complex)
    V = np.zeros((m,m),dtype=complex)
    T = np.zeros((m,m),dtype=complex)
    V[0] = np.zeros(m,dtype=complex)
    V[1] = v1
    for j in range(1,m-1):
        w = A.dot(V[:,j])-beta[j]*V[j-1]
        alpha[j] = np.inner(w, V[:,j])
        w -= alpha[j]*V[:,j]
        beta[j+1] = np.linalg.norm(w,2)
        if beta[j+1] == 0:
            break
        else:
            V[:,j+1] = w / beta[j+1]

    for i in range(1,m):
        if i == 1:
            T[i-1, i-1] = alpha[i-1]
        else:
            T[i,i] = alpha[i]
        if i != m:
            T[i,i-1] = beta[i]
            T[i-1,i] = beta[i]

    return T, V


A = scipy.io.mmread('mhd1280b.mtx')
m = A.shape[0]
v = np.ones(m)

T, V = lanczos(A, v, m)
#print(V)
tau = 1
beta = np.zeros(m)
beta[0] = 1/np.sqrt(1280)


#print(np.exp(-z.imag*tau*T))
#u = V.dot(np.exp(-z.imag*tau*T)*beta)
#print(u)

unit_vec = np.zeros(m)
unit_vec[0] = 1.
u_tau = V.dot(scipy.linalg.expm(-1j*T*tau).dot(unit_vec))

print(u_tau)

