import numpy as np
import scipy
import scipy.io
import scipy.linalg
import scipy.sparse as sp
import cmath
import matplotlib.pyplot as plt

def CG(A, b, x, tol):
    r0 = b - A.dot(x)
    p = r0
    end_not_reached = True
    j = 0

    while end_not_reached:
        alpha = r0.dot(r0)/(A.dot(p)).dot(p)
        x += alpha * p
        r1 = r0 - alpha * A.dot(p)
        beta = r1.dot(r1) / r0.dot(r0)
        p = r1 + beta*p
        r0 = r1

        if np.linalg.norm(r1,2) < tol:
            end_not_reached = False
        else:
            j += 1

    return x, j

A = scipy.io.mmread('bcsstk16.mtx')
x0 = np.zeros(A.shape[0])
b = np.zeros(A.shape[0])
b[0] = 1

tol_list = [1E-5, 1E-10, 1E-15, 1E-20]

for tol in tol_list:
    x, j = CG(A, b, x0, tol)
    print('With a tolerance of', tol, 'I get:')
    print('- The solution of the system is x =', x)
    print('- Solution reached after', j, 'steps')
    print('- The residual of the system, ||b - Ax||, is', np.linalg.norm(b-A.dot(x),2), '\n')
