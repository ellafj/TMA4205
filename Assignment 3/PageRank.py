import numpy as np
import scipy.io
import scipy.sparse as sparse
import matplotlib.pyplot as plt

def power_iteration(d, tol):
    # Initializing variables
    A = scipy.io.mmread('stanford.mtx')
    N = A.get_shape()
    residuals = [1]
    r = 1
    iter = 0
    x0 = np.array(np.ones((N[0],1))) / (N[0])

    # Calculating B-matrix
    k = 1/sparse.csc_matrix.getnnz(A.tocsc(),axis=0)
    B = A.multiply(k)

    while r > tol:
        iter += 1

        M1 = (1-d)*sum(x0)*np.ones((N[0],1))/N[0]
        M2 = d * B.dot(x0)
        Mx = M1 + M2

        x = Mx / np.linalg.norm(Mx)
        r = np.linalg.norm(x - x0)
        residuals.append(r)
        x0 = x

    return x0, residuals, iter

# Please give up to a minute for this to run
d = 0.85
tol = 1E-8
x, residuals, iter = power_iteration(d, tol)

print('Used %d iterations to obtain a tolerance of %s' % (iter, tol), '\n')
print('The most visited site has index', (np.where(x == np.max(x)))[0])

iterations = np.linspace(0, iter, len(residuals))
plt.semilogy(iterations, residuals, 'k.')
plt.xlabel('Iterations')
plt.ylabel('Residuals')
plt.title('Convergence of Residuals')
plt.show()
