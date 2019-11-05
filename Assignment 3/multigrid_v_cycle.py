import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3-d plot
from matplotlib import cm


def five_point(N):
    u = np.eye(N**2)
    Au = 4 * u

    Au[:,:-N] = Au[:,:-N] - u[:,N:]
    Au[:-N,:] = Au[:-N,:] - u[N:,:]

    diag = np.ones(N)
    diag[-1] = 0
    np.fill_diagonal(u, diag)

    Au[1:,:] = Au[1:,:] - u[:-1,:]

    diag = np.ones(N)
    diag[0] = 0
    np.fill_diagonal(u, diag)

    Au[:-1,:] = Au[:-1,:] - u[1:,:]

    return -Au

#A = five_point(3)
#print(A)

def rhs(N):
    N2 = N**2

    p = np.zeros((N2, 2))
    for i in range(1, N2):
        p[i, 0] = (i % N)/(N-1)
        p[i, 1] = math.floor(i/N)/(N-1)

    f = lambda x,y: math.sin(x*y)

    b = np.zeros(N2)
    for i in range(N2):
        b[i] = f(p[i,0], p[i,1])

    # Boundary conditions
    b[0:N] = 0                  # y=0
    b[N2-N:N2] = 0              # y=1
    b[0:N2:N] = 0               # x=0
    b[N-1:N2:N] = 0             # x=1

    b = np.reshape(b, (N, N))
    return b

print(rhs(5))


#Just a test function
def plot_mesh(N):
    b = rhs(N)

    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)

    fig = plt.figure(1)
    plt.clf()
    ax = fig.gca(projection='3d')
    X,Y = np.meshgrid(x, y)
    print(X)
    ax.plot_surface(X, Y, b, cmap=cm.coolwarm, rstride=1, cstride=1, linewidth=0)
    ax.view_init(azim=30)              # Rotate the figure
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

plot_mesh(10)


def solve_direct(rhs):
    """
    Insert code for solving the 2D poisson problem with right hand side rhs directly
    """

def jacobi(u0, rhs, w, nu):
    """
    u0: Initial guess
    rhs: Right-hand side, f
    w: Weights
    nu: Number of presmoothings
    """

    # Initializing variables
    N2 = u0.shape[0]
    N = int(np.sqrt(N2))
    nu = nu + 1

    I = np.eye(N2)
    A = five_point(N)
    Jw = I - 1/4 * w * A

    # Calculating the weighted Jacobi iteration
    u = Jw.dot(u0) + w * 1/4 * I * rhs

    return u, nu

b = rhs(4)
u0 = np.random.rand(4,4)
u, nu = jacobi(u0, b, 4/5, 1)
print(u)

def residual(u,rhs):
    """
    Insert code for calculating the residual
    """

def restriction(rh):
    """
    Insert code for restricting to coarser grid
    """

def interpolation(d2h):
    """
    Insert code for interpolating to finer grid
    """

def mgv(u0, rhs, nu1, nu2, level, max_level):
    """
    mgv(u0, rhs, nu1, nu2, level, max_level)
    performs one multi-grid V-cycle for the 2D Poisson problem
    on the unit square [0,1]x[0,1] with initial guess u0 and
    right-hand side rhs
    input: u0         - initial guess ((N-1)x(N-1) matrix
                        representing internal nodes)
           rhs        - right-hand side ((N-1)x(N-1) matrix)
           nu1        - number of presmoothings
           nu2        - number of postsmoothings
           level      - current level
           max_level  - total number of levels
    """
    if level == max_level:
        """
        Solve small problem exactly
        """
        return solve_direct(rhs)
    else:
        Nh = u0.shape[0] + 1
        u, nu1 = jacobi(u0, rhs, 0.8, nu1)

        rh = residual(u, rhs)
        r2h = restriction(rh)

        N2h = Nh // 2
        d2h = mgv(np.zeros((N2h-1, N2h-1)), r2h, nu1, nu2, level + 1, max_level)
        dh = interpolation(d2h)

        u = u + dh
        u = jacobi(u, rhs, 0.8, nu2)
    return u
