import numpy as np

def solve_direct(rhs):
    print('not done')
    return rhs

def jacobi(u0, rhs, weight, nu1):
    print('not done')
    return u0

def residual(u, rhs):
    print('not done')
    return rhs

def restriction(rh):
    print('not done')
    return rh

def interpolation(d2):
    print('not done')
    return d2


def mgv(u0, rhs, nu1, nu2, level, max_level):
    """
    mgv(u0, rhs, nu1, nu2, level, max_level)
    performs one multi-grid V-cycle for the 2D Poisson problem on the unit square [0,1]x[0,1] with initial guess u0 and right-hand side rhs

    input:
    u0 - initial guess ((N-1)x(N-1) matrix representing internal nodes)
    rhs - right-hand side ((N-1)x(N-1) matrix
    nu1 - number of presmoothings
    nu2 - number of postsmoothings
    level - current level
    max_level - total number of levels
    """

    if level == max_level:
        """ 
        Solve small problem exactly 
        """
        return solve_direct(rhs)
    else:
        Nh = u0.shape[0]+1
        u = jacobi(u0, rhs, 0.8, nu1)
        rh = residual(u, rhs)
        r2h = restriction(rh)
        N2h = Nh//2
        d2h = mgv(np.zeros((N2h-1,N2h-1)), r2h, nu1,nu2,level+1, max_level)
        dh = interpolation(d2h)
        u = u + dh
        u = jacobi(u, rhs, 0.8, nu2)
    return u
