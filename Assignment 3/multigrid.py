import numpy as np
import math
np.set_printoptions(linewidth=500)

def lhs(u):
    Au = 4 * u
    Au[1:,:] = Au[1:,:] - u[:-1,:]
    Au[:-1,:] = Au[:-1,:] - u[1:,:]
    Au[:,1:] = Au[:,1:] - u[:,:-1]
    Au[:,:-1] = Au[:,:-1] - u[:,1:]
    return Au

def solve_direct(rhs):
    """
    Insert code for solving the 2D poisson problem with right hand side rhs directly
    Remember to add boundary conditions!!
    """
    

def jacobi(u0, rhs, w, nu):
    # Initializing variables
    n = u0.shape[0]
    Au = lhs(u0)

    # Calculating u_{j+1} = (I - 1/4 w A)u_j + f_w
    return np.eye(n).dot(u0) - 1/4 * w * Au + rhs


def residual(u,rhs):
    return rhs - lhs(u)


def restriction(rh):
    """
    rh: residual matrix
    """
    h = rh.shape[0]     # Original grid size
    H = (h+1)/2 - 1     # New coarser grid size

    # Checks if H is a whole number
    if H.is_integer():
        H = int(H)
    else:
        print('error')
        return 0

    # Initializing smaller grid
    rH = np.zeros((H, H))
    rectangle = np.zeros((H, h))

    # Transforming a (h x h)-matrix into a (H x h)-matrix
    for i in range(H):
        rows = rh[2*i:2*i+3, :]
        row = np.zeros(h)
        row += (rows[0] + 2*rows[1] + rows[2])/H
        rectangle[i] += row

    # Transforming a (H x h)-matrix into (H x H)-matrix
    for i in range(H):
        cols = rectangle[:, 2*i:2*i+3]
        rH[:,i] += (cols[:,0] + 2*cols[:,1] + cols[:,2])/H

    #print(rectangle)
    print(rH)


rh = np.random.rand(4,4)
print(rh, '\n')
#restriction(rh)

def interpolation(d2h):
    """
    d2h: u in coarse matrix
    """
    H = d2h.shape[0]
    h = 2*(H+1)-1

    # Initializing the finer matrix
    d2 = np.zeros((h, h))
    for i in range(1,H+1):
        # Initializing variables
        row = d2[2*i-1,:]
        row[1::2] = d2h[i-1,:]           # Hvorfor funker dette?
        col = d2[:, 2*i-1]
        col[1::2] = d2h[:,i-1]

        # Finding averages for the rows
        small_row = row[1::2]
        row[2:-2:2] = [(a+b)/2 for a, b in zip(small_row[::], small_row[1::])]

        # Finding averages for the columns
        small_col = col[1::2]
        col[2:-2:2] = [(a+b)/2 for a, b in zip(small_col[::], small_col[1::])]

    for i in range(H):
        # Finding averages for the midpoints
        midpoint_row = d2[2*i,:]
        upper_row = d2[2*i-1,:]
        lower_row = d2[2*i+1,:]
        small_upper_row = upper_row[1::2]
        small_lower_row = lower_row[1::2]
        midpoint_row[2:-2:2] = [(a+b+c+d)/4 for a, b, c, d in zip(small_upper_row[::], small_upper_row[1::], small_lower_row[::], small_lower_row[1::])]

    d2[0, :] = d2[1, :] / 2
    d2[-1, :] = d2[-2, :] / 2
    d2[:, 0] = d2[:, 1] / 2
    d2[:, -1] = d2[:, -2] / 2

    print(d2)

    return d2

interpolation(rh)


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
        Nh = u0.shape[0]+1
        u = jacobi(u0, rhs, 0.8,nu1)
        rh = residual(u, rhs)
        r2h = restriction(rh)
        N2h = Nh//2
        d2h = mgv(np.zeros((N2h-1,N2h-1)), r2h, nu1, nu2, level+1, max_level)
        dh = interpolation(d2h)
        u = u + dh
        u = jacobi(u, rhs, 0.8, nu2)
    return u
