from numpy import  pi,  arctan, arctan2, log,  exp, inf
from scipy.integrate import quad, romberg, fixed_quad

def r(z, p):
    return (1. - exp(-z))**p

def k_0(n):
    return 1./(4.*pi)*(11. - 2./3.*n)

def k_1(n):
    return 1./(4.*pi)*(102. - 38./3.*n)/(11. - 2./3.*n)

def funct_rho_0(z, n, p):
    return -pi*k_0(n)*r(z, p)

def funct_e0(z, n, p, min_limit=0, max_limit=inf):
    expr = lambda x1: funct_rho_0(x1, n, p) / (x1 + 1) / (x1 + z)
    int1 = quad(expr, min_limit, max_limit, limit=500, epsabs=1e-7)
    return (1. - z)*int1[0] / pi  # , int1[1]/int[0]

def funct_epsilon_0(s, lambda1, n, p, min_limit=0, max_limit=inf):
    e0_s = funct_e0(s/lambda1**2, n, p, min_limit, max_limit)
    e0_0 = funct_e0(0, n, p, min_limit, max_limit)
    return e0_s - e0_0

def funct_f_part(x):
    if x > 1:
        return -arctan2(pi,log(x)) # to check the sign
    else:
        return -(pi + arctan2(pi,log(x)))

def funct_rho_1(z, n, p=0.8):
    return k_1(n)*r(z, p)*funct_f_part(z)

def funct_e1(z, n, p, min_limit=0, max_limit=inf):
    expr = lambda x1: funct_rho_1(x1, n, p) / (x1 + 1) / (x1 + z)
    int1 = quad(expr, min_limit, max_limit, limit=500, epsabs=1e-7)
    return (1. - z)*int1[0] / pi  # , int1[1]/int[0]

def funct_epsilon_1(s, lambda1, n, p, min_limit=0, max_limit=inf):
    e1_s = funct_e1(s/lambda1**2, n, p)
    e1_0 = funct_e1(0, n, p, min_limit, max_limit)
    return e1_s - e1_0

def funct_epsilon(s, lambda1, n, p, min_limit=0, max_limit=inf):
    return (funct_epsilon_0(s, lambda1, n, p, min_limit, max_limit)
            + funct_epsilon_1(s, lambda1, n, p, min_limit, max_limit))

def funct_alpha_disp(s, lambda1, n, p, min_limit=0, max_limit=inf):
    return 1./funct_epsilon(s, lambda1, n, p, min_limit, max_limit)