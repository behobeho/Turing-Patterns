import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import fsolve

#set parameters
N = 200 #grid size
Du, Dv = 0.01, 1.0 #diffusivity of morphogens
F, k = 0.035, 0.065 
h = 1.0
alpha, beta = 0.1, 0.9 #gierer-meinhardt parameters
ksq_max, n_k = 200, 1000

#discretised laplacian, boundary 
def laplacian(Z):
    # pad with edge values so ∂Z/∂n = 0 at boundaries
    Zp = np.pad(Z, pad_width=1, mode='edge')
    # apply the standard 5-point stencil to the padded array
    return (
        Zp[:-2, 1:-1] +  # one above
        Zp[2: , 1:-1] +  # one below
        Zp[1:-1, :-2] +  # one left
        Zp[1:-1, 2: ] -  # one right
        4 * Zp[1:-1, 1:-1]  # center
    ) / h**2

#using the Gierer-Meinhardt model
def steady(vars):
    u, v = vars
    return [
        alpha - beta*u + u**2/v,
        u**2 - v
    ]

u0, v0 = fsolve(steady, [alpha+beta, (alpha+beta)**2])

# direct Jacobian at (u0,v0)
f_u = -beta + 2*u0/v0    
f_v = -u0**2 / v0**2     
g_u =  2*u0        
g_v = -1.0 

J0 = np.array([[f_u, f_v],
               [g_u, g_v]])

ksq = np.linspace(0, ksq_max, n_k)           # k^2 values to try
growth = np.empty_like(ksq)

for i, k2 in enumerate(ksq):
    Jk = J0 - np.diag([Du, Dv]) * k2
    eigs = np.linalg.eigvals(Jk)
    growth[i] = np.max(eigs.real)

stable_no_diff = growth[0] < 0
unstable_with_diff = np.any(growth[1:] > 0)

if stable_no_diff and unstable_with_diff:
    # find first unstable k
    k2_crit = ksq[np.argmax(growth)]
    print(f"Turing instability! grows fastest at k^2 ≈ {k2_crit:.2f}, growth rate ≈ {growth.max():.3f}")
else:
    print("No Turing instability in these parameters.")