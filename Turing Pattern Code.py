import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import fsolve

#set parameters
N = 200 #grid size
Du, Dv = 0.01, 1.0
F, k = 0.035, 0.065
dt = 0.2
h = 1.0
alpha, beta = 0.1, 0.9

#discretised laplacian, boundary 
def laplacian(Z):
    return (
        -4*Z
        + np.roll(Z, +1, axis=0)
        + np.roll(Z, -1, axis=0)
        + np.roll(Z, +1, axis=1)
        + np.roll(Z, -1, axis=1)
    ) / h**2

#using the Gray-Scott model
def steady(vars):
    u, v = vars
    return [
        alpha - beta*u + u**2/v,
        u**2 - v
    ]
u0, v0 = fsolve(steady, [1.0, 1.0])

# direct Jacobian at (1,0)
f_u = -beta + 2*u0/v0    
f_v = -u0**2 / v0**2     
g_u =  2*u0        
g_v = -1.0 

J0 = np.array([[f_u, f_v],
               [g_u, g_v]])

ksq = np.linspace(0, 200, 1000)           # k^2 values to try
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
    print(f"✅ Turing instability! grows fastest at k^2 ≈ {k2_crit:.2f}, growth rate ≈ {growth.max():.3f}")
else:
    print("❌ No Turing instability in this parameter set.")