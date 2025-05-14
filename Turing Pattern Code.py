import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve

#set parameters
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
#growth = np.empty_like(ksq)

# 5) Sweep D_u and D_v
Du_vals = np.arange(0.001, 2.001, 0.05)
Dv_vals = Du_vals.copy()

records = []
for Du in Du_vals:
    for Dv in Dv_vals:
        # compute max real eigenvalue across ksq
        growth = [np.max(np.linalg.eigvals(J0 - np.diag([Du, Dv]) * k2).real) 
                  for k2 in ksq]
        # Turing instability criteria
        stable_no_diff    = growth[0] < 0
        unstable_with_diff = np.any(np.array(growth[1:]) > 0)
        turing_flag = int(stable_no_diff and unstable_with_diff)
        records.append({'Du': Du, 'Dv': Dv, 'turing': turing_flag})

# 6) Save results to Excel
df = pd.DataFrame(records)
excel_path = 'C:/Users/bella/OneDrive - Imperial College London/Python/Turing Pattens/turing_sweep.xlsx'
df.to_excel(excel_path, index=False)

