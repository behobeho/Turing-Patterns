import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Load the refined Du-Dv scan results
df = pd.read_excel('C:/Users/bella/OneDrive - Imperial College London/Python/Turing Pattens/turing_scan_005.xlsx')

# 2) Pivot to grid form: rows = Dv, columns = Du, values = turing flag (0 or 1)
pivot = df.pivot(index='Dv', columns='Du', values='turing').astype(int)

# 3) Prepare tick values at 0.0, 0.1, ..., up to the max of Du/Dv
du_vals = pivot.columns.values
dv_vals = pivot.index.values
tick_vals = np.around(np.arange(0, max(du_vals.max(), dv_vals.max()) + 0.1, 0.1), 1)

# Find nearest index positions for ticks
xticks = [np.argmin(np.abs(du_vals - tv)) for tv in tick_vals]
yticks = [np.argmin(np.abs(dv_vals - tv)) for tv in tick_vals]

# 4) Plot heatmap
fig, ax = plt.subplots(figsize=(8, 6))
cmap = ListedColormap(['white', 'blue'])  # white = no instability, blue = instability
ax.imshow(pivot.values, origin='lower', aspect='auto', cmap=cmap)

# 5) Add binary legend
legend_handles = [
    Patch(facecolor='white', edgecolor='black', label='No Turing Instability'),
    Patch(facecolor='blue',  edgecolor='black', label='Turing Instability')
]
ax.legend(handles=legend_handles, loc='upper right')

# 6) Configure ticks
ax.set_xticks(xticks)
ax.set_xticklabels([f"{tv:.1f}" for tv in tick_vals], rotation=90)
ax.set_yticks(yticks)
ax.set_yticklabels([f"{tv:.1f}" for tv in tick_vals])

# 7) Labels and title
ax.set_xlabel('$D_u$')
ax.set_ylabel('$D_v$')
ax.set_title('Turing Instability Heatmap for Gierer-Meinhardt Model')

plt.tight_layout()
plt.show()
