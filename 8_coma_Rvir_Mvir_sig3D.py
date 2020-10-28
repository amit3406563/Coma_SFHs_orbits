# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 00:51:02 2020

@author: amit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def format_float_arr(arr):
    new_arr = np.array([])
    for ar in arr:
        new_arr = np.append(new_arr,'{:.2f}'.format(ar))
    return new_arr

def format_exp_arr(arr):
    new_arr = np.array([])
    for ar in arr:
        new_arr = np.append(new_arr,'{:.2e}'.format(ar))
    return new_arr

Rvir = 2.9 # virial radius in Mpc of Coma cluster from Lokas and Mamon 2003
# if virial is defined in 200c instead of 360b, this conversion is used:
# r200c/r360b = 0.73

# uncertainty in Rvir
uRvir = 0.3 # in Mpc
Rvirp = Rvir + uRvir
Rvirm = Rvir - uRvir


# Mvir = (4/3 * pi * Rvir^3 * (360 * omega_m * rho_c)) / M_sol
Mvir = ((4 / 3 * np.pi * (Rvir * 10**6 * 3.086 * 10**16)**3
        * 8.62 * 10**(-27) * 0.307 * 360) / (1.98 * 10**30))

# uncertainty in Mvir
Mvirp = ((4 / 3 * np.pi * (Rvirp * 10**6 * 3.086 * 10**16)**3
        * 8.62 * 10**(-27) * 0.307 * 360) / (1.98 * 10**30))
Mvirm = ((4 / 3 * np.pi * (Rvirm * 10**6 * 3.086 * 10**16)**3
        * 8.62 * 10**(-27) * 0.307 * 360) / (1.98 * 10**30))

print('Mvir_Coma: '+'{:.2e}'.format(Mvir))


a = 0.00989464 # in km/s
b = 0.330353
sig1d = a * Mvir**b # from Kyle thesis

# uncertainty in sig1d
sig1dp = a * Mvirp**b
sig1dm = a * Mvirm**b

print('sig1d_Coma: '+str(sig1d))

sig3d = np.sqrt(3) * sig1d

# uncertainty in sig3d
sig3dp = np.sqrt(3) * sig1dp
sig3dm = np.sqrt(3) * sig1dm

print('sig3d_Coma: '+str(sig3d))

data = {'Rvir':[Rvir], 'Rvir+':[Rvirp], 'Rvir-':[Rvirm], 
        'Mvir':[Mvir], 'Mvir+':[Mvirp], 'Mvir-':[Mvirm], 
        'sig1d':[sig1d], 'sig1d+':[sig1dp], 'sig1d-':[sig1dm], 
        'sig3d':[sig3d], 'sig3d+':[sig3dp], 'sig3d-':[sig3dm]}

df = pd.DataFrame(data, columns = ['Rvir', 'Rvir+', 'Rvir-', 
                                   'Mvir', 'Mvir+', 'Mvir-', 
                                   'sig1d', 'sig1d+', 'sig1d-', 
                                   'sig3d', 'sig3d+', 'sig3d-'])

df.to_csv('Coma_props.csv', index = False)


df1 = pd.DataFrame()
mass_cols = ['Mvir', 'Mvir+', 'Mvir-']
for col in df.columns.tolist():
    if col in mass_cols:
        df1[col] = format_exp_arr(np.array(df[col]))
    else:
        df1[col] = format_float_arr(np.array(df[col]))

fig, ax = plt.subplots(figsize=(3,1))
#ax.axis('tight')
ax.axis('off')
tab = ax.table(cellText=df1.values,colLabels=df1.columns,loc='center')
tab.auto_set_font_size(False)
tab.set_fontsize(6)
tab.auto_set_column_width(col=list(range(len(df.columns))))
pp = PdfPages("Coma_props.pdf")
pp.savefig(fig, bbox_inches='tight')
pp.close()