# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:39:30 2020

@author: amit
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def formatting(arr,order):
    res = np.array([])
    if order == 2:
        for item in arr:
            res = np.append(res,'{:.2f}'.format(item))
    elif order == 4:
        for item in arr:
            res = np.append(res,'{:.4f}'.format(item))
    elif order == 5:
        for item in arr:
            res = np.append(res,'{:.5f}'.format(item))
    return res

gmp_names = ['3254','3269', '3291', '3329', '3352', '3367', '3414', '3484',
            '3534', '3565', '3639', '3664']

log_Ms = np.loadtxt('logMsp_coma.m')
log_Mh = np.loadtxt('logMhp_coma1.m')

z = np.loadtxt('redshift.z')

coord_coma = pd.read_csv('Coma_coords.csv')

R = np.loadtxt('R.vir')
V = np.loadtxt('V.sig3d')

int_frac = np.array(pd.read_csv('int_frac.csv')['int_frac'])*100

df = pd.DataFrame()
df['GMP'] = gmp_names
df['RA'] = coord_coma['RA']
df['DEC'] = coord_coma['DEC']
df['z'] = formatting(z, 5)
df['log_Ms'] = formatting(log_Ms, 2)
df['log_Mh'] = formatting(log_Mh, 2)
df['R/R_vir'] = formatting(R, 4)
df['V/sig3d'] = formatting(V, 4)
df['int_frac'] = formatting(int_frac, 2)

fig, ax = plt.subplots(figsize=(4,3))
#ax.axis('tight')
ax.axis('off')
tab = ax.table(cellText=df.values,colLabels=df.columns,loc='center')
tab.auto_set_font_size(False)
tab.set_fontsize(6)
tab.auto_set_column_width(col=list(range(len(df.columns))))
pp = PdfPages("coma_prop_tab.pdf")
pp.savefig(fig, bbox_inches='tight')
pp.close()