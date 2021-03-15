# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 11:57:08 2020

@author: amit
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable, viridis as cmap
from matplotlib.colors import Normalize
from matplotlib.ticker import AutoMinorLocator

sat_names = ['3254','3269', '3291', '3329', '3352', '3367', '3414', '3484',
             '3534', '3565', '3639', '3664']

log_Ms = np.array(pd.read_csv('smhm_behroozi2010.csv')['logMs'])

log_Mh = np.array(pd.read_csv('smhm_behroozi2010.csv')['logMh'])

r = np.array(pd.read_csv('Coma_PPS.csv')['R'])

v = np.array(pd.read_csv('Coma_PPS.csv')['V'])


norm = Normalize(vmin=11, vmax=13)
smap = ScalarMappable(cmap=cmap, norm=norm)
smap.set_array([])

fig, ax = plt.subplots(figsize=(10,7))
for name, log_Mi, ri, vi in zip(sat_names, log_Mh, r, v):
    c = cmap(norm(log_Mi))
    ax.plot(ri, vi, 's', markersize=12, color=c)
    ax.annotate(name, (ri, vi), xytext=(-26, 9), textcoords='offset points',
                color='k', fontsize=14)
ax.set_xlim(-0.005,0.055)
ax.set_ylim(-0.1,1.3)
xticks = ax.xaxis.get_major_ticks()
xticks[0].label1.set_visible(False)
xticks[-1].label1.set_visible(False)
ax.set_ylabel(r'$V/\sigma_{3D}$',fontsize=18)
ax.set_xlabel(r'$R/r_{vir}$',fontsize=18)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both',which='major',direction='in', 
               bottom = True, top = True,left = True, right = True, length=10,
               labelsize=18)
ax.tick_params(axis='both',which='minor',direction='in', 
               bottom = True, top = True,left = True, right = True, length=5,
               labelsize=18)
#cbar = fig.colorbar(smap, ticks=[9., 9.5, 10.0, 10.5, 11.0])
cbar = fig.colorbar(smap, ticks=[11., 11.5, 12.0, 12.5, 13.0])
cbar.set_label(r'$\log_(M_\mathrm{sat}/\mathrm{M}_\odot)$',fontsize=18)
cbar.ax.tick_params(axis='y', direction='in', length=10, labelsize=18)
plt.savefig('PPS.pdf',dpi=500)