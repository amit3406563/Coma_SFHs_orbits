# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 11:57:08 2020

@author: amit
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable, viridis as cmap
from matplotlib.colors import Normalize
from matplotlib.ticker import AutoMinorLocator

sat_names = ['3254','3269', '3291', '3329', '3352', '3367', '3414', '3484',
             '3534', '3565', '3639', '3664']

log_Ms = np.loadtxt('logMs_coma.m')

log_Mh = np.loadtxt('logMh_coma.m')

r = np.loadtxt('R.vir')

v = np.loadtxt('V.sig3d')


norm = Normalize(vmin=11, vmax=13)
smap = ScalarMappable(cmap=cmap, norm=norm)
smap.set_array([])

fig, ax = plt.subplots(figsize=(10,7))
for name, log_Mi, ri, vi in zip(sat_names, log_Mh, r, v):
    c = cmap(norm(log_Mi))
    ax.plot(ri, vi, '*', markersize=12, color=c)
    ax.annotate(name, (ri, vi), xytext=(-30, 5), textcoords='offset points',
                color='b', fontsize=16)
ax.set_ylabel(r'$V/\sigma_{3D}$',fontsize=18)
ax.set_xlabel(r'$R/r_{vir}$',fontsize=18)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both',which='both',direction='in', 
               bottom = True, top = True,left = True, right = True,
               labelsize=18)
#cbar = fig.colorbar(smap, ticks=[9., 9.5, 10.0, 10.5, 11.0])
cbar = fig.colorbar(smap, ticks=[11., 11.5, 12.0, 12.5, 13.0])
cbar.set_label(r'$\log_{10}(M_{sat}/{\rm M}_\odot)$',fontsize=18)
cbar.ax.tick_params(axis='y', direction='in',labelsize=18)
plt.savefig('PPS.pdf',dpi=500)