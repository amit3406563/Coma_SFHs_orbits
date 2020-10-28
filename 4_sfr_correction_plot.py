# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:54:30 2020

@author: Amit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as sf
import matplotlib.lines as mlines
from matplotlib.ticker import AutoMinorLocator


incorr_sfr = pd.read_csv('steckmap_sfr_tables/miles/steckmap_sfr.csv')
corr_sfr = pd.read_csv('sfr_ssfr_tables/miles/corr_sfr.csv')

gmp_names = ['3254','3269', '3291', '3352', '3367', '3414', '3484',
             '3534', '3565', '3639', '3664']

def sav_gol(x,win,poly):
    return sf(x, window_length=win, polyorder=poly, mode='interp')

fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(7,5))

for name in gmp_names:
    #ax1.scatter(np.log10(np.array(incorr_sfr['Age_Gyr'])),
             #np.log10(np.array(incorr_sfr[name])),s=10)
    ax.plot(np.array(incorr_sfr['Age_Gyr']),
             sav_gol(np.array(incorr_sfr[name]),9,3),':',label=name)
    #ax2.scatter(np.log10(np.array(corr_sfr['Age_Gyr'])),
             #np.log10(np.array(corr_sfr[name])),s=10)
    
    
line1 = mlines.Line2D([], [], color='grey', marker='None', linestyle=':',
                          label='Filtered_SFR_STECKMAP')
circ1 = mlines.Line2D([], [], color='grey', marker='o', linestyle='None',
                          markersize=10, label='SFR_STECKMAP')
line2 = mlines.Line2D([], [], color='grey', marker='None', linestyle=':',
                          label='Filtered_SFR_Corrected')
circ2 = mlines.Line2D([], [], color='grey', marker='o', linestyle='None',
                          markersize=10, label='SFR_Corrected')
ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

#ax1.set_xlim(9,13.7)
#ax1.set_ylim(0,0.000009)
ax.grid(False)
#ax1.legend(handles=[circ1,line1],loc=2,frameon=False, framealpha=1.0)
ax.legend(frameon=False, framealpha=1.0,fontsize=14)
ax.tick_params(axis='both',which='both',direction='in', bottom = True, 
                top = True, left = True, right = True,labelsize=18)
ax.set_xlabel('Lookback time [Gyr]',fontsize=18)
ax.set_ylabel(r'SFR[${\rm M}_\odot$/yr]',fontsize=18)
    

fig.tight_layout()
#plt.savefig('sfr_incor.png',dpi=200)
plt.savefig('sfr_ssfr_tables/miles/sfr_incor.pdf',dpi=500)


fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(7,5))

for name in gmp_names:
    ax.plot(np.array(corr_sfr['Age_Gyr']),
             sav_gol(np.array(corr_sfr[name]),9,3),':',label=name)

line1 = mlines.Line2D([], [], color='grey', marker='None', linestyle=':',
                          label='Filtered_SFR_STECKMAP')
circ1 = mlines.Line2D([], [], color='grey', marker='o', linestyle='None',
                          markersize=10, label='SFR_STECKMAP')
line2 = mlines.Line2D([], [], color='grey', marker='None', linestyle=':',
                          label='Filtered_SFR_Corrected')
circ2 = mlines.Line2D([], [], color='grey', marker='o', linestyle='None',
                          markersize=10, label='SFR_Corrected')
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

ax.grid(False)
#ax2.legend(handles=[circ2,line2],loc=2,frameon=False, framealpha=1.0)
ax.legend(frameon=False, framealpha=1.0,fontsize=14)
ax.tick_params(axis='both',which='both',direction='in', bottom = True, 
                top = True, left = True, right = True,labelsize=18)
ax.set_xlabel('Lookback time [Gyr]',fontsize=18)
ax.set_ylabel(r'SFR[${\rm M}_\odot$/yr]',fontsize=18)
fig.tight_layout()
#plt.savefig('sfr_corr.png',dpi=200)
plt.savefig('sfr_ssfr_tables/miles/sfr_corr.pdf',dpi=500)