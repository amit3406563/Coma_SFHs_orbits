# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 10:44:33 2020

@author: amit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as sf
from matplotlib.ticker import AutoMinorLocator

def sav_gol(x,win,poly):
    return sf(x, window_length=win, polyorder=poly, mode='interp')


sfr_miles = pd.read_csv('sfr_ssfr_tables/miles/corr_sfr.csv')
sfr_bc03 = pd.read_csv('sfr_ssfr_tables/bc03/corr_sfr.csv')
sfr_phr = pd.read_csv('sfr_ssfr_tables/phr/corr_sfr.csv')

fig, ax = plt.subplots(figsize=(7,5))
name = '3254'
ax.plot(np.array(sfr_miles['Age_Gyr']),
        sav_gol(np.array(sfr_miles[name]),9,3),'r-',label='MILES')
ax.plot(np.array(sfr_bc03['Age_Gyr']),
        sav_gol(np.array(sfr_bc03[name]),9,3),'g:',label='BC03')
ax.plot(np.array(sfr_phr['Age_Gyr']),
        sav_gol(np.array(sfr_phr[name]),9,3),'c:',label='PHR')
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

ax.grid(False)
ax.legend(frameon=False, framealpha=1.0,fontsize=14)
ax.tick_params(axis='both',which='both',direction='in', bottom = True, 
                top = True, left = True, right = True,labelsize=18)
ax.set_xlabel('Lookback time [Gyr]',fontsize=18)
ax.set_ylabel(r'SFR[${\rm M}_\odot$/yr]',fontsize=18)
fig.tight_layout()
plt.savefig('sfr_ssfr_tables/ssp_compare.pdf',dpi=500)