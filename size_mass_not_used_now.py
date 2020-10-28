# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 05:03:59 2020

@author: amit
"""


import numpy as np
from matplotlib.cm import ScalarMappable, viridis as cmap
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

coma_dist = 99 ## in Mpc

sat_names = ['3254','3269', '3291', '3329', '3352', '3367', '3414', '3484',
             '3534', '3565', '3639', '3664']

Re_log_ang = [0.54,0.40,1.08,1.85,0.48,0.87,0.92,0.49,0.64,0.60,0.49,0.89] ## in arcsec
# Trager 2008
Re_ang = np.power(10,Re_log_ang)
Re_trager = np.multiply(np.multiply(Re_ang,4.8481368110954*10**(-6)),coma_dist*10**3)


gal_names = ['3254','3269','3291','3329','3367','3414']
Re_pix = [116.39, 77.1, 183.34, 144.15, 184.69, 150.93] ## Hoyos 2011
flat = [0.388, 0.548, 0.339, 0.057, 0.222, 0.311] # Hoyos 2011

#log_Ms_df = pd.read_csv('log_Ms_coma.csv') ## Trager 2008
#log_Ms_trager = log_Ms_df['log_Ms']
log_Ms_trager = np.loadtxt('logMs_coma.m')
log_Ms_hoyos = np.array([9.92, 9.98, 9.92, 11.43, 10.65, 10.73])

Re_hoyos = np.multiply(np.multiply(Re_pix, 0.05 * 4.8481368110954*10**(-6)),coma_dist*10**3)
## Re in kpc

## From Shen 2003 : Size - Stellar mass relation
M = np.linspace(10**9,10**12,100)
M_sol = 1.98 * 10**30

## from Shen et al. 2003
# early-type
R_etype = 2.88*10**(-6) * M**0.56
#late type
#R_ltype = 0.10 * M**0.14 * (1 + (M * M_sol) / (3.98 * 10**10))**(0.39-0.14)


## scatter
sig1 = 0.47
sig2 = 0.34
M0 = 3.98 * 10**10

sig = np.array([])
for m in log_Ms_hoyos:
    sig_temp = sig2 + ((sig1 - sig2) / (1 + 10**m / M0))
    sig = np.append(sig,sig_temp)

sigt = np.array([])
for m in log_Ms_trager:
    sig_temp = sig2 + ((sig1 - sig2) / (1 + 10**m / M0))
    sigt = np.append(sigt,sig_temp)
## plotting

norm = Normalize(vmin=0, vmax=1.0)
smap = ScalarMappable(cmap=cmap, norm=norm)
smap.set_array([])

fig, ax = plt.subplots(figsize=(10,7))
ax.plot(np.log10(M),np.log10(R_etype),color='m')

for f, s, log_Msi, Rei in zip(flat,sig,log_Ms_hoyos,Re_hoyos):
    c = cmap(norm(f))
    yerri = [[s/2],[s/2]]
    ax.errorbar(log_Msi,np.log10(Rei),yerr=yerri,elinewidth=1, 
                capsize=5, ecolor=c, marker='o', mec=c, mfc=c,markersize=8)

for i, txt in enumerate(gal_names):
    ax.annotate(txt, (log_Ms_hoyos[i], np.log10(Re_hoyos[i])),
                xytext=(-30, 3), textcoords='offset points',color='b',
                fontsize=14)


yerrt4 = [[sigt[4]/2],[sigt[4]/2]]

ax.errorbar(log_Ms_trager[4],np.log10(Re_trager[4]),yerr=yerrt4,elinewidth=1, 
            capsize=5, ecolor='r', marker='*', mec='r', mfc='r',markersize=8)

for s, log_Msi, Rei in zip(sigt[7:],log_Ms_trager[7:],Re_trager[7:]):
    yerri = [[s/2],[s/2]]
    ax.errorbar(log_Msi,np.log10(Rei),yerr=yerri,elinewidth=1,
            capsize=5, ecolor='r', marker='*', mec='r', mfc='r',markersize=8)

ax.annotate('3664', (log_Ms_trager[11], np.log10(Re_trager[11])),
            xytext=(5, 3), textcoords='offset points',color = 'r',fontsize=14)
for i, txt in enumerate(sat_names[7:10]):
    ax.annotate(txt, (log_Ms_trager[i+7], np.log10(Re_trager[i+7])),
                xytext=(5, 3), textcoords='offset points',color = 'r',
                fontsize=14)

ax.set_ylabel(r'$log_{10}R_e$ [kpc]',fontsize=18)
ax.set_xlabel(r'$\log_{10}(M_\star/{\rm M}_\odot)$',fontsize=18)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both',which='both',direction='in', 
               bottom = True, top = True,left = True, right = True,labelsize=18)
cbar = fig.colorbar(smap)
cbar.set_label('Flattening',fontsize=18)
cbar.ax.tick_params(axis='y',direction='in',labelsize=18)
circ = mlines.Line2D([], [], color='grey', marker='o', linestyle='None',
                          markersize=10, label='Coma galaxies: Hoyos (2011)')
star = mlines.Line2D([], [], color='grey', marker='*', linestyle='None',
                          markersize=10, label='Coma galaxies: Trager (2008)')
line = mlines.Line2D([], [], color='m', marker='None', linestyle='-',
                          label='E-type galaxies: Shen (2003)')
ax.legend(handles=[circ,star,line],loc=2,frameon=False, framealpha=1.0,fontsize=16)
ax.grid(False)


axins2 = zoomed_inset_axes(ax, 6, loc = 4,bbox_to_anchor=(0.8,0.05), 
                           bbox_transform=ax.transAxes)
axins2.plot(np.log10(M),np.log10(R_etype),color='m')

for s, log_Msi, Rei in zip(sigt,log_Ms_trager,Re_trager):
    yerri = [[s/2],[s/2]]
    axins2.errorbar(log_Msi,np.log10(Rei),yerr=yerri,elinewidth=1,
            capsize=5, ecolor='r', marker='*', mec='r', mfc='r',markersize=8)

for i, txt in enumerate(sat_names):
    axins2.annotate(txt, (log_Ms_trager[i], np.log10(Re_trager[i])),
                xytext=(5, 3), textcoords='offset points',color = 'r',
                fontsize=14)

for i, txt in enumerate(gal_names):
    axins2.annotate(txt, (log_Ms_hoyos[i], np.log10(Re_hoyos[i])),
                xytext=(8, 3), textcoords='offset points',color='b',
                fontsize=14)

x1, x2, y1, y2 = 10.56, 10.67, 0.13, 0.2 # specify the limits
axins2.set_xlim(x1, x2) # apply the x-limits
axins2.set_ylim(y1, y2) # apply the y-limits
axins2.xaxis.set_minor_locator(AutoMinorLocator())
axins2.yaxis.set_minor_locator(AutoMinorLocator())
axins2.tick_params(axis='both',which='both',direction='in', 
                   bottom = True, top = True, left = True, right = True,
                   labelsize=12)
mark_inset(ax, axins2, loc1=1, loc2=3, fc="none", ec="0.5")
axins2.grid(False)

plt.savefig('size_mass.pdf',dpi=500)
#plt.savefig('size-mass.png',dpi=200)