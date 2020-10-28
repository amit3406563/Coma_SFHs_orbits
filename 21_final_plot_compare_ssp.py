# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 02:50:42 2020

@author: amit
"""


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.linear_model import LinearRegression as lr
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.cm import ScalarMappable, viridis as cmap
from matplotlib.colors import Normalize
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

# computes linear regression
def lin_reg(x,y,xnew):
    x = x.reshape(-1,1)
    model = lr().fit(x,y)
    intercept = model.intercept_
    slope = model.coef_
    xnew = xnew.reshape(-1,1)
    ynew = model.predict(xnew)
    return ynew, intercept, slope

# reads tables required for plot for different ssp cases 
def read_plot_dfs(ssp):
    ## reading tables for plots
    res1 = pd.read_csv('./cumMs_at_inf_peri/'+ssp+'/Rvir_Ms/Rvir_Ms_'+ssp+'_cum_Ms_at_exp_orb_time_table.csv')
    res1.set_index('GMP',inplace=True)
    res1.astype(float)
    return res1
    
## read tables
res1 = read_plot_dfs('miles')
res1b = read_plot_dfs('bc03')
res1p = read_plot_dfs('phr')

## plotting
# colormap setup
norm = Normalize(vmin=9, vmax=11)
smap = ScalarMappable(cmap=cmap, norm=norm)
smap.set_array([])
# plot
fig, (ax1, ax2,  ax3) = plt.subplots(1,3, figsize=(15,5))
gs1 = gridspec.GridSpec(1, 3)
gs1.update(wspace=0.025, hspace=0.05)
for name in res1.index:
    ## plot-1
    # colors for errorbars based on stellar mass
    c = cmap(norm(res1.loc[name]['log_Ms']))
    # infall points - MILES
    ax1.errorbar(res1.loc[name]['tinf'],res1.loc[name]['%Ms_tinf']/100,
                 xerr=[[res1.loc[name]['tinf-']],[res1.loc[name]['tinf+']]],
                 yerr=[[res1.loc[name]['%Ms_tinf-']/100],
                       [res1.loc[name]['%Ms_tinf+']/100]],
                 elinewidth=1, capsize=5, ecolor=c, marker='o', mec=c, mfc=c,
                 markersize=8)
    # pericenter points - MILES
    ax1.errorbar(res1.loc[name]['tperi'],res1.loc[name]['%Ms_tperi']/100,
                 xerr=[[res1.loc[name]['tperi-']],[res1.loc[name]['tperi+']]],
                 yerr=[[res1.loc[name]['%Ms_tperi-']/100],
                       [res1.loc[name]['%Ms_tperi+']/100]],
                 elinewidth=1, capsize=5, ecolor=c, marker='*', mec=c, mfc=c,
                 markersize=8)
    # infall points - BC03
    ax2.errorbar(res1b.loc[name]['tinf'],res1b.loc[name]['%Ms_tinf']/100,
                 xerr=[[res1b.loc[name]['tinf-']],[res1b.loc[name]['tinf+']]],
                 yerr=[[res1b.loc[name]['%Ms_tinf-']/100],
                       [res1b.loc[name]['%Ms_tinf+']/100]],
                 elinewidth=1, capsize=5, ecolor=c, marker='o', mec=c, mfc=c,
                 markersize=8)
    # pericenter points - BC03
    ax2.errorbar(res1b.loc[name]['tperi'],res1b.loc[name]['%Ms_tperi']/100,
                 xerr=[[res1b.loc[name]['tperi-']],[res1b.loc[name]['tperi+']]],
                 yerr=[[res1b.loc[name]['%Ms_tperi-']/100],
                       [res1b.loc[name]['%Ms_tperi+']/100]],
                 elinewidth=1, capsize=5, ecolor=c, marker='*', mec=c, mfc=c,
                 markersize=8)
    # infall points -PHR
    ax3.errorbar(res1p.loc[name]['tinf'],res1p.loc[name]['%Ms_tinf']/100,
                 xerr=[[res1p.loc[name]['tinf-']],[res1p.loc[name]['tinf+']]],
                 yerr=[[res1p.loc[name]['%Ms_tinf-']/100],
                       [res1p.loc[name]['%Ms_tinf+']/100]],
                 elinewidth=1, capsize=5, ecolor=c, marker='o', mec=c, mfc=c,
                 markersize=8)
    # pericenter points - PHR
    ax3.errorbar(res1.loc[name]['tperi'],res1.loc[name]['%Ms_tperi']/100,
                 xerr=[[res1.loc[name]['tperi-']],[res1.loc[name]['tperi+']]],
                 yerr=[[res1.loc[name]['%Ms_tperi-']/100],
                       [res1.loc[name]['%Ms_tperi+']/100]],
                 elinewidth=1, capsize=5, ecolor=c, marker='*', mec=c, mfc=c,
                 markersize=8)
# plot settings - MILES
ax1.set_ylim(0.50,1.05)
ax1.set_yticks(ax1.get_yticks()[1:-1]) # Remove first and last ticks
ax1.set_xlim(-1,11.0)
ax1.set_ylabel(r'Fraction of cumulative $M_\star$ formed',fontsize=18)
#ax1.set_xlabel('Lookback time [Gyr]',fontsize=18)
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.tick_params(axis='both',which='both',direction='in', bottom = True, 
                top = True,left = True, right = True,labelsize=18)
circ = mlines.Line2D([], [], color='grey', marker='o', linestyle='None',
                     markersize=8, label='inf: MILES') 
star = mlines.Line2D([], [], color='grey', marker='*', linestyle='None',
                     markersize=10, label='peri: MILES')
ax1.legend(handles=[circ, star],frameon=False, framealpha=1.0,loc=3,fontsize=12) 
# cbar_divider = make_axes_locatable(ax1)
# cbar_ax = cbar_divider.append_axes('right', size='5%', pad=0.05)
# cbar = fig.colorbar(smap, ticks=[9., 9.5, 10.0, 10.5, 11.0],cax=cbar_ax)
# cbar.set_label(r'$\log_{10}(M_\star/{\rm M}_\odot)$',fontsize=18)
# cbar_ax.tick_params(axis='y', direction='in',labelsize=18)
ax1.grid(False)
ax1.set_facecolor('w')
# plot settings - BC03
ax2.set_ylim(0.50,1.05)
ax2.set_yticks(ax2.get_yticks()[1:-1]) # Remove first and last ticks
ax2.set_yticklabels([])
ax2.set_xlim(-1,11.0)
#ax2.set_ylabel(r'Fraction of cumulative $M_\star$ formed',fontsize=18)
ax2.set_xlabel('Lookback time [Gyr]',fontsize=18)
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax2.tick_params(axis='both',which='both',direction='in', bottom = True, 
                top = True,left = True, right = True,labelsize=18)
circ = mlines.Line2D([], [], color='grey', marker='o', linestyle='None',
                     markersize=8, label='inf: BC03') 
star = mlines.Line2D([], [], color='grey', marker='*', linestyle='None',
                     markersize=10, label='peri: BC03')
ax2.legend(handles=[circ, star],frameon=False, framealpha=1.0,loc=3,fontsize=12) 
# cbar_divider2 = make_axes_locatable(ax2)
# cbar_ax2 = cbar_divider2.append_axes('right', size='5%', pad=0.05)
# cbar2 = fig.colorbar(smap, ticks=[9., 9.5, 10.0, 10.5, 11.0],cax=cbar_ax2)
# cbar2.set_label(r'$\log_{10}(M_\star/{\rm M}_\odot)$',fontsize=18)
# cbar_ax2.tick_params(axis='y', direction='in',labelsize=18)
ax2.grid(False)
ax2.set_facecolor('w')
# plot settings - PHR
ax3.set_ylim(0.50,1.05)
#ax3.set_xticklabels([0.,2.5,5.0,7.5,10.])
ax3.set_yticks(ax3.get_yticks()[1:-1]) # Remove first and last ticks
ax3.set_yticklabels([])
ax3.set_xlim(-1,11.0)
#ax3.set_ylabel(r'Fraction of cumulative $M_\star$ formed',fontsize=18)
#ax3.set_xlabel('Lookback time [Gyr]',fontsize=18)
ax3.xaxis.set_minor_locator(AutoMinorLocator())
ax3.yaxis.set_minor_locator(AutoMinorLocator())
ax3.tick_params(axis='both',which='both',direction='in', bottom = True, 
                top = True,left = True, right = True,labelsize=18)
circ = mlines.Line2D([], [], color='grey', marker='o', linestyle='None',
                     markersize=8, label='inf: PHR') 
star = mlines.Line2D([], [], color='grey', marker='*', linestyle='None',
                     markersize=10, label='peri: PHR')
ax3.legend(handles=[circ, star],frameon=False, framealpha=1.0,loc=3,fontsize=12) 
cbar_divider3 = make_axes_locatable(ax3)
cbar_ax3 = cbar_divider3.append_axes('right', size='5%', pad=0.05)
cbar3 = fig.colorbar(smap, ticks=[9., 9.5, 10.0, 10.5, 11.0],cax=cbar_ax3)
cbar3.set_label(r'$\log_{10}(M_\star/{\rm M}_\odot)$',fontsize=18)
cbar_ax3.tick_params(axis='y', direction='in',labelsize=18)
ax3.grid(False)
ax3.set_facecolor('w')
# fig setting
fig.tight_layout()
# saving fig
out_dir = './final_plots/'
plt.savefig(out_dir+'final_plot_comparison_ssp.pdf',dpi=500)