# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 02:50:42 2020

@author: amit
"""


import warnings
warnings.filterwarnings("ignore")

import numpy as np
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
def read_plot1(ssp):
    ## reading tables for plots
    res1 = pd.read_csv('./cumMs_at_inf_peri/'+ssp+'/Rvir_Ms/Rvir_Ms_'+ssp+'_cum_Ms_at_exp_orb_time_table.csv')
    res1.set_index('GMP',inplace=True)
    res1.astype(float)
    return res1

def read_plot2(ssp):
    ## reading tables for plots
    res2 = pd.read_csv('./f_ms_inf_peri/'+ssp+'/Rvir_Ms/'+ssp+'_f_ms_inf_peri.csv')
    res2.set_index('GMP',inplace=True)
    res2.astype(float)
    return res2
    
## read tables
res1 = read_plot1('miles')
res1b = read_plot1('bc03')
res1p = read_plot1('phr')
res2 = read_plot2('miles')
res2b = read_plot2('bc03')
res2p = read_plot2('phr')

## plotting
# colormap setup
norm = Normalize(vmin=9, vmax=11)
smap = ScalarMappable(cmap=cmap, norm=norm)
smap.set_array([])
# plot
fig, ((ax1, ax2,  ax3),(ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(18,11))
gs1 = gridspec.GridSpec(2, 3)
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
    # plot2 - MILES
    ax4.errorbar(res2.loc[name]['log_Ms'], res2.loc[name]['m_%Ms_ip'], 
                 yerr=[[res2.loc[name]['e-']],[res2.loc[name]['e+']]],
                 elinewidth=1, capsize=5, ecolor='k', marker='D', mec='k', 
                 mfc='k', markersize=8)
    # plot2 - BC03
    ax5.errorbar(res2b.loc[name]['log_Ms'], res2b.loc[name]['m_%Ms_ip'], 
                 yerr=[[res2b.loc[name]['e-']],[res2b.loc[name]['e+']]],
                 elinewidth=1, capsize=5, ecolor='k', marker='D', mec='k', 
                 mfc='k', markersize=8)
    # plot2- PHR
    ax6.errorbar(res2p.loc[name]['log_Ms'], res2p.loc[name]['m_%Ms_ip'], 
                 yerr=[[res2p.loc[name]['e-']],[res2p.loc[name]['e+']]],
                 elinewidth=1, capsize=5, ecolor='k', marker='D', mec='k', 
                 mfc='k', markersize=8)
# plot2- lin-reg -MILES
lin_reg_x = np.linspace(9.,11.,21)
lin_reg_y, b, m = lin_reg(np.array(res2['log_Ms']),np.array(res2['m_%Ms_ip']),
                          lin_reg_x)
b = '{:.2f}'.format(b)
m = '{:.2f}'.format(m[0])
ax4.plot(lin_reg_x,lin_reg_y,c='m',linestyle='-',linewidth=4)
# plot2- settings - MILES
ax4.set_ylim(-0.5,50.5)
ax4.set_yticks(ax4.get_yticks()[1:-1]) # Remove first and last ticks
ax4.set_xlim(8.8,11.2)
ax4.set_xticks(ax4.get_xticks()[1:-1]) # Remove first and last ticks
#ax4.set_xlabel(r'$\log_{\rm 10}M_\star/{\rm M}_\odot$',
#                  fontsize=18)
ax4.set_ylabel(r'Fractional $M_\star$ increase from $t_\mathrm{inf}$ to $t_\mathrm{peri}$',
                      fontsize=18)
ax4.xaxis.set_minor_locator(AutoMinorLocator())
ax4.yaxis.set_minor_locator(AutoMinorLocator())
ax4.tick_params(axis='both',which='major',direction='in', bottom = True, 
                            top = True,left = True, right = True, length=10, 
                            labelsize=18)
ax4.tick_params(axis='both',which='minor',direction='in', bottom = True, 
                   top = True, left = True, right = True,  length=5,
                   labelsize=18)
dia = mlines.Line2D([], [], color='k', marker='D', 
                    linestyle='None', markersize=8, 
                    label=r'$\frac{\Delta M_{\star,\mathrm{inf-peri}}}{M_{\star,\mathrm{final}}}\,\%\,:\,y$')
line = mlines.Line2D([], [], color='m', marker='None', linestyle='-',
                     markersize=8,linewidth=4, 
                     label=r'$y=$'+m+r'$\,\log(M_\star/\mathrm{M}_\odot)+\,$'+b)
ax4.legend(handles=[dia,line],fontsize=13, loc=2, bbox_to_anchor=(0.02,0.98), 
              bbox_transform=ax4.transAxes) 
ax4.grid(False)
ax4.set_facecolor('w')
# plot2- lin-reg - BC03
lin_reg_x = np.linspace(9.,11.,21)
lin_reg_y, b, m = lin_reg(np.array(res2b['log_Ms']),np.array(res2b['m_%Ms_ip']),
                          lin_reg_x)
b = '{:.2f}'.format(b)
m = '{:.2f}'.format(m[0])
ax5.plot(lin_reg_x,lin_reg_y,c='m',linestyle='-',linewidth=4)
# plot2- settings - BC03
ax5.set_ylim(-0.5,50.5)
ax5.set_yticks(ax5.get_yticks()[1:-1]) # Remove first and last ticks
ax5.set_yticklabels([])
ax5.set_xlim(8.8,11.2)
ax5.set_xticks(ax5.get_xticks()[1:-1]) # Remove first and last ticks
ax5.set_xlabel(r'$\log(M_\star/\mathrm{M}_\odot)$',
                      fontsize=18)
#ax5.set_ylabel(r'${\Delta M_\star}_{{\rm inf}-{\rm peri}}/M_\star$ %',
#                  fontsize=18)
ax5.xaxis.set_minor_locator(AutoMinorLocator())
ax5.yaxis.set_minor_locator(AutoMinorLocator())
ax5.tick_params(axis='both',which='major',direction='in', bottom = True, 
                            top = True,left = True, right = True, length=10, 
                            labelsize=18)
ax5.tick_params(axis='both',which='minor',direction='in', bottom = True, 
                   top = True, left = True, right = True,  length=5,
                   labelsize=18)
dia = mlines.Line2D([], [], color='k', marker='D', 
                    linestyle='None', markersize=8, 
                    label=r'$\frac{\Delta M_{\star,\mathrm{inf-peri}}}{M_{\star,\mathrm{final}}}\,\%\,:\,y$')
line = mlines.Line2D([], [], color='m', marker='None', linestyle='-',
                     markersize=8,linewidth=4, 
                     label=r'$y=$'+m+r'$\,\log(M_\star/\mathrm{M}_\odot)+\,$'+b)
ax5.legend(handles=[dia,line],fontsize=13, loc=2, bbox_to_anchor=(0.02,0.98), 
              bbox_transform=ax5.transAxes) 
ax5.grid(False)
ax5.set_facecolor('w')
# plot2- lin-reg - PHR
lin_reg_x = np.linspace(9.,11.,21)
lin_reg_y, b, m = lin_reg(np.array(res2p['log_Ms']),np.array(res2p['m_%Ms_ip']),
                          lin_reg_x)
b = '{:.2f}'.format(b)
m = '{:.2f}'.format(m[0])
ax6.plot(lin_reg_x,lin_reg_y,c='m',linestyle='-',linewidth=4)
# plot2- settings - PHR
ax6.set_ylim(-0.5,50.5)
ax6.set_yticks(ax6.get_yticks()[1:-1]) # Remove first and last ticks
ax6.set_yticklabels([])
ax6.set_xlim(8.8,11.2)
ax6.set_xticks(ax6.get_xticks()[1:-1]) # Remove first and last ticks
#ax6.set_xlabel(r'$\log_{\rm 10}M_\star/{\rm M}_\odot$',
#                  fontsize=18)
#ax6.set_ylabel(r'${\Delta M_\star}_{{\rm inf}-{\rm peri}}/M_\star$ %',
#                  fontsize=18)
ax6.xaxis.set_minor_locator(AutoMinorLocator())
ax6.yaxis.set_minor_locator(AutoMinorLocator())
ax6.tick_params(axis='both',which='major',direction='in', bottom = True, 
                            top = True,left = True, right = True, length=10, 
                            labelsize=18)
ax6.tick_params(axis='both',which='minor',direction='in', bottom = True, 
                   top = True, left = True, right = True,  length=5,
                   labelsize=18)
dia = mlines.Line2D([], [], color='k', marker='D', 
                    linestyle='None', markersize=8, 
                    label=r'$\frac{\Delta M_{\star,\mathrm{inf-peri}}}{M_{\star,\mathrm{final}}}\,\%\,:\,y$')
line = mlines.Line2D([], [], color='m', marker='None', linestyle='-',
                     markersize=8,linewidth=4, 
                     label=r'$y=$'+m+r'$\,\log(M_\star/\mathrm{M}_\odot)+\,$'+b)
ax6.legend(handles=[dia,line],fontsize=13, loc=2, bbox_to_anchor=(0.02,0.98), 
              bbox_transform=ax6.transAxes) 
ax6.grid(False)
ax6.set_facecolor('w')
# plot settings - MILES
ax1.set_ylim(0.20,1.05)
ax1.set_yticks(ax1.get_yticks()[1:-1]) # Remove first and last ticks
ax1.set_xlim(-1,11.0)
ax1.set_ylabel(r'Fraction of cumulative $M_\star$ formed',fontsize=18)
#ax1.set_xlabel('Lookback time [Gyr]',fontsize=18)
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.tick_params(axis='both',which='major',direction='in', bottom = True, 
                   top = True, left = True, right = True,  length=10,
                   labelsize=18)
ax1.tick_params(axis='both',which='minor',direction='in', bottom = True, 
                   top = True, left = True, right = True,  length=5,
                   labelsize=18)
circ = mlines.Line2D([], [], color='grey', marker='o', linestyle='None',
                     markersize=8, label='inf') 
star = mlines.Line2D([], [], color='grey', marker='*', linestyle='None',
                     markersize=10, label='peri')
ax1.legend(handles=[circ, star],frameon=True, framealpha=0.7, 
              edgecolor='k', fontsize=15, loc=3, bbox_to_anchor=(0.02,0.02), 
              bbox_transform=ax1.transAxes)
ax1.grid(False)
ax1.set_facecolor('w')
# plot settings - BC03
ax2.set_ylim(0.20,1.05)
ax2.set_yticks(ax2.get_yticks()[1:-1]) # Remove first and last ticks
#ax2.set_yticklabels([])
ax2.set_yticklabels([])
ax2.set_xlim(-1,11.0)
#ax2.set_ylabel(r'Fraction of cumulative $M_\star$ formed',fontsize=18)
ax2.set_xlabel('Lookback time [Gyr]',fontsize=18)
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax2.tick_params(axis='both',which='major',direction='in', bottom = True, 
                            top = True,left = True, right = True, length=10, 
                            labelsize=18)
ax2.tick_params(axis='both',which='minor',direction='in', bottom = True, 
                   top = True, left = True, right = True,  length=5,
                   labelsize=18)
circ = mlines.Line2D([], [], color='grey', marker='o', linestyle='None',
                     markersize=8, label='inf') 
star = mlines.Line2D([], [], color='grey', marker='*', linestyle='None',
                     markersize=10, label='peri')
ax2.legend(handles=[circ, star],frameon=True, framealpha=0.7, 
              edgecolor='k', fontsize=15, loc=3, bbox_to_anchor=(0.02,0.02), 
              bbox_transform=ax2.transAxes) 
ax2.grid(False)
ax2.set_facecolor('w')
# plot settings - PHR
ax3.set_ylim(0.20,1.05)
#ax3.set_xticklabels([0.,2.5,5.0,7.5,10.])
ax3.set_yticks(ax3.get_yticks()[1:-1]) # Remove first and last ticks
#ax3.set_yticklabels([])
ax3.set_yticklabels([])
ax3.set_xlim(-1,11.0)
#ax3.set_ylabel(r'Fraction of cumulative $M_\star$ formed',fontsize=18)
#ax3.set_xlabel('Lookback time [Gyr]',fontsize=18)
ax3.xaxis.set_minor_locator(AutoMinorLocator())
ax3.yaxis.set_minor_locator(AutoMinorLocator())
ax3.tick_params(axis='both',which='major',direction='in', bottom = True, 
                            top = True,left = True, right = True, length=10, 
                            labelsize=18)
ax3.tick_params(axis='both',which='minor',direction='in', bottom = True, 
                   top = True, left = True, right = True,  length=5,
                   labelsize=18)
circ = mlines.Line2D([], [], color='grey', marker='o', linestyle='None',
                     markersize=8, label='inf') 
star = mlines.Line2D([], [], color='grey', marker='*', linestyle='None',
                     markersize=10, label='peri')
ax3.legend(handles=[circ, star],frameon=True, framealpha=0.7, 
              edgecolor='k', fontsize=15, loc=3, bbox_to_anchor=(0.02,0.02), 
              bbox_transform=ax3.transAxes) 
cbar_divider3 = make_axes_locatable(ax3)
cbar_ax3 = cbar_divider3.append_axes('right', size='5%', pad=0.05)
cbar3 = fig.colorbar(smap, ticks=[9., 9.5, 10.0, 10.5, 11.0],cax=cbar_ax3)
cbar3.set_label(r'$\log(M_\star/\mathrm{M}_\odot)$',fontsize=18)
cbar_ax3.tick_params(axis='y', direction='in',labelsize=18)
ax3.grid(False)
ax3.set_facecolor('w')
# titles
ax1.set_title('MILES', fontsize=18, color = 'k')
ax2.set_title('BC03', fontsize=18, color = 'k')
ax3.set_title('PHR', fontsize=18, color = 'k')
# fig setting
fig.tight_layout()
# saving fig
out_dir = './final_plots/'
plt.savefig(out_dir+'sys_error_ssp_plot.pdf',dpi=500)