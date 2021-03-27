# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:39:10 2020

@author: amit
"""

import warnings
warnings.filterwarnings("ignore")

import os
import shutil
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as lr
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.cm import ScalarMappable, viridis as cmap
from matplotlib.colors import Normalize
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

# output directory
def gen_out_dir(ssp):
    out_dir = './final_plots/'+ssp+'/'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    return out_dir

# computes linear regression
def lin_reg(x,y,xnew):
    x = x.reshape(-1,1)
    model = lr().fit(x,y)
    intercept = model.intercept_
    slope = model.coef_
    xnew = xnew.reshape(-1,1)
    ynew = model.predict(xnew)
    return ynew, intercept, slope

# computations and plotting
def plot_all(ssp):
    ## reading tables for plots
    res1 = pd.read_csv('./cumMs_at_inf_peri/'+ssp+'/Rvir_Ms/Rvir_Ms_'+ssp+'_cum_Ms_at_exp_orb_time_table.csv')
    res1.set_index('GMP',inplace=True)
    res1.astype(float)
    res2 = pd.read_csv('./f_ms_inf_peri/'+ssp+'/Rvir_Ms/'+ssp+'_f_ms_inf_peri.csv')
    res2.set_index('GMP',inplace=True)
    res2.astype(float)
    
    ## plotting
    # colormap setup
    norm = Normalize(vmin=9, vmax=11)
    smap = ScalarMappable(cmap=cmap, norm=norm)
    smap.set_array([])
    # plot
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,6))
    for name in res1.index:
        ## plot-1
        # colors for errorbars based on stellar mass
        c = cmap(norm(res1.loc[name]['log_Ms']))
        # infall points
        ax1.errorbar(res1.loc[name]['tinf'],res1.loc[name]['%Ms_tinf']/100,
                      xerr=[[res1.loc[name]['tinf-']],[res1.loc[name]['tinf+']]],
                      yerr=[[res1.loc[name]['%Ms_tinf-']/100],
                            [res1.loc[name]['%Ms_tinf+']/100]],
                      elinewidth=1, capsize=5, ecolor=c, marker='o', mec=c, mfc=c,
                      markersize=8)
        # pericenter points
        ax1.errorbar(res1.loc[name]['tperi'],res1.loc[name]['%Ms_tperi']/100,
                     xerr=[[res1.loc[name]['tperi-']],[res1.loc[name]['tperi+']]],
                     yerr=[[res1.loc[name]['%Ms_tperi-']/100],
                           [res1.loc[name]['%Ms_tperi+']/100]],
                     elinewidth=1, capsize=5, ecolor=c, marker='*', mec=c, 
                     mfc=c, markersize=8)
        ## plot-2
        ax2.errorbar(res2.loc[name]['log_Ms'],res2.loc[name]['m_%Ms_ip'],
                     yerr=[[res2.loc[name]['e-']],[res2.loc[name]['e+']]],
                     elinewidth=1, capsize=5, ecolor='k', marker='D', mec='k', 
                     mfc='k', markersize=8)
    # plot-2: regression line
    lin_reg_x = np.linspace(9.,11.,21)
    xin = np.array(res2['log_Ms'])
    yin = np.array(res2['m_%Ms_ip'])
    lin_reg_y, b, m = lin_reg(xin,yin,lin_reg_x)
    b = '{:.2f}'.format(b)
    m = '{:.2f}'.format(m[0])
    ax2.plot(lin_reg_x,lin_reg_y,c='m',linestyle='-',linewidth=4)
    # plot-1: settings
    ax1.set_ylim(0.50,1.05)
    ax1.set_yticks(ax1.get_yticks()[1:-1]) # Remove first and last ticks
    ax1.set_xlim(-1,11.0)
    ax1.set_ylabel(r'Fraction of cumulative $M_\star$ formed',fontsize=18)
    ax1.set_xlabel('Lookback time [Gyr]',fontsize=18)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(axis='both',which='major',direction='in', bottom = True, 
                            top = True,left = True, right = True, length=10, 
                            labelsize=18)
    ax1.tick_params(axis='both',which='minor',direction='in', bottom = True, 
                            top = True,left = True, right = True, length=5, 
                            labelsize=18)
    circ = mlines.Line2D([], [], color='grey', marker='o', linestyle='None',
                                  markersize=8, label='Infall time') 
    star = mlines.Line2D([], [], color='grey', marker='*', linestyle='None',
                                  markersize=10, label='Pericenter time')
    ax1.legend(handles=[circ, star],frameon=True, framealpha=0.7, 
              edgecolor='k', fontsize=15, loc=3, bbox_to_anchor=(0.02,0.02), 
              bbox_transform=ax1.transAxes) 
    cbar_divider = make_axes_locatable(ax1)
    cbar_ax = cbar_divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(smap, ticks=[9., 9.5, 10.0, 10.5, 11.0],cax=cbar_ax)
    cbar.set_label(r'$\log(M_\star/\mathrm{M}_\odot)$',fontsize=18)
    cbar_ax.tick_params(axis='y', direction='in',length=10, labelsize=18)
    ax1.grid(False)
    ax1.set_facecolor('w')
    # plot-2 settings
    ax2.set_ylim(-0.5,30.5)
    ax2.set_yticks(ax2.get_yticks()[1:-1]) # Remove first and last ticks
    ax2.set_xlim(8.8,11.2)
    ax2.set_xticks(ax2.get_xticks()[1:-1]) # Remove first and last ticks
    ax2.set_xlabel(r'$\log(M_\star/\mathrm{M}_\odot)$',
                  fontsize=18)
    ax2.set_ylabel(r'Fractional $M_\star$ increase from $t_\mathrm{inf}$ to $t_\mathrm{peri}$',
                      fontsize=18)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.tick_params(axis='both',which='major',direction='in', bottom = True, 
                            top = True,left = True, right = True, length=10, 
                            labelsize=18)
    ax2.tick_params(axis='both',which='minor',direction='in', bottom = True, 
                            top = True,left = True, right = True, length=5, 
                            labelsize=18)
    dia = mlines.Line2D([], [], color='k', marker='D', 
                        linestyle='None', markersize=8, 
                        label=r'$\frac{\Delta M_{\star,\mathrm{inf-peri}}}{M_{\star,\mathrm{final}}}\,\%\,:\,y$')
    line = mlines.Line2D([], [], color='m', marker='None', linestyle='-',
                         markersize=8,linewidth=4, 
                         label=r'$y=$'+m+r'$\,\log(M_\star/\mathrm{M}_\odot)+\,$'+b)
    # ax.legend(handles=[dia,vline,line],frameon=False, framealpha=1.0,loc=2,
    #           fontsize=14)
    ax2.legend(handles=[dia,line],frameon=True, framealpha=0.7, 
              edgecolor='k', fontsize=14, loc=2, bbox_to_anchor=(0.02,0.98), 
              bbox_transform=ax2.transAxes)
    ax2.grid(False)
    ax2.set_facecolor('w')
    # fig setting
    fig.tight_layout()
    # saving fig
    out_dir = gen_out_dir(ssp)
    plt.savefig(out_dir+ssp+'_final_plot.pdf',dpi=500)

###############################################################################
## Combined final plot ##
###############################################################################

#ssps = ['miles','bc03','phr']
ssps = ['miles']
for ssp in ssps:
    plot_all(ssp)
