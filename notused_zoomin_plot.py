# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 04:39:43 2020

@author: amit
"""

import warnings
warnings.filterwarnings("ignore")

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
#import matplotlib.colors as mcolors
# from matplotlib.cm import ScalarMappable, viridis as cmap
# from matplotlib.colors import Normalize
from matplotlib.ticker import AutoMinorLocator
#from mpl_toolkits.axes_grid1 import make_axes_locatable

# use color pallette
#css  = mcolors.CSS4_COLORS

# sat names
gmp_names = ['3254','3269', '3291', '3329', '3352', '3367', '3414', '3484',
            '3534', '3565', '3639', '3664']

# output directory
def gen_out_dir(ssp):
    out_dir = './zoomin_plots/'+ssp+'/'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    return out_dir

# reading files
def read_file(ext,ssp):
    df = pd.read_csv('./cumMs_at_inf_peri/'+ssp+'/'+ext+'/'+ext+'_'+ssp+'_'+'cum_Ms_at_exp_orb_time_table.csv')
    df.set_index('GMP', inplace = True)
    df.astype(float)
    return df

#ssps = ['miles','bc03','phr']

# reads dataframes for zoom-in computations and plotting
def read_req_dfs(ssp):
    df = read_file('Rvir_Ms',ssp)
    df1 = read_file('Rvir_Ms+',ssp)
    df2 = read_file('Rvir_Ms-',ssp)
    df3 = read_file('Rvir+_Ms',ssp)
    df4 = read_file('Rvir-_Ms',ssp)
    return df, df1, df2, df3, df4

# plotting and saving
def plot_all(ssp):
    df, df1, df2, df3, df4 = read_req_dfs(ssp)
    # reading stellar mass
    log_Ms = np.array(pd.read_csv('smhm_behroozi2010.csv')['logMs'])
    # log_Msp = np.array(pd.read_csv('smhm_behroozi2010.csv')['logMs+'])
    # log_Msm = np.array(pd.read_csv('smhm_behroozi2010.csv')['logMs-'])
    
    # plotting
    fig, ax = plt.subplots(figsize=(10,9))
    for name, log_Mi in zip(gmp_names, log_Ms):
        col = 'tab:gray'
        if name == '3329':
            continue
        else:
            name = int(name)
            
            ## R+M
            # plotting original points: inf and peri and annotatig galaxy names
            ax.scatter(df.loc[name]['tinf'], df.loc[name]['%Ms_tinf']/100,
                        marker='o', c=col, s=200)
            ax.scatter(df.loc[name]['tperi'], df.loc[name]['%Ms_tperi']/100,
                        marker='*', c=col, s=250)
            ## R, M+
            ax.annotate('', xy = (df1.loc[name]['tinf'], 
                                   df1.loc[name]['%Ms_tinf']/100),
                         xytext = (df.loc[name]['tinf'], 
                                   df.loc[name]['%Ms_tinf']/100), 
                         arrowprops=dict(arrowstyle='->',color='k',
                                         linestyle='-.',linewidth=1.5))
            ax.annotate('', xy = (df1.loc[name]['tperi'], 
                                   df1.loc[name]['%Ms_tperi']/100),
                         xytext = (df.loc[name]['tperi'], 
                                   df.loc[name]['%Ms_tperi']/100), 
                         arrowprops=dict(arrowstyle='->',color='k',
                                         linestyle='-.',linewidth=1.5))
            ## R, M-
            ax.annotate('', xy = (df2.loc[name]['tinf'], 
                                   df2.loc[name]['%Ms_tinf']/100),
                         xytext = (df.loc[name]['tinf'], 
                                   df.loc[name]['%Ms_tinf']/100), 
                         arrowprops=dict(arrowstyle='->',color='k',
                                         linestyle='--',linewidth=1.5))
            ax.annotate('', xy = (df2.loc[name]['tperi'], 
                                   df2.loc[name]['%Ms_tperi']/100),
                         xytext = (df.loc[name]['tperi'], 
                                   df.loc[name]['%Ms_tperi']/100), 
                         arrowprops=dict(arrowstyle='->',color='k',
                                         linestyle='--',linewidth=1.5))            
            ## R+, M
            ax.annotate('', xy = (df3.loc[name]['tinf'], 
                                   df3.loc[name]['%Ms_tinf']/100),
                         xytext = (df.loc[name]['tinf'], 
                                   df.loc[name]['%Ms_tinf']/100), 
                         arrowprops=dict(arrowstyle='->',color='r',
                                         linestyle='-.',linewidth=1.5))
            ax.annotate('', xy = (df3.loc[name]['tperi'], 
                                   df3.loc[name]['%Ms_tperi']/100),
                         xytext = (df.loc[name]['tperi'], 
                                   df.loc[name]['%Ms_tperi']/100), 
                         arrowprops=dict(arrowstyle='->',color='r',
                                         linestyle='-.',linewidth=1.5))
            ## R-, M
            ax.annotate('', xy = (df4.loc[name]['tinf'], 
                                   df4.loc[name]['%Ms_tinf']/100),
                         xytext = (df.loc[name]['tinf'], 
                                   df.loc[name]['%Ms_tinf']/100), 
                         arrowprops=dict(arrowstyle='->',color='r',
                                         linestyle='--',linewidth=1.5))
            ax.annotate('', xy = (df4.loc[name]['tperi'], 
                                   df4.loc[name]['%Ms_tperi']/100),
                         xytext = (df.loc[name]['tperi'], 
                                   df.loc[name]['%Ms_tperi']/100), 
                         arrowprops=dict(arrowstyle='->',color='r',
                                         linestyle='--',linewidth=1.5))   
    ax.set_ylim(0.75,1.05)
    ax.set_yticks(ax.get_yticks()[1:-1]) # Remove first and last ticks
    ax.set_xlim(0.,10.0)
    ax.set_xticks(ax.get_xticks()[1:-1]) # Remove first and last ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both',which='major',direction='in', bottom = True, 
                       top = True,left = True, right = True, length=10, 
                       labelsize=18)
    ax.tick_params(axis='both',which='minor',direction='in', bottom = True, 
                       top = True,left = True, right = True, length=5,
                       labelsize=18)
    circ = mlines.Line2D([], [], color=col, marker='o', linestyle='None',
                              markersize=8, label='Infall time') 
    star = mlines.Line2D([], [], color=col, marker='*', linestyle='None',
                              markersize=10, label='Pericenter time')
    plus = mlines.Line2D([], [], color=col, linestyle='-.', linewidth=2., 
                          label=r'$+$')
    minus = mlines.Line2D([], [], color=col, linestyle='--', linewidth=2., 
                          label=r'$-$')
    rvir = mlines.Line2D([], [], color='r', marker='>', linestyle='-', 
                         markersize=8, markeredgewidth=2, 
                         label=r'$R_{\rm vir}\,\pm 10\%$')
    ms = mlines.Line2D([], [], color='k', marker='>', linestyle='-', 
                         markersize=8, markeredgewidth=2, 
                         label=r'$M_\mathrm{h}\,\pm 0.5 \, \mathrm{dex}$')
    # ax.legend(handles=[circ,star,rvir,ms,plus,minus],frameon=False, 
    #            framealpha=1.0, loc=3, fontsize=18)
    ax.legend(handles=[circ,star,rvir,ms,plus,minus],frameon=True, 
              framealpha=1.0, edgecolor='k', fontsize=18, loc=3, 
              bbox_to_anchor=(0.03,0.02), bbox_transform=ax.transAxes)
    ax.grid(False)
    ax.set_facecolor('w')
    # ax.set_title(r'$R_{\rm vir,Coma}=R_{\rm vir,Coma}\pm10$%, $M_\star=M_\star\pm25$%',
    #               fontsize=18)
    ax.set_xlabel('Lookback time [Gyr]',fontsize=18)
    ax.set_ylabel(r'Fraction of cumulative $M_\star$ formed',fontsize=18)
    
    #plt.tight_layout()
    out_dir = gen_out_dir(ssp)
    plt.savefig(out_dir+ssp+'_'+'zoomin_plot.pdf',dpi=500)

###############################################################################
## Plotting Shift in results due to uncertainty in Rvir,Coma and M* of sats ##
###############################################################################

#ssps = ['miles','bc03','phr']

ssps = ['miles']

for ssp in ssps:
    plot_all(ssp)
   