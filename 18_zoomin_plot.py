# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 09:52:27 2020

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

ssps = ['miles','bc03','phr']

# reads dataframes for zoom-in computations and plotting
def read_req_dfs(ssp):
    df = read_file('Rvir_Ms',ssp)
    df3 = read_file('Rvir+_Ms',ssp)
    df4 = read_file('Rvir+_Ms+',ssp)
    df5 = read_file('Rvir+_Ms-',ssp)
    df6 = read_file('Rvir-_Ms',ssp)
    df7 = read_file('Rvir-_Ms+',ssp)
    df8 = read_file('Rvir-_Ms-',ssp)
    return df, df3, df4, df5, df6, df7, df8

# plotting and saving
def plot_all(ssp):
    df, df3, df4, df5, df6, df7, df8 = read_req_dfs(ssp)
    # reading stellar mass
    log_Ms = np.array(pd.read_csv('smhm_behroozi2010.csv')['logMs'])
    # log_Msp = np.array(pd.read_csv('smhm_behroozi2010.csv')['logMs+'])
    # log_Msm = np.array(pd.read_csv('smhm_behroozi2010.csv')['logMs-'])
    
    # plotting
    fig, (ax3, ax5) = plt.subplots(2,1,figsize=(10,10))
    for name, log_Mi in zip(gmp_names, log_Ms):
        if name == '3329':
            continue
        else:
            name = int(name)
            col = 'k'
            
            ## R+M
            # plotting original points: inf and peri and annotatig galaxy names
            ax3.scatter(df.loc[name]['tinf'], df.loc[name]['%Ms_tinf']/100,
                        marker='o', c=col, s=100)
            # ax3.annotate(str(name), (df.loc[name]['tinf'], 
            #                          df.loc[name]['%Ms_tinf']/100),
            #              xytext=(5, 3), textcoords='offset points',color = 'g', 
            #              fontsize=12)
            ax3.scatter(df.loc[name]['tperi'], df.loc[name]['%Ms_tperi']/100,
                        marker='*', c=col, s=100)
            # ax3.annotate(str(name), (df.loc[name]['tperi'],
            #                              df.loc[name]['%Ms_tperi']/100),
            #                  xytext=(5, 3), textcoords='offset points',
            #                  color = 'g', fontsize=12)
            # plotting shifted points due to R+ and dashed line connecting
            # ax3.scatter(df3.loc[name]['tinf'], df3.loc[name]['%Ms_tinf']/100,
            #             marker='o', c=css['slategray'], s=50)
            # ax3.scatter(df3.loc[name]['tperi'], df3.loc[name]['%Ms_tperi']/100,
            #             marker='*', c=css['slategray'], s=50)
            ax3.annotate('', xy = (df3.loc[name]['tinf'], 
                                   df3.loc[name]['%Ms_tinf']/100),
                         xytext = (df.loc[name]['tinf'], 
                                   df.loc[name]['%Ms_tinf']/100), 
                         arrowprops=dict(arrowstyle='->',color='r',
                                         linestyle='-',linewidth=1.5))
            ax3.annotate('', xy = (df3.loc[name]['tperi'], 
                                   df3.loc[name]['%Ms_tperi']/100),
                         xytext = (df.loc[name]['tperi'], 
                                   df.loc[name]['%Ms_tperi']/100), 
                         arrowprops=dict(arrowstyle='->',color='r',
                                         linestyle='-',linewidth=1.5))
            # plotting arrows to indicate M+- shift
            ax3.annotate('', xy = (df4.loc[name]['tinf'], 
                                   df4.loc[name]['%Ms_tinf']/100),
                         xytext = (df3.loc[name]['tinf'], 
                                   df3.loc[name]['%Ms_tinf']/100), 
                         arrowprops=dict(arrowstyle='->',color='b',
                                         linestyle='-',linewidth=1.5))
            ax3.annotate('', xy = (df5.loc[name]['tinf'], 
                                   df5.loc[name]['%Ms_tinf']/100),
                         xytext = (df3.loc[name]['tinf'], 
                                   df3.loc[name]['%Ms_tinf']/100), 
                         arrowprops=dict(arrowstyle='->',color='b',
                                         linestyle='--',linewidth=1.5))
            ax3.annotate('', xy = (df4.loc[name]['tperi'], 
                                   df4.loc[name]['%Ms_tperi']/100),
                         xytext = (df3.loc[name]['tperi'], 
                                   df3.loc[name]['%Ms_tperi']/100), 
                         arrowprops=dict(arrowstyle='->',color='b',
                                         linestyle='-',linewidth=1.5))
            ax3.annotate('', xy = (df5.loc[name]['tperi'], 
                                   df5.loc[name]['%Ms_tperi']/100),
                         xytext = (df3.loc[name]['tperi'], 
                                   df3.loc[name]['%Ms_tperi']/100), 
                         arrowprops=dict(arrowstyle='->',color='b',
                                         linestyle='--',linewidth=1.5))
            
            # arrow for Rvir-, Ms+
            ax5.scatter(df.loc[name]['tinf'], df.loc[name]['%Ms_tinf']/100,
                        marker='o', c=col, s=100)
            # ax5.annotate(str(name), (df.loc[name]['tinf'], 
            #                          df.loc[name]['%Ms_tinf']/100),
            #              xytext=(5, 3), textcoords='offset points', color = 'g', 
            #              fontsize=12)
            ax5.scatter(df.loc[name]['tperi'], df.loc[name]['%Ms_tperi']/100,
                        marker='*', c=col, s=100)
            # ax5.annotate(str(name), (df.loc[name]['tperi'], 
            #                              df.loc[name]['%Ms_tperi']/100),
            #                  xytext=(5, 3), textcoords='offset points', 
            #                  color = 'g', fontsize=12)
            
            # ax5.scatter(df6.loc[name]['tinf'], df6.loc[name]['%Ms_tinf']/100,
            #             marker='o', c=css['slategray'], s=50)
            # ax5.scatter(df6.loc[name]['tperi'], df6.loc[name]['%Ms_tperi']/100,
            #             marker='*', c=css['slategray'], s=50)
            ax5.annotate('', xy = (df6.loc[name]['tinf'], 
                                   df6.loc[name]['%Ms_tinf']/100),
                         xytext = (df.loc[name]['tinf'], 
                                   df.loc[name]['%Ms_tinf']/100), 
                         arrowprops=dict(arrowstyle='->',color='r',
                                         linestyle='--',linewidth=1.5))
            ax5.annotate('', xy = (df6.loc[name]['tperi'], 
                                   df6.loc[name]['%Ms_tperi']/100),
                         xytext = (df.loc[name]['tperi'], 
                                   df.loc[name]['%Ms_tperi']/100), 
                         arrowprops=dict(arrowstyle='->',color='r',
                                         linestyle='--',linewidth=1.5))
            ax5.annotate('', xy = (df7.loc[name]['tinf'], 
                                   df7.loc[name]['%Ms_tinf']/100),
                         xytext = (df6.loc[name]['tinf'], 
                                   df6.loc[name]['%Ms_tinf']/100), 
                         arrowprops=dict(arrowstyle='->',color='b',
                                         linestyle='-',linewidth=1.5))
            ax5.annotate('', xy = (df8.loc[name]['tinf'], 
                                   df8.loc[name]['%Ms_tinf']/100),
                         xytext = (df6.loc[name]['tinf'], 
                                   df6.loc[name]['%Ms_tinf']/100), 
                         arrowprops=dict(arrowstyle='->',color='b',
                                         linestyle='--',linewidth=1.5))
            ax5.annotate('', xy = (df7.loc[name]['tperi'], 
                                   df7.loc[name]['%Ms_tperi']/100),
                         xytext = (df6.loc[name]['tperi'], 
                                   df6.loc[name]['%Ms_tperi']/100), 
                         arrowprops=dict(arrowstyle='->',color='b',
                                         linestyle='-',linewidth=1.5))
            ax5.annotate('', xy = (df8.loc[name]['tperi'], 
                                   df8.loc[name]['%Ms_tperi']/100),
                         xytext = (df6.loc[name]['tperi'], 
                                   df6.loc[name]['%Ms_tperi']/100), 
                         arrowprops=dict(arrowstyle='->',color='b',
                                         linestyle='--',linewidth=1.5))
    
    
    ax3.set_ylim(0.75,1.01)
    ax3.set_yticks(ax3.get_yticks()[1:-1]) # Remove first and last ticks
    ax3.set_xlim(0.5,9.0)
    #ax3.set_ylabel(r'Fraction of cumulative $M_\star$ formed',fontsize=18)
    # ax3.set_xlabel('Lookback time [Gyr]',fontsize=18)
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    ax3.tick_params(axis='both',which='both',direction='in', bottom = True, 
                       top = True,left = True, right = True,labelsize=18)
    ax5.set_ylim(0.75,1.01)
    ax5.set_yticks(ax5.get_yticks()[1:-1]) # Remove first and last ticks
    ax5.set_xlim(0.5,9.0)
    #ax5.set_ylabel(r'Fraction of cumulative $M_\star$ formed',fontsize=18)
    #ax5.set_xlabel('Lookback time [Gyr]',fontsize=18)
    ax5.xaxis.set_minor_locator(AutoMinorLocator())
    ax5.yaxis.set_minor_locator(AutoMinorLocator())
    ax5.tick_params(axis='both',which='both',direction='in', bottom = True, 
                       top = True,left = True, right = True,labelsize=18)
    
    # ax4.set_ylim(0.95,1.01)
    # ax4.set_yticks(ax4.get_yticks()[1:-1]) # Remove first and last ticks
    # ax4.set_xlim(0.5,5.5)
    # # ax4.set_ylabel(r'Fraction of cumulative $M_\star$ formed',fontsize=18)
    # # ax4.set_xlabel('Lookback time [Gyr]',fontsize=18)
    # ax4.xaxis.set_minor_locator(AutoMinorLocator())
    # ax4.yaxis.set_minor_locator(AutoMinorLocator())
    # ax4.tick_params(axis='both',which='both',direction='in', bottom = True, 
    #                    top = True,left = True, right = True,labelsize=18)
    # ax6.set_ylim(0.95,1.01)
    # ax6.set_yticks(ax6.get_yticks()[1:-1]) # Remove first and last ticks
    # ax6.set_xlim(0.5,5.5)
    # # ax6.set_ylabel(r'Fraction of cumulative $M_\star$ formed',fontsize=18)
    # ax6.set_xlabel('Lookback time [Gyr]',fontsize=18)
    # ax6.xaxis.set_minor_locator(AutoMinorLocator())
    # ax6.yaxis.set_minor_locator(AutoMinorLocator())
    # ax6.tick_params(axis='both',which='both',direction='in', bottom = True, 
    #                    top = True,left = True, right = True,labelsize=18)
    
    
    circ = mlines.Line2D([], [], color='grey', marker='o', linestyle='None',
                              markersize=8, label='Infall time') 
    star = mlines.Line2D([], [], color='grey', marker='*', linestyle='None',
                              markersize=10, label='Pericenter time')
    plus = mlines.Line2D([], [], color='grey', linestyle='-', linewidth=2., 
                         label=r'$+$')
    minus = mlines.Line2D([], [], color='grey', linestyle='--', linewidth=2., 
                         label=r'$-$')
    rvir = mlines.Line2D([], [], color='r', marker='>', linestyle='-', 
                         markersize=8, markeredgewidth=2, label=r'$R_{\rm vir}$')
    ms = mlines.Line2D([], [], color='b', marker='>', linestyle='-', 
                         markersize=8, markeredgewidth=2, label=r'$M_\star$')
    # circrp = mlines.Line2D([], [], color=css['slategray'], marker='o', linestyle='None',
    #                           markersize=8, label='Rvir+')
    # circrm = mlines.Line2D([], [], color=css['slategray'], marker='o', linestyle='None',
    #                           markersize=8, label='Rvir-') 
    # starrp = mlines.Line2D([], [], color=css['slategray'], marker='*', linestyle='None',
    #                           markersize=8, label='Rvir+')
    # starrm = mlines.Line2D([], [], color=css['slategray'], marker='*', linestyle='None',
    #                           markersize=8, label='Rvir-')
    
    ax3.legend(handles=[circ,star,rvir,ms,plus,minus],frameon=False, 
               framealpha=1.0, loc=3, fontsize=18)
    ax5.legend(handles=[circ,star,rvir,ms,plus,minus],frameon=False, 
               framealpha=1.0, loc=3, fontsize=18)
    # ax4.legend(handles=[star,starrp,arrrp,arrmp,arrmm],frameon=False, framealpha=1.0,loc=3,
    #            fontsize=14)
    # ax6.legend(handles=[star,starrm,arrrm,arrmp,arrmm],frameon=False, framealpha=1.0,loc=3,
    #            fontsize=14) 
    
    # divider4 = make_axes_locatable(ax4)
    # cax4 = divider4.append_axes('right', size='5%', pad=0.05)
    # cbar4 = fig.colorbar(smap, ticks=[9., 9.5, 10.0, 10.5, 11.0],cax=cax4)
    # cbar4.set_label(r'$\log_{10}(M_\star/{\rm M}_\odot)$',fontsize=18)
    # cax4.tick_params(axis='y', direction='in',labelsize=18)
    
    # divider6 = make_axes_locatable(ax6)
    # cax6 = divider6.append_axes('right', size='5%', pad=0.05)
    # cbar6 = fig.colorbar(smap, ticks=[9., 9.5, 10.0, 10.5, 11.0],cax=cax6)
    # cbar6.set_label(r'$\log_{10}(M_\star/{\rm M}_\odot)$',fontsize=18)
    # cax6.tick_params(axis='y', direction='in',labelsize=18)
    
    ax3.grid(False)
    ax3.set_facecolor('w')
    # ax4.grid(False)
    # ax4.set_facecolor('w')
    ax5.grid(False)
    ax5.set_facecolor('w')
    # ax6.grid(False)
    # ax6.set_facecolor('w')
    
    ax3.set_title(r'$R_{\rm vir,Coma}=R_{\rm vir,Coma}+10$%, $M_\star=M_\star\pm25$%',
                  fontsize=18)
    ax5.set_title(r'$R_{\rm vir,Coma}=R_{\rm vir,Coma}-10$%, $M_\star=M_\star\pm25$%',
                  fontsize=18)
    # ax4.set_title(r'$R_{\rm vir,Coma}=R_{\rm vir,Coma}+0.3$, $M_\star=M_\star\pm25$%',
    #               fontsize=18)
    # ax6.set_title(r'$R_{\rm vir,Coma}=R_{\rm vir,Coma}-0.3$, $M_\star=M_\star\pm25$%',
    #               fontsize=18)
    
    
    # fig.legend(handles=[arrmp,arrmm,arrrp,arrrm,circrp,circrm,starrp,starrm],
    #            frameon=True, framealpha=1.0,loc=3,fontsize=16)
    
    xlab = 'Lookback time [Gyr]'
    ylab = r'Fraction of cumulative $M_\star$ formed'
    fig.text(0.5, 0.06, xlab, ha='center',fontsize=18)
    fig.text(0.02, 0.5, ylab, va='center', rotation='vertical',fontsize=18)
    
    #plt.tight_layout()
    out_dir = gen_out_dir(ssp)
    plt.savefig(out_dir+ssp+'_'+'zoomin_plot.pdf',dpi=500)

###############################################################################
## Plotting Shift in results due to uncertainty in Rvir,Coma and M* of sats ##
###############################################################################

ssps = ['miles','bc03','phr']

for ssp in ssps:
    plot_all(ssp)
   