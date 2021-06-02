# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 07:01:54 2021

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
from matplotlib.ticker import AutoMinorLocator

# sat names
gmp_names = ['3254','3269', '3291', '3329', '3352', '3367', '3414', '3484',
            '3534', '3565', '3639', '3664']

# output directory
def gen_out_dir(ssp):
    out_dir = './sys_err_coma/'+ssp+'/'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    return out_dir

# reading files - cumMs
def read_file_cumMs(ext,ssp):
    df = pd.read_csv('./cumMs_at_inf_peri/'+ssp+'/'+ext+'/'+ext+'_'+ssp+'_'+'cum_Ms_at_exp_orb_time_table.csv')
    df.set_index('GMP', inplace = True)
    df.astype(float)
    return df

# reads dataframes cumMs
def read_req_dfs_cumMs(ssp):
    df = read_file_cumMs('Rvir_Ms',ssp)
    df1 = read_file_cumMs('Rvir_Ms+',ssp)
    df2 = read_file_cumMs('Rvir_Ms-',ssp)
    df3 = read_file_cumMs('Rvir+_Ms',ssp)
    df4 = read_file_cumMs('Rvir-_Ms',ssp)
    return df, df1, df2, df3, df4

# reading files - fration of Ms formed b/w inf-peri
def read_file_fMsIP(ext,ssp):
    df = pd.read_csv('./f_ms_inf_peri/'+ssp+'/'+ext+'/'+ssp+'_f_ms_inf_peri.csv')
    df.set_index('GMP', inplace = True)
    df.astype(float)
    return df

# reads dataframes for fraction of Ms formed b/w inf-peri
def read_req_dfs_fMsIP(ssp):
    df = read_file_fMsIP('Rvir_Ms',ssp)
    df1 = read_file_fMsIP('Rvir_Ms+',ssp)
    df2 = read_file_fMsIP('Rvir_Ms-',ssp)
    df3 = read_file_fMsIP('Rvir+_Ms',ssp)
    df4 = read_file_fMsIP('Rvir-_Ms',ssp)
    return df, df1, df2, df3, df4

# computes linear regression
def lin_reg(x,y,xnew):
    x = x.reshape(-1,1)
    model = lr().fit(x,y)
    intercept = model.intercept_
    slope = model.coef_
    xnew = xnew.reshape(-1,1)
    ynew = model.predict(xnew)
    return ynew, intercept, slope

# plotting and saving
def plot_all(ssp):
    
    ## Reading - cumMs system errors
    cdf, cdf1, cdf2, cdf3, cdf4 = read_req_dfs_cumMs(ssp)
    ## reading stellar mass
    log_Ms = np.array(pd.read_csv('smhm_behroozi2010.csv')['logMs'])
    
    ## Reading - fMsIP system errors
    fdf, fdf1, fdf2, fdf3, fdf4 = read_req_dfs_fMsIP(ssp)
    
    
    #### Plotting
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(21,9))
    
    for name, log_Mi in zip(gmp_names, log_Ms):
        col = 'tab:gray'
        if name == '3329':
            continue
        else:
            name = int(name)
            
            ### Plotting cumMs - syserr
            
            ## R+M
            # plotting original points: inf and peri and annotatig galaxy names
            ax1.scatter(cdf.loc[name]['tinf'], cdf.loc[name]['%Ms_tinf']/100,
                        marker='o', c=col, s=200)
            ax1.scatter(cdf.loc[name]['tperi'], cdf.loc[name]['%Ms_tperi']/100,
                        marker='*', c=col, s=250)
            ## R, M+
            ax1.annotate('', xy = (cdf1.loc[name]['tinf'], 
                                   cdf1.loc[name]['%Ms_tinf']/100),
                         xytext = (cdf.loc[name]['tinf'], 
                                   cdf.loc[name]['%Ms_tinf']/100), 
                         arrowprops=dict(arrowstyle='->',color='k',
                                         linestyle='-.',linewidth=1.5))
            ax1.annotate('', xy = (cdf1.loc[name]['tperi'], 
                                   cdf1.loc[name]['%Ms_tperi']/100),
                         xytext = (cdf.loc[name]['tperi'], 
                                   cdf.loc[name]['%Ms_tperi']/100), 
                         arrowprops=dict(arrowstyle='->',color='k',
                                         linestyle='-.',linewidth=1.5))
            ## R, M-
            ax1.annotate('', xy = (cdf2.loc[name]['tinf'], 
                                   cdf2.loc[name]['%Ms_tinf']/100),
                         xytext = (cdf.loc[name]['tinf'], 
                                   cdf.loc[name]['%Ms_tinf']/100), 
                         arrowprops=dict(arrowstyle='->',color='k',
                                         linestyle='--',linewidth=1.5))
            ax1.annotate('', xy = (cdf2.loc[name]['tperi'], 
                                   cdf2.loc[name]['%Ms_tperi']/100),
                         xytext = (cdf.loc[name]['tperi'], 
                                   cdf.loc[name]['%Ms_tperi']/100), 
                         arrowprops=dict(arrowstyle='->',color='k',
                                         linestyle='--',linewidth=1.5))            
            ## R+, M
            ax1.annotate('', xy = (cdf3.loc[name]['tinf'], 
                                   cdf3.loc[name]['%Ms_tinf']/100),
                         xytext = (cdf.loc[name]['tinf'], 
                                   cdf.loc[name]['%Ms_tinf']/100), 
                         arrowprops=dict(arrowstyle='->',color='r',
                                         linestyle='-.',linewidth=1.5))
            ax1.annotate('', xy = (cdf3.loc[name]['tperi'], 
                                   cdf3.loc[name]['%Ms_tperi']/100),
                         xytext = (cdf.loc[name]['tperi'], 
                                   cdf.loc[name]['%Ms_tperi']/100), 
                         arrowprops=dict(arrowstyle='->',color='r',
                                         linestyle='-.',linewidth=1.5))
            ## R-, M
            ax1.annotate('', xy = (cdf4.loc[name]['tinf'], 
                                   cdf4.loc[name]['%Ms_tinf']/100),
                         xytext = (cdf.loc[name]['tinf'], 
                                   cdf.loc[name]['%Ms_tinf']/100), 
                         arrowprops=dict(arrowstyle='->',color='r',
                                         linestyle='--',linewidth=1.5))
            ax1.annotate('', xy = (cdf4.loc[name]['tperi'], 
                                   cdf4.loc[name]['%Ms_tperi']/100),
                         xytext = (cdf.loc[name]['tperi'], 
                                   cdf.loc[name]['%Ms_tperi']/100), 
                         arrowprops=dict(arrowstyle='->',color='r',
                                         linestyle='--',linewidth=1.5))
            
            ### PLotting fMsIP - system error
            ## R, M
            ax2.plot(fdf.loc[name]['log_Ms'], fdf.loc[name]['m_%Ms_ip'], c=col, 
                     marker='D', markersize=12)
            ## R, M+
            ax2.annotate('', xy = (fdf1.loc[name]['log_Ms'], 
                                  fdf1.loc[name]['m_%Ms_ip']), 
                        xytext = (fdf.loc[name]['log_Ms'], 
                                  fdf.loc[name]['m_%Ms_ip']), 
                        arrowprops=dict(arrowstyle='->',color='k', 
                                        linestyle='-.', linewidth=1.5))
            ## R, M-
            ax2.annotate('', xy = (fdf2.loc[name]['log_Ms'], 
                                   fdf2.loc[name]['m_%Ms_ip']), 
                         xytext = (fdf.loc[name]['log_Ms'], 
                                   fdf.loc[name]['m_%Ms_ip']), 
                         arrowprops=dict(arrowstyle='->',color='k', 
                                         linestyle='--', linewidth=1.5))
            ## R+, M
            ax2.annotate('', xy = (fdf3.loc[name]['log_Ms'], 
                                   fdf3.loc[name]['m_%Ms_ip']), 
                         xytext = (fdf.loc[name]['log_Ms'], 
                                   fdf.loc[name]['m_%Ms_ip']), 
                         arrowprops=dict(arrowstyle='->',color='r', 
                                         linestyle='-.', linewidth=1.5))
            ## R-, M
            ax2.annotate('', xy = (fdf4.loc[name]['log_Ms'], 
                                   fdf4.loc[name]['m_%Ms_ip']), 
                         xytext = (fdf.loc[name]['log_Ms'], 
                                   fdf.loc[name]['m_%Ms_ip']), 
                         arrowprops=dict(arrowstyle='->',color='r', 
                                         linestyle='--', linewidth=1.5))
    
    ## linear regression - R, M
    lin_reg_x = np.linspace(9.,11.,21)
    xin = np.array(fdf['log_Ms'])
    yin = np.array(fdf['m_%Ms_ip'])
    lin_reg_y, b, m = lin_reg(xin,yin,lin_reg_x)
    b = '{:.2f}'.format(b)
    m = '{:.2f}'.format(m[0])
    ax2.plot(lin_reg_x,lin_reg_y,c='tab:gray', linestyle='-', linewidth=4)
    
    ## linear regression - R, M+
    lin_reg_x = np.linspace(9.,11.,21)
    xin = np.array(fdf1['log_Ms'])
    yin = np.array(fdf1['m_%Ms_ip'])
    lin_reg_y, b, m = lin_reg(xin,yin,lin_reg_x)
    b = '{:.2f}'.format(b)
    m = '{:.2f}'.format(m[0])
    ax2.plot(lin_reg_x,lin_reg_y,c='k', linestyle='-.', linewidth=1.5)
    
    ## linear regression - R, M-
    lin_reg_x = np.linspace(9.,11.,21)
    xin = np.array(fdf2['log_Ms'])
    yin = np.array(fdf2['m_%Ms_ip'])
    lin_reg_y, b, m = lin_reg(xin,yin,lin_reg_x)
    b = '{:.2f}'.format(b)
    m = '{:.2f}'.format(m[0])
    ax2.plot(lin_reg_x,lin_reg_y,c='k', linestyle='--', linewidth=1.5)
    
    ## linear regression - R+, M
    lin_reg_x = np.linspace(9.,11.,21)
    xin = np.array(fdf3['log_Ms'])
    yin = np.array(fdf3['m_%Ms_ip'])
    lin_reg_y, b, m = lin_reg(xin,yin,lin_reg_x)
    b = '{:.2f}'.format(b)
    m = '{:.2f}'.format(m[0])
    ax2.plot(lin_reg_x,lin_reg_y,c='r', linestyle='-.', linewidth=1.5)
    
    ## linear regression - R-, M
    lin_reg_x = np.linspace(9.,11.,21)
    xin = np.array(fdf4['log_Ms'])
    yin = np.array(fdf4['m_%Ms_ip'])
    lin_reg_y, b, m = lin_reg(xin,yin,lin_reg_x)
    b = '{:.2f}'.format(b)
    m = '{:.2f}'.format(m[0])
    ax2.plot(lin_reg_x,lin_reg_y,c='r', linestyle='--', linewidth=1.5)
    
    ### Plot setting cumMs - system error
    ax1.set_ylim(0.77,1.01)
    ax1.set_yticks([0.80,0.85,0.90,0.95,1.00])
    #ax1.set_yticks(ax1.get_yticks()[1:-1]) # Remove first and last ticks
    ax1.set_xlim(0.,9.5)
    ax1.set_xticks([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.])
    #ax1.set_xticks(ax1.get_xticks()[1:-1]) # Remove first and last ticks
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(axis='both',which='major',direction='in', bottom = True, 
                       top = True,left = True, right = True, length=10, pad=15,
                       labelsize=18)
    ax1.tick_params(axis='both',which='minor',direction='in', bottom = True, 
                       top = True,left = True, right = True, length=5, pad=15,
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
    ax1.legend(handles=[circ,star,rvir,ms,plus,minus],frameon=True, 
              framealpha=1.0, edgecolor='k', fontsize=18, loc=3, 
              bbox_to_anchor=(0.03,0.02), bbox_transform=ax1.transAxes)
    ax1.grid(False)
    ax1.set_facecolor('w')
    ax1.set_xlabel('Lookback time [Gyr]',fontsize=20)
    ax1.set_ylabel(r'Fraction of cumulative $M_\star$ formed',fontsize=20)
    
    ### Plot setting cumMs - system error
    ax2.set_ylim(0.,14.)
    ax2.set_yticks([0.,2.,4.,6.,8.,10.,12.,14.])
    #ax2.set_yticks(ax2.get_yticks()[1:-1]) # Remove first and last ticks
    ax2.set_xlim(9.,11.)
    ax2.set_xticks([9.0,9.5,10.0,10.5,11.0])
    #ax2.set_xticks(ax2.get_xticks()[1:-1]) # Remove first and last ticks
    ax2.set_xlabel(r'$\log(M_\star/\mathrm{M}_\odot)$', fontsize=20)
    ax2.set_ylabel(r'Fractional $M_\star$ increase from $t_\mathrm{inf}$ to $t_\mathrm{peri}$', 
                  fontsize=20)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.tick_params(axis='both',which='major',direction='in', bottom = True, 
                   top = True,left = True, right = True, length=10, pad=15, 
                   labelsize=18)
    ax2.tick_params(axis='both',which='minor',direction='in', bottom = True, 
                   top = True,left = True, right = True, length=5, pad=15, 
                   labelsize=18)
    ax2.grid(False)
    ax2.set_facecolor('w')
    
    dia = mlines.Line2D([], [], color='tab:gray', marker='D', 
                        linestyle='None', markersize=8, 
                        label=r'$\frac{\Delta M_{\star,\mathrm{inf-peri}}}{M_{\star,\mathrm{final}}}\,\%$')
    reg = mlines.Line2D([], [], color='tab:gray', linestyle='-', 
                         linewidth=4., label='Linear fit')
    plus = mlines.Line2D([], [], color='tab:gray', linestyle='-.', 
                         linewidth=2., label=r'$+$')
    minus = mlines.Line2D([], [], color='tab:gray', linestyle='--', 
                          linewidth=2., label=r'$-$')
    rvir = mlines.Line2D([], [], color='r', marker='>', linestyle='-', 
                         markersize=8, markeredgewidth=2, 
                         label=r'$R_{\rm vir}\,\pm 10\%$')
    ms = mlines.Line2D([], [], color='k', marker='>', linestyle='-', 
                         markersize=8, markeredgewidth=2, 
                         label=r'$M_\mathrm{h}\,\pm 0.5 \, \mathrm{dex}$')
    ax2.legend(handles=[dia,reg,rvir,ms,plus,minus],frameon=True, 
              framealpha=1.0, edgecolor='k', fontsize=18, loc=3, 
              bbox_to_anchor=(0.03,0.02), bbox_transform=ax2.transAxes)
    
    ### saving the figure in  PDF file
    out_dir = gen_out_dir(ssp)
    plt.savefig(out_dir+ssp+'_'+'sys_err_coma_plot.pdf',dpi=500)
    
###############################################################################
## Plotting Shift in results due to uncertainty in Rvir,Coma and M* of sats ##
###############################################################################

#ssps = ['miles','bc03','phr']

ssps = ['miles']

for ssp in ssps:
    plot_all(ssp)