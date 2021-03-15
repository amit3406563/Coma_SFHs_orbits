# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 05:04:54 2021

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
from matplotlib.ticker import AutoMinorLocator
import matplotlib.lines as mlines

# output directory
def gen_out_dir(ssp):
    out_dir = './f_ms_inf_peri_uncert/'+ssp+'/'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    return out_dir

# reading files
def read_file(ext,ssp):
    df = pd.read_csv('./f_ms_inf_peri/'+ssp+'/'+ext+'/'+ssp+'_f_ms_inf_peri.csv')
    df.set_index('GMP', inplace = True)
    df.astype(float)
    return df

# computes linear regression
def lin_reg(x,y,xnew):
    x = x.reshape(-1,1)
    model = lr().fit(x,y)
    intercept = model.intercept_
    slope = model.coef_
    xnew = xnew.reshape(-1,1)
    ynew = model.predict(xnew)
    return ynew, intercept, slope

# reads dataframes for fraction of Ms formed b/w inf-peri computations and plotting
def read_req_dfs(ssp):
    df = read_file('Rvir_Ms',ssp)
    df1 = read_file('Rvir_Ms+',ssp)
    df2 = read_file('Rvir_Ms-',ssp)
    df3 = read_file('Rvir+_Ms',ssp)
    df4 = read_file('Rvir-_Ms',ssp)
    return df, df1, df2, df3, df4

def plot_all(ssp):
    df, df1, df2, df3, df4 = read_req_dfs(ssp)

    fig, ax = plt.subplots(figsize=(10,9))
    for name in df.index:
        ## R, M
        ax.plot(df.loc[name]['log_Ms'], df.loc[name]['m_%Ms_ip'], c='tab:gray', 
                marker='D', markersize=8)
        ## R, M+
        ax.annotate('', xy = (df1.loc[name]['log_Ms'], 
                              df1.loc[name]['m_%Ms_ip']), 
                    xytext = (df.loc[name]['log_Ms'], 
                              df.loc[name]['m_%Ms_ip']), 
                    arrowprops=dict(arrowstyle='->',color='k', linestyle='-.', 
                                    linewidth=1.5))
        ## R, M-
        ax.annotate('', xy = (df2.loc[name]['log_Ms'], 
                              df2.loc[name]['m_%Ms_ip']), 
                    xytext = (df.loc[name]['log_Ms'], 
                              df.loc[name]['m_%Ms_ip']), 
                    arrowprops=dict(arrowstyle='->',color='k', linestyle='--', 
                                    linewidth=1.5))
        ## R+, M
        ax.annotate('', xy = (df3.loc[name]['log_Ms'], 
                              df3.loc[name]['m_%Ms_ip']), 
                    xytext = (df.loc[name]['log_Ms'], 
                              df.loc[name]['m_%Ms_ip']), 
                    arrowprops=dict(arrowstyle='->',color='r', linestyle='-.', 
                                    linewidth=1.5))
        ## R-, M
        ax.annotate('', xy = (df4.loc[name]['log_Ms'], 
                              df4.loc[name]['m_%Ms_ip']), 
                    xytext = (df.loc[name]['log_Ms'], 
                              df.loc[name]['m_%Ms_ip']), 
                    arrowprops=dict(arrowstyle='->',color='r', linestyle='--', 
                                    linewidth=1.5))
    
    ## linear regression - R, M
    lin_reg_x = np.linspace(9.,11.,21)
    xin = np.array(df['log_Ms'])
    yin = np.array(df['m_%Ms_ip'])
    lin_reg_y, b, m = lin_reg(xin,yin,lin_reg_x)
    b = '{:.2f}'.format(b)
    m = '{:.2f}'.format(m[0])
    ax.plot(lin_reg_x,lin_reg_y,c='tab:gray', linestyle='-', linewidth=4)
    
    ## linear regression - R, M+
    lin_reg_x = np.linspace(9.,11.,21)
    xin = np.array(df1['log_Ms'])
    yin = np.array(df1['m_%Ms_ip'])
    lin_reg_y, b, m = lin_reg(xin,yin,lin_reg_x)
    b = '{:.2f}'.format(b)
    m = '{:.2f}'.format(m[0])
    ax.plot(lin_reg_x,lin_reg_y,c='k', linestyle='-.', linewidth=1.5)
    
    ## linear regression - R, M-
    lin_reg_x = np.linspace(9.,11.,21)
    xin = np.array(df2['log_Ms'])
    yin = np.array(df2['m_%Ms_ip'])
    lin_reg_y, b, m = lin_reg(xin,yin,lin_reg_x)
    b = '{:.2f}'.format(b)
    m = '{:.2f}'.format(m[0])
    ax.plot(lin_reg_x,lin_reg_y,c='k', linestyle='--', linewidth=1.5)
    
    ## linear regression - R+, M
    lin_reg_x = np.linspace(9.,11.,21)
    xin = np.array(df3['log_Ms'])
    yin = np.array(df3['m_%Ms_ip'])
    lin_reg_y, b, m = lin_reg(xin,yin,lin_reg_x)
    b = '{:.2f}'.format(b)
    m = '{:.2f}'.format(m[0])
    ax.plot(lin_reg_x,lin_reg_y,c='r', linestyle='-.', linewidth=1.5)
    
    ## linear regression - R-, M
    lin_reg_x = np.linspace(9.,11.,21)
    xin = np.array(df4['log_Ms'])
    yin = np.array(df4['m_%Ms_ip'])
    lin_reg_y, b, m = lin_reg(xin,yin,lin_reg_x)
    b = '{:.2f}'.format(b)
    m = '{:.2f}'.format(m[0])
    ax.plot(lin_reg_x,lin_reg_y,c='r', linestyle='--', linewidth=1.5)
    
    ## axes settings
    ax.set_ylim(-0.5,15.5)
    ax.set_yticks(ax.get_yticks()[1:-1]) # Remove first and last ticks
    ax.set_xlim(8.8,11.2)
    ax.set_xticks(ax.get_xticks()[1:-1]) # Remove first and last ticks
    ax.set_xlabel(r'$\log(M_\star/\mathrm{M}_\odot)$', fontsize=18)
    ax.set_ylabel(r'Fractional $M_\star$ increase from $t_\mathrm{inf}$ to $t_\mathrm{peri}$', 
                  fontsize=18)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both',which='major',direction='in', bottom = True, 
                   top = True,left = True, right = True, length=10, 
                   labelsize=18)
    ax.tick_params(axis='both',which='minor',direction='in', bottom = True, 
                   top = True,left = True, right = True, length=5, 
                   labelsize=18)
    ax.grid(False)
    ax.set_facecolor('w')
    
    dia = mlines.Line2D([], [], color='tab:gray', marker='D', 
                        linestyle='None', markersize=8, 
                        label='Coma galaxies')
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
    ax.legend(handles=[dia,reg,rvir,ms,plus,minus],frameon=True, 
              framealpha=1.0, edgecolor='k', fontsize=18, loc=3, 
              bbox_to_anchor=(0.03,0.02), bbox_transform=ax.transAxes)
        
    out_dir = gen_out_dir(ssp)
    plt.savefig(out_dir+ssp+'_'+'f_Ms_inf_peri_uncert_plot.pdf',dpi=500)

###############################################################################
## Plotting Shift in results due to uncertainty in Rvir,Coma and M* of sats ##
###############################################################################

#ssps = ['miles','bc03','phr']

ssps = ['miles']

for ssp in ssps:
    plot_all(ssp)