# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 21:37:02 2020

@author: amit
"""

import warnings
warnings.filterwarnings("ignore")

import os
import shutil
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression as lr
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import AutoMinorLocator

# output directory
def gen_out_dir(ssp,ext):
    out_dir = './f_ms_inf_peri/'+ssp+'/'+ext+'/'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    return out_dir

# combines infall and peri comuns of each satellite in a single df
def combine_inf_peri(inf,peri,name):
    inf_peri = pd.concat([inf[name],peri[name]],join='outer',axis=1)
    return inf_peri

# finds out the index where NaN value is seen in infall and peri columns 
def index_with_nan(df):
    rows_with_nan = []
    for index, row in df.iterrows():
        is_nan_series = row.isnull()
        if is_nan_series.any():
            rows_with_nan.append(index)
    return rows_with_nan

# makes a fresh df with only non-NaN values of inf and peri
def non_nan_inf_peri(inf,peri,name):
    inf_peri_df = combine_inf_peri(inf,peri,name)
    nan_idx = index_with_nan(inf_peri_df)
    inf_peri = inf_peri_df.drop(nan_idx,axis=0)
    inf_peri.columns = ['inf','peri']
    inf_peri = inf_peri*10**9
    return inf_peri

# calculates diff in fraction of Ms formed at any given t (in yr)
def f_ms_t(t,sfr,logMs, name):
    tend = 13.6 * 10**9 # end of time array 13.6 Gyr
    delt = 0.1 * 10**9 # delta t of time array 0.1 Gyr
    t = np.arange(t,tend,delt) # array from give time to end time with del t width
    age = sfr['Age_Gyr']*10**9 # age array as read from sfr dataframe
    s = sfr[name] # sfr array as read from sfr dataframe
    f = interp1d(age, s, fill_value='extrapolate')
    si = f(t) # sfr values at ti
    msi = np.array([]) # invoke empty array to store Ms formed in each time bin
    for i in range(len(t)-1):
        msi = np.append(msi, 0.5*delt*(si[i]+si[i+1]))
    ms = np.sum(msi) # add the Ms formed in each bin to get total Ms formed at t
    fms = ms / 10**logMs.loc[int(name)]['logMs'] # fraction of Ms formed at t
    return fms

def diff_pms(inf, peri, sfr, logMs, name):
    inf_peri = non_nan_inf_peri(inf,peri,name)
    inf_peri['fms_inf'] = inf_peri['inf'].apply(f_ms_t, args=[sfr,logMs,name])
    inf_peri['fms_peri'] = inf_peri['peri'].apply(f_ms_t, 
                                                  args=[sfr,logMs,name])
    inf_peri['diff_fms'] = inf_peri['fms_peri'] - inf_peri['fms_inf']
    inf_peri['diff_pms'] = inf_peri['diff_fms'] * 100
    return inf_peri
    
# computes median with 1 sigma limits
def stats_comp(diff):
    sort_diff = np.sort(diff)
    p = 1. * np.arange(len(diff)) / (len(diff) - 1)
    f = interp1d(p, sort_diff, fill_value='extrapolate')
    med = f(0.5)
    errp = f(0.84) - f(0.5)
    errm = f(0.5) - f(0.16)
    return med, errp, errm    

# computes median and 1 sigma limits of % of Ms formed between inf and peri
def pms_inf_peri(inf, peri, sfr, logMs, name):
    inf_peri = diff_pms(inf, peri, sfr, logMs, name)
    diff = inf_peri['diff_pms']
    med, errp, errm = stats_comp(diff)
    return med, errp, errm

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
def plot_all(ssp,ext):
    # reading data files    
    inf_df = pd.read_csv('./inf_peri_files/'+ext+'_inf_time.csv')
    peri_df = pd.read_csv('./inf_peri_files/'+ext+'_peri_time.csv')
    sfr_df = pd.read_csv('./sfr_ssfr_tables/'+ssp+'/corr_sfr.csv')
    log_Ms = pd.read_csv('smhm_behroozi2010.csv')[['GMP','logMs']]
    log_Ms.set_index('GMP',inplace=True)
    
    # extracting GMP names
    sat_names = inf_df.columns.tolist()

    # invoking empty arrays to save data in table
    m_pms_ip = np.array([])
    errp_pms_ip = np.array([])
    errm_pms_ip = np.array([])
    n = np.array([])
    lMs = np.array([])
    
    fig, ax = plt.subplots(figsize=(10,7))
    for name in sat_names:
        if name == '3329':
            continue
        else:
            med, errp, errm = pms_inf_peri(inf_df,peri_df,sfr_df,log_Ms,name)
            m_pms_ip = np.append(m_pms_ip, med)
            errp_pms_ip = np.append(errp_pms_ip, errp)
            errm_pms_ip = np.append(errm_pms_ip, errm)
            n = np.append(n, name)
            lMs = np.append(lMs, log_Ms.loc[int(name)]['logMs'])
            ax.errorbar(log_Ms.loc[int(name)]['logMs'], med,
                         yerr=[[errm],[errp]],
                         elinewidth=1, capsize=5, ecolor='k', marker='D', 
                         mec='k', mfc='k',markersize=8)
    # linear regression
    lin_reg_x = np.linspace(9.,11.,21)
    lin_reg_y, b, m = lin_reg(lMs,m_pms_ip,lin_reg_x)
    b = '{:.2f}'.format(b)
    m = '{:.2f}'.format(m[0])
    ax.plot(lin_reg_x,lin_reg_y,c='r',linestyle='-',linewidth=4)
    # plot settings
    ax.set_ylim(-0.5,30.5)
    ax.set_yticks(ax.get_yticks()[1:-1]) # Remove first and last ticks
    ax.set_xlim(8.8,11.2)
    ax.set_xticks(ax.get_xticks()[1:-1]) # Remove first and last ticks
    ax.set_xlabel(r'$\log_{\rm 10}M_\star/{\rm M}_\odot$',
                      fontsize=18)
    ax.set_ylabel(r'Fractional $M_\star$ increase from $t_\mathrm{inf}$ to $t_\mathrm{peri}$',
                      fontsize=18)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both',which='both',direction='in', bottom = True, 
                   top = True,left = True, right = True, labelsize=18)
    # dia = mlines.Line2D([], [], color='b', marker='D', linestyle='None',
    #                     markersize=8, label=r'% of $M_\star$ formed b/w inf-peri')
    # vline = mlines.Line2D([], [], color='b', marker='|', linestyle='None',
    #                       markersize=20, markeredgewidth=2,
    #                       label=r'$\sigma_{\rm \: \% \: of \: M_\star \: formed \: b/w \: inf-peri}$')
    line = mlines.Line2D([], [], color='r', marker='None', linestyle='-',
                         markersize=8,linewidth=4, 
                         label=r'$\frac{\Delta\,M_{\star,\mathrm{inf-peri}}}{\mathrm{M}_\odot}\,\%=\,$'+m+r'$\,\log_{10}\frac{M_\star}{\mathrm{M}_\odot}+\,$'+b)
    # ax.legend(handles=[dia,vline,line],frameon=False, framealpha=1.0,loc=2,
    #           fontsize=14)
    ax.legend(handles=[line],fontsize=14, loc=2, bbox_to_anchor=(0.02,0.98), 
              bbox_transform=ax.transAxes)
    ax.grid(False)
    ax.set_facecolor('w')
    out_dir = gen_out_dir(ssp,ext)
    #plt.savefig(out_dir+ssp+'_'+'f_ms_inf_peri.pdf',dpi=500)
    # saving data as table
    df = pd.DataFrame([])
    df['GMP'] = n
    df['log_Ms'] = lMs 
    df['m_%Ms_ip'] = m_pms_ip
    df['e+'] = errp_pms_ip
    df['e-'] = errm_pms_ip
    df.to_csv(out_dir+ssp+'_'+'f_ms_inf_peri.csv',index=False)
        
###############################################################################
## Plotting M*% formed between infall and peri vs. M* ##
###############################################################################

ssps = ['miles','bc03','phr']
#ssps = ['miles']
uncert_ext = ['Rvir_Ms','Rvir_Ms+','Rvir_Ms-','Rvir+_Ms','Rvir-_Ms']
#uncert_ext = ['Rvir_Ms']
for ssp in ssps:
    for ext in uncert_ext:
        plot_all(ssp,ext)