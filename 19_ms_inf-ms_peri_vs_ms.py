# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 04:16:37 2020

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
def gen_out_dir(ssp):
    out_dir = './ms_inf-ms_peri_vs_ms/'+ssp+'/'
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
    return inf_peri

# computes stellar mass for corresponding inf and peri time and saves as new
# column in same data frame of non-nan inf peri
def ms_at_inf_peri(sfr,inf,peri,name):
    age = np.array(sfr['Age_Gyr'])*10**9
    s = np.array(sfr[name])
    age_bins = np.array([])
    for i in range(len(age)-1):
        age_bins = np.append(age_bins,(age[i+1]+age[i])*0.5)
    m = np.array([])
    for i in range(len(age)-1):
        m = np.append(m,0.5*(age[i+1]-age[i])*(s[i+1]+s[i]))
    f = interp1d(age_bins,m,fill_value='extrapolate')
    inf_peri = non_nan_inf_peri(inf,peri,name)
    inf_peri['inf_ms'] = f(np.array(inf_peri['inf'])*10**9)
    inf_peri['peri_ms'] = f(np.array(inf_peri['peri'])*10**9)
    return inf_peri

# computes difference between inf and peri & stellar masses at corresponding 
# inf and peri, then updates the data frame with new columns of diff values
# return difference arrays for inf & peri and corresponding stellar masses
def inf_peri_and_ms_diff(sfr,inf,peri,name):
    inf_peri = ms_at_inf_peri(sfr,inf,peri,name)
    inf_peri['diff_ip'] = inf_peri['inf'] - inf_peri['peri']
    inf_peri['diff_ms'] = inf_peri['inf_ms'] - inf_peri['peri_ms']
    # remove index where - ve value is seen in diff_ms
    diff_df = inf_peri.drop(index=inf_peri.index[inf_peri['diff_ms'] < 0.].tolist())
    diff_ip = np.array(diff_df['diff_ip'])
    diff_ms = np.array(diff_df['diff_ms'])
    return diff_ip, diff_ms

# computes median with 1 sigma limits
def stats_comp(diff):
    sort_diff = np.sort(diff)
    p = 1. * np.arange(len(diff)) / (len(diff) - 1)
    f = interp1d(p, sort_diff, fill_value='extrapolate')
    med = f(0.5)
    errp = f(0.84) - f(0.5)
    errm = f(0.5) - f(0.16)
    return med, errp, errm

# for both difference in inf & peri and
# corresponding stellar masses
def stats_diff(sfr,inf,peri,name):
    diff_ip, diff_ms = inf_peri_and_ms_diff(sfr,inf,peri,name)
    med_ip, errp_ip, errm_ip = stats_comp(diff_ip)
    med_ms, errp_ms, errm_ms = stats_comp(np.log10(diff_ms))
    return med_ip, errp_ip, errm_ip, med_ms, errp_ms, errm_ms

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
    # reading data files    
    inf_df = pd.read_csv('./inf_peri_files/Rvir_Ms_inf_time.csv')
    peri_df = pd.read_csv('./inf_peri_files/Rvir_Ms_peri_time.csv')
    sfr_df = pd.read_csv('./sfr_ssfr_tables/'+ssp+'/corr_sfr.csv')
    log_Ms = np.array(pd.read_csv('smhm_behroozi2010.csv')['logMs'])
    
    # extracting GMP names
    sat_names = inf_df.columns.tolist()
    
    # plotting
    # norm = Normalize(vmin=9, vmax=11)
    # smap = ScalarMappable(cmap=cmap, norm=norm)
    # smap.set_array([])
    # arrays for saving data
    ip = np.array([])
    epip = np.array([])
    emip = np.array([])
    ms = np.array([])
    epms = np.array([])
    emms = np.array([])
    lMs = np.array([])
    elMs = np.array([])
    n = np.array([])
    fig, ax = plt.subplots(figsize=(10,7))
    for name, log_Mi in zip(sat_names, log_Ms):
        if name == '3329':
            continue
        else:
            med_ip,errp_ip,errm_ip,med_ms,errp_ms,errm_ms = stats_diff(sfr_df,inf_df,
                                                                        peri_df,name)
            # c = cmap(norm(log_Mi))
            xerr = np.abs(np.log10(.25))
            ax.errorbar(log_Mi, med_ms, xerr=xerr,
                         yerr=[[errp_ms],[errm_ms]],elinewidth=1, capsize=5, 
                         ecolor='b', marker='D', mec='b', mfc='b',markersize=8)
            ip = np.append(ip,med_ip)
            epip = np.append(epip,errp_ip)
            emip = np.append(emip,errm_ip)
            ms = np.append(ms,med_ms)
            epms = np.append(epms,errp_ms)
            emms = np.append(emms,errm_ms)
            lMs = np.append(lMs,log_Mi)
            elMs = np.append(elMs, xerr)
            n = np.append(n,name)
    lin_reg_x = np.linspace(9.,11.,21)
    lin_reg_y, b, m = lin_reg(lMs,ms,lin_reg_x)
    b = '{:.2f}'.format(b)
    m = '{:.2f}'.format(m[0])
    ax.plot(lin_reg_x,lin_reg_y,c='r',linestyle='--',linewidth=4)
    ax.set_ylim(7.,11.)
    ax.set_yticks(ax.get_yticks()[1:-1]) # Remove first and last ticks
    ax.set_xlim(8.5,11.5)
    ax.set_xticks(ax.get_xticks()[1:-1]) # Remove first and last ticks
    ax.set_xlabel(r'$\log_{\rm 10}M_\star/{\rm M}_\odot$',
                  fontsize=18)
    ax.set_ylabel(r'$\log_{\rm 10}({\Delta M_\star/{\rm M}_\odot}_{{\rm inf}-{\rm peri}})$',
                  fontsize=18)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both',which='both',direction='in', bottom = True, 
                           top = True,left = True, right = True, labelsize=18)
    dia = mlines.Line2D([], [], color='b', marker='D', linestyle='None',
                                  markersize=8, label='stellar mass formed between inf and peri')
    vline = mlines.Line2D([], [], color='b', marker='|', linestyle='None',
                              markersize=20, markeredgewidth=2,
                              label=r'$\sigma_{\rm stellar \: mass \: formed \: between \: inf \: and \: peri}$')
    hline = mlines.Line2D([], [], color='b', marker='_', linestyle='None',
                              markersize=20, markeredgewidth=2,
                              label=r'$\pm25\%{\rm \: of \: stellar \: mass}$')
    line = mlines.Line2D([], [], color='r', marker='None', linestyle='--',
                              markersize=8,linewidth=4,
                              label='f(x) = '+m+'x + '+b)
    ax.legend(handles=[dia,vline,hline,line],frameon=False, framealpha=1.0,loc=2,
              fontsize=14) 
    # cbar = fig.colorbar(smap, ticks=[9., 9.5, 10.0, 10.5, 11.0])
    # cbar.set_label(r'$\log_{10}(M_\star/{\rm M}_\odot)$',fontsize=18)
    # cbar.ax.tick_params(axis='y', direction='in',labelsize=18)
    ax.grid(False)
    ax.set_facecolor('w')
    out_dir = gen_out_dir(ssp)
    plt.savefig(out_dir+ssp+'_'+'ms_inf-ms_peri_vs_ms.pdf',dpi=500)        
            
    df = pd.DataFrame()
    df['GMP'] = n
    df['tinf-tperi'] = ip
    df['et+'] = epip
    df['et-'] = emip
    df['Msinf-Msperi'] = ms
    df['em+'] = epms
    df['em-'] = emms
    df['logMs'] = lMs
    df['elogMs'] = elMs
    df.to_csv(out_dir+ssp+'_'+'ms_inf-ms_peri_vs_ms.csv',index=False)

###############################################################################
## Plotting M* formed between infall and peri vs. M* ##
###############################################################################

ssps = ['miles','bc03','phr']

for ssp in ssps:
    plot_all(ssp)        