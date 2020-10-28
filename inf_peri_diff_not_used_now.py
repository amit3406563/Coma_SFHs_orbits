# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 02:31:25 2020

@author: amit
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from matplotlib.cm import ScalarMappable, viridis as cmap
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_pdf import PdfPages

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


# reading data files    
inf_df = pd.read_csv('./inf_peri_files/Rvir_Ms_inf_time.csv')
peri_df = pd.read_csv('./inf_peri_files/Rvir_Ms_peri_time.csv')
sfr_df = pd.read_csv('./sfr_ssfr_tables/miles/corr_sfr.csv')
log_Ms = np.array(pd.read_csv('smhm_behroozi2010.csv')['logMs'])

# extracting GMP names
sat_names = inf_df.columns.tolist()

# plotting
norm = Normalize(vmin=9, vmax=11)
smap = ScalarMappable(cmap=cmap, norm=norm)
smap.set_array([])

fig, ax = plt.subplots(figsize=(10,7))
for name, log_Mi in zip(sat_names, log_Ms):
    if name == '3329':
        continue
    else:
        med_ip,errp_ip,errm_ip,med_ms,errp_ms,errm_ms = stats_diff(sfr_df,inf_df,
                                                                    peri_df,name)
        c = cmap(norm(log_Mi))
        ax.errorbar(med_ms, med_ip, xerr=[[errp_ms],[errm_ms]],
                     yerr=[[errp_ip],[errm_ip]],elinewidth=1, capsize=5, 
                     ecolor=c, marker='D', mec=c, mfc=c,markersize=8)
ax.set_xlabel(r'$\log_{\rm 10}({\Delta M_\star/{\rm M}_\odot}_{{\rm inf}-{\rm peri}})$',
              fontsize=18)
ax.set_ylabel(r'${\Delta t}_{{\rm inf}-{\rm peri}}$ [Gyr]',fontsize=18)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both',which='both',direction='in', bottom = True, 
                       top = True,left = True, right = True, labelsize=18)
dia = mlines.Line2D([], [], color='grey', marker='D', linestyle='None',
                              markersize=8, label=r'$\tilde{x}\pm\sigma$')
ax.legend(handles=[dia],frameon=False, framealpha=1.0,loc=4,fontsize=18) 
cbar = fig.colorbar(smap, ticks=[9., 9.5, 10.0, 10.5, 11.0])
cbar.set_label(r'$\log_{10}(M_\star/{\rm M}_\odot)$',fontsize=18)
cbar.ax.tick_params(axis='y', direction='in',labelsize=18)
ax.grid(False)
ax.set_facecolor('w')
plt.savefig('median(Ms_tinf-Ms_tperi)_vs_median(tinf-tperi).pdf',dpi=500)        
        
        