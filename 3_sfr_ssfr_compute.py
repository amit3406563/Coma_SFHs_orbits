# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 03:54:49 2020

@author: amit
"""


import os
import shutil
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter as sf
from scipy.interpolate import interp1d
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages

gmp_names = ['3254','3269', '3291', '3329', '3352', '3367', '3414', '3484',
             '3534', '3565', '3639', '3664']

log_Ms = np.loadtxt('logMs_coma.m')
Ms = 10**log_Ms

def gen_out_dir(ssp):
    out_dir = './sfr_ssfr_tables/'+ssp+'/'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    return out_dir

def sav_gol(x,win,poly):
    return sf(x, window_length=win, polyorder=poly, mode='interp')

def sfr_interp(corr_sfr_df):
    age = np.array(corr_sfr_df['Age_Gyr'])
    age_extended_bins = np.linspace(age[0],age[-1],100)
    sfr_interp = pd.DataFrame()
    sfr_interp['Age_Gyr'] = age_extended_bins
    for name in gmp_names:
        f = interp1d(age,np.array(corr_sfr_df[name]),fill_value='extrapolate')
        sfr_interp[name] = sav_gol(f(age_extended_bins),9,3)
    return sfr_interp

def sfr_find(steckmap_sfr,name,M):
    age = np.array(steckmap_sfr['Age_Gyr'])
    a = age*10**9
    s = np.array(steckmap_sfr[name])
    s = sav_gol(s,9,3)
    mi = np.array([])
    ti = np.array([])
    for i in range(len(a)-1):
        mi = np.append(mi, 0.5*(a[i+1]-a[i])*(s[i+1]+s[i]))
        ti = np.append(ti, 0.5*(a[i+1]+a[i]))
    f = interp1d(ti,mi,fill_value='extrapolate')
    ms = f(a)
    m = np.sum(ms)
    corr = M/m
    sfr = s*corr
    sfr = sav_gol(sfr,9,3)
    return sfr

def ssfr_find(corr_sfr_df, name):
    age = np.array(corr_sfr_df['Age_Gyr'])
    a = age*10**9
    s = np.array(corr_sfr_df[name])
    s = sav_gol(s,9,3)
    mi = np.array([])
    ti = np.array([])
    for i in range(len(a)-1):
        mi = np.append(mi, 0.5*(a[i+1]-a[i])*(s[i+1]+s[i]))
        ti = np.append(ti, 0.5*(a[i+1]+a[i]))
    f = interp1d(ti,mi,fill_value='extrapolate')
    ms = f(a)
    msc = np.flip(np.cumsum(np.flip(ms)))
    ssfr = np.true_divide(s,msc)
    ssfr = sav_gol(ssfr,9,3)
    # log_ssfr = np.log10(ssfr)
    # log_ssfr = sav_gol(log_ssfr,9,3)
    return ssfr

def remove_negative_ssfr_vals(ssfr_df):
    for name in gmp_names:
        x = np.array(ssfr_df[name])
        idx = np.where(x < 0.)
        idx = idx[0]
        for j in idx:
            k = j
            while np.any(idx == j):
                j = j+1
            x[k] = (x[k-1] + x[j])*0.5
        ssfr_df[name] = x
    return ssfr_df

def correct_and_interpolate_sfr_ssfr(ssp):
    steckmap_sfr = pd.read_csv('steckmap_sfr_tables/'+ssp+'/'+'steckmap_sfr.csv')
    age = np.array(steckmap_sfr['Age_Gyr'])
    sfr_corr_df = pd.DataFrame()
    sfr_corr_df['Age_Gyr'] = age

    for name, M in zip(gmp_names, Ms):
        sfr_corr_df[name] = sfr_find(steckmap_sfr,name,M)

    sfr_df = sfr_interp(sfr_corr_df)
    
    ssfr_df = pd.DataFrame()
    ssfr_df['Age_Gyr'] = np.array(sfr_df['Age_Gyr'])
    for name in gmp_names:
        ssfr_df[name] = ssfr_find(sfr_df,name)

    ssfr_df = remove_negative_ssfr_vals(ssfr_df)
    
    return sfr_df, ssfr_df

def save_sfr_ssfr_tables(ssp):
    sfr_df, ssfr_df = correct_and_interpolate_sfr_ssfr(ssp)
    out_dir = gen_out_dir(ssp)
    sfr_df.to_csv(out_dir+'corr_sfr.csv',index=False)
    ssfr_df.to_csv(out_dir+'ssfr.csv',index=False)
    
ssps = ['miles','bc03','phr']
#ssps = ['miles']

for ssp in ssps:
    save_sfr_ssfr_tables(ssp)