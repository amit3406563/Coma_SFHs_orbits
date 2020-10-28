# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 05:39:30 2020

@author: amit
"""

import os
import shutil
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter as sf
from scipy import integrate
from scipy import interpolate
#import matplotlib.pyplot as plt

out_dir = './sfr_ssfr_tables/miles/'
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir)

def sav_gol(x,win,poly):
    return sf(x, window_length=win, polyorder=poly, mode='interp')

def filter_arr(name,df):
    unfilt_arr = np.array(df[name])
    filt_arr = sav_gol(unfilt_arr,9,3)
    return filt_arr

def ssfr_find(name):
    sfr = np.array(sfr_df[name])
    #sfr = filter_arr(name,sfr_df)
    ms = np.flip(np.abs(integrate.cumtrapz(np.flip(sfr),np.flip(age*10**9))))
    ssfr = np.divide(sfr[:-1],ms)
    ssfr = sav_gol(ssfr,9,3)
    return age[:-1], ssfr, ms

gmp_names = ['3254','3269', '3291', '3329', '3352', '3367', '3414', '3484',
             '3534', '3565', '3639', '3664']

log_Ms = np.loadtxt('logMs_coma.m')
Ms = 10**log_Ms

steckmap_sfr = pd.read_csv('steckmap_sfr_tables/miles/'+'steckmap_sfr.csv')

age = np.array(steckmap_sfr['Age_Gyr'])

## calibrating SFR
sfr_df = pd.DataFrame()
sfr_df['Age_Gyr'] = list(age)
for name, m in zip(gmp_names, Ms):
    ms = np.abs(integrate.trapz(np.flip(steckmap_sfr[name]),
                                   np.flip(age*10**9)))
    corr = m/ms
    sfr = list(np.array(steckmap_sfr[name]) * corr)
    sfr_df[name] = sfr

sfr_df.to_csv(out_dir+'corr_sfr.csv',index=False)

## computing SSFR
ssfr_df = pd.DataFrame()
ssfr_df['Age_Gyr'] = list(age)
for name in gmp_names:
    t, ssfr, ms = ssfr_find(name)
    f = interpolate.interp1d(t,ssfr,fill_value='extrapolate')
    ssfr = f(age)
    ssfr_df[name] = list(ssfr)

ssfr_df.to_csv(out_dir+'ssfr.csv',index=False)

# fig, ax = plt.subplots(figsize=(10,7))
# for name in gmp_names:
#     ax.plot(age,ssfr_df[name])