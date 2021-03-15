# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 03:21:16 2020

@author: amit
"""

import os
import shutil
import pandas as pd
from astropy.cosmology import WMAP7 as cosmo

out_dir = './inf_peri_files/'
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir)

gmp_names = ['3254','3269', '3291', '3329', '3352', '3367', '3414', '3484',
             '3534', '3565', '3639', '3664']

def list_files(path,key):
    unsorted_datafiles = [os.path.join(root, name)
            for root, dirs, files in os.walk(path)
            for name in files
            if name.endswith(key)]
    datafiles = sorted(unsorted_datafiles)
    return datafiles

def atotdf(a): # gives age in Gyrs
    r = 1/a - 1
    t = cosmo.age(r).value
    return t

def read_time(file_list,col,sats):
    time_df = pd.DataFrame()
    nFiles = len(file_list)
    for i in range(nFiles):
        df = pd.read_csv(file_list[i],usecols=[col])
        df = df.dropna().apply(atotdf)
        time_df = pd.concat([time_df,df],axis=1,ignore_index=False)
    time_df.columns = sats
    return time_df

def extract_inf_peri(ext):
    df_list = list_files("./cuts_df/cuts_sat/"+ext+"/",".csv")
    inf = read_time(df_list,'t_infall',gmp_names)
    peri = read_time(df_list,'t_peri',gmp_names)
    inf_corr = 13.7 - inf
    peri_corr = 13.7 - peri
    inf_corr.to_csv(out_dir+ext+'_'+'inf_time.csv',index=False)
    peri_corr.to_csv(out_dir+ext+'_'+'peri_time.csv',index=False)


###############################################################################
## Extracting Infall and Pericenter time in Gyrs ##
###############################################################################

# Rvir_Ms
extract_inf_peri('Rvir_Ms')    
# Rvir_Ms+
extract_inf_peri('Rvir_Ms+') 
# Rvir_Ms-
extract_inf_peri('Rvir_Ms-') 
# Rvir+_Ms
extract_inf_peri('Rvir+_Ms')
# Rvir+_Ms+
# extract_inf_peri('Rvir+_Ms+')
# # Rvir+_Ms-
# extract_inf_peri('Rvir+_Ms-')
# Rvir-_Ms
extract_inf_peri('Rvir-_Ms')
# Rvir-_Ms+
# extract_inf_peri('Rvir-_Ms+')
# # Rvir-_Ms-
# extract_inf_peri('Rvir-_Ms-')