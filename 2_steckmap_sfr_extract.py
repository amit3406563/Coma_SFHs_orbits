# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 11:48:42 2020

@author: amit
"""


import os
import shutil
import numpy as np
import pandas as pd

out_dir = './steckmap_sfr_tables/phr/'
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir)

def list_files(path,key):
    unsorted_datafiles = [os.path.join(root, name)
            for root, dirs, files in os.walk(path)
            for name in files
            if name.endswith(key)]
    datafiles = sorted(unsorted_datafiles)
    return datafiles

def sat_name(f_name):
    temp = f_name.split('/')
    temp1 = temp[3].split('_')
    name = temp1[0]
    return name

sfr_files = list_files('./out_files_steckmap/phr/', '_SFR.dat')

sfr_df = pd.DataFrame()
sfr_df['Age_Gyr'] = (10**np.array(pd.read_csv(sfr_files[0],delimiter=' ')
                                  ['log[Age(Myr)]']))/1000
for file in sfr_files:
    name = sat_name(file)
    sfr_steckmap = pd.read_csv(file,delimiter=' ')[name+'_SFR']
    sfr_df = pd.concat([sfr_df,sfr_steckmap],axis=1,ignore_index=False)
    sfr_df.rename(columns={name+'_SFR':name}, inplace=True)

sfr_df.to_csv(out_dir+'steckmap_sfr.csv',index=False)