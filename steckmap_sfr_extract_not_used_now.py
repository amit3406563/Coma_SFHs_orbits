# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 03:57:53 2020

@author: amit
"""


import os
import numpy as np
import pandas as pd

## GMP IDs of satellites
gmp_names = ['3254', '3269', '3291', '3484', '3534', '3565', '3664', '3639', 
             '3414', '3352', '3367', '3329']

## Generic IDs of satellites
gen_names = ['d127', 'd128', 'd154', 'd157', 'd158', 'gmp3565', 'n4864', 
             'n4867', 'n4871', 'n4872', 'n4873', 'n4874']

def list_files(path,key):
    unsorted_datafiles = [os.path.join(root, name)
            for root, dirs, files in os.walk(path)
            for name in files
            if name.endswith(key)]
    datafiles = sorted(unsorted_datafiles)
    return datafiles

sfr_files = list_files('./out_files/', '_SFR.dat')

sfr_df = pd.DataFrame()
sfr_df['Age_Gyr'] = (10**np.array(pd.read_csv(sfr_files[0],delimiter=' ')
                                  ['log[Age(Myr)]']))/1000
for file, gen_name, gmp_name in zip(sfr_files, gen_names, gmp_names):
    sfr_steckmap = pd.read_csv(file,delimiter=' ')[gen_name+'_SFR']
    sfr_df = pd.concat([sfr_df,sfr_steckmap],axis=1,ignore_index=False)
    sfr_df.rename(columns={gen_name+'_SFR':gmp_name}, inplace=True)

cols = sfr_df.columns.tolist()
cols = sorted(cols)
cols = cols[-1:] + cols[:-1]
sfr_df = sfr_df[cols]

sfr_df.to_csv('steckmap_sfr.csv',index=False)