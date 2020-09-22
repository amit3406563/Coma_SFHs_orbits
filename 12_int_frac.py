# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 03:31:54 2020

@author: amit
"""


import os
import numpy as np
import pandas as pd

gmp_names = ['3254','3269', '3291', '3329', '3352', '3367', '3414', '3484',
             '3534', '3565', '3639', '3664']

def list_files(path,key):
    unsorted_datafiles = [os.path.join(root, name)
            for root, dirs, files in os.walk(path)
            for name in files
            if name.endswith(key)]
    datafiles = sorted(unsorted_datafiles)
    return datafiles

sat_files = list_files('./cuts_df/cuts_sat/', '.csv')
int_files = list_files('./cuts_df/cuts_int/', '.csv')

int_frac = np.array([])
n_sat = np.array([])
n_int = np.array([])
for i in range(len(gmp_names)):
    sat_shape = pd.read_csv(sat_files[i]).shape[0]
    int_shape = pd.read_csv(int_files[i]).shape[0]
    n_sat = np.append(n_sat,sat_shape)
    n_int = np.append(n_int,int_shape)
    int_frac = np.append(int_frac, int_shape / (int_shape + sat_shape))

int_frac_df = pd.DataFrame()
int_frac_df['GMP'] = gmp_names
int_frac_df['n_sat'] = n_sat
int_frac_df['n_int'] = n_int
int_frac_df['int_frac'] = int_frac
int_frac_df['int_frac_percentage'] = int_frac * 100
int_frac_df.to_csv('int_frac.csv',index = False)