# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 17:44:33 2021

@author: amit
"""

import os
import pandas as pd
import numpy as np

gmp_names = ['3254','3269', '3291', '3329', '3352', '3367', '3414', '3484',
             '3534', '3565', '3639', '3664']

def list_files(path,key):
    unsorted_datafiles = [os.path.join(root, name)
            for root, dirs, files in os.walk(path)
            for name in files
            if name.endswith(key)]
    datafiles = sorted(unsorted_datafiles)
    return datafiles

file_names = list_files('./cuts_df/cuts_sat/Rvir_Ms/', '.csv')
# sats = []
# size_cut = []
# gtr_1e14 = []
# gtr_1e135 = []
# gtr_1e13 = []

dfs = []
for fname, satname in zip(file_names, gmp_names):
    if fname.split('_')[-1].split('.')[0] != '3329':
        df = pd.read_csv(fname)
        dfs.append(df)
        # print('\nGMP '+satname)
        # print('Number of matches after cut: ', df.shape[0])
        # print('Matches for Mmax > 1e+14: ', len(df[df['m_max'] > 1e+14]))
        # print('Matches for Mmax > 1e+13.5: ', len(df[df['m_max'] > 10**13.5]))
        # print('Matches for Mmax > 1e+13: ', len(df[df['m_max'] > 1e+13]))
        # sats.append(satname)
        # size_cut.append(df.shape[0])
        # gtr_1e14.append(len(df[df['m_max'] > 1e+14]))
        # gtr_1e135.append(len(df[df['m_max'] > 10**13.5]))
        # gtr_1e13.append(len(df[df['m_max'] > 1e+13]))
        
# df = pd.DataFrame()
# df['GMP'] = sats
# df['#Matches_after_cut'] = size_cut
# df['#Matches_Mmax>1e14'] = gtr_1e14
# df['#Matches_Mmax>1e13_5'] = gtr_1e135
# df['#Matches_Mmax>1e13'] = gtr_1e13

# df.to_csv('counts.csv', index=False)

dfss = pd.concat(dfs)

print(len(dfss['Mhost'].unique()))

print(len(dfss[dfss['m_max'] > 1e+14]))
print(len(dfss[dfss['m_max'] > 10**13.5]))
print(len(dfss[dfss['m_max'] > 1e+13]))



    