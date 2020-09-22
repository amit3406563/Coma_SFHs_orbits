# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 00:35:40 2020

@author: Amit
"""

import h5py
import pandas as pd

# simulation output HDF5 file name
filename = 'pdf_mmax.hdf5'

# read HDF5 file
f = h5py.File(filename,'r')

# list keys: 'satellites', 'interlopers', and 'config' 
f_keys = list(f.keys())

# read data for all groups individually within each key
# 'interlopers': 'Mhost', 'Msat', 'R', and 'V'
# 'satellites': 'Mhost', 'Msat', 'R', 'V', 'm_infall', 'm_max', 'small_r'/'r',
#               'r_min', 't_infall', 't_peri', 'small_v'/'v', 'v_max'
# invoke empty dataframes for interlopers and satellites tables
orbit_df_int = pd.DataFrame() 
orbit_df_sat = pd.DataFrame()
for key in f_keys:
    k_groups = list(f[key])
    for group in k_groups:
        dset = f[key][group][()]
        if key == 'interlopers': # if interloper key
            orbit_df_int[group] = dset # add data column with same group name
        elif key == 'satellites': # if satellite key
            orbit_df_sat[group] = dset # add data column with same group name
        print(key+'_'+group)
f.close()

orbit_df_int.to_csv('orbit_int.csv',index=False)
print('orbit_int.csv Saved')
orbit_df_sat.to_csv('orbit_sat.csv',index=False)
print('orbit_sat.csv Saved')