# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:13:20 2020

@author: amit
"""


import os
import shutil
import numpy as np
import pandas as pd

## function to create output directories for saving tables: sat&+- and int&+-
def out_dirs(ext):
    dir_sat = './cuts_df/cuts_sat/'+ext+'/'
    if os.path.exists(dir_sat):
        shutil.rmtree(dir_sat)
    os.makedirs(dir_sat)

    dir_int = './cuts_df/cuts_int/'+ext+'/'
    if os.path.exists(dir_int):
        shutil.rmtree(dir_int)
    os.makedirs(dir_int)
    
    return dir_sat, dir_int

## function to perform cut on a particular parameter
def perform_cut(d,p,v,f,subd,log_bool):
    # d: input data frame, p: parameter name, v = parameter value, f: factor
    a = v - f # lower limit in log
    b = v + f # upper limit in log
    if log_bool == True:
        cut_arr = np.array(d[p].loc[(np.log10(d[p]) > a) & (np.log10(d[p]) < b)])
    else:
        cut_arr = np.array(d[p].loc[(d[p] > a) & (d[p] < b)])
    cut_d = subd.loc[subd[p].isin(cut_arr)]
    return cut_d

## function to perform Processing cuts for all satellites and Interlopers
def processing(f_ext, sat_names, Mcoma_vir, log_Mh, R, V):
    ## creating directories for output files:
    dir_sat, dir_int = out_dirs(f_ext)
    
    ###########################################################################
    ## satellite cuts ##
    ###########################################################################
    ## 1. cut on Mhost
    cut_Mhost = perform_cut(data_sat, 'Mhost', np.log10(Mcoma_vir), np.log10(3), 
                            data_sat,True)
    # print('Shape after Mhost cut: '+str(cut_Mhost.shape)+'\n')

    ## 2. cut on Msat
    # for name, lMh in zip(sat_names,log_Mh):
    #     temp = vars()['cut_Msat_'+name] = perform_cut(data_sat, 'Msat', lMh, np.log10(3), 
    #                                            cut_Mhost,True)
    #     print('Shape after Msat cut for GMP '+name+': '+str(temp.shape))
    # print('\n')

    ## 3. cut on Mmax
    for name, lMh in zip(sat_names,log_Mh):
        temp = vars()['cut_Mmax_'+name] = perform_cut(data_sat, 'm_max', 
                                                      lMh, np.log10(3), 
                                                      cut_Mhost,True)
    #     print('Shape after Mmax cut for GMP '+name+': '+str(temp.shape))
    # print('\n')  
  
    ## 4. cut on R   
    for name, r in zip(sat_names,R):
        temp = vars()['cut_R_'+name] = perform_cut(data_sat, 'R', r, 0.05, 
                                                   vars()['cut_Mmax_'+name],False)    
    #     print('Shape after R cut for GMP '+name+': '+str(temp.shape))
    # print('\n')

    ## 5. cut on V
    for name, v in zip(sat_names,V):
        temp = vars()['cut_V_'+name] = perform_cut(data_sat, 'V', v, 0.05, 
                                                   vars()['cut_R_'+name],False)
        print('Shape after V cut for GMP '+name+': '+str(temp.shape))
    print('\n')

    ## 6. saving data
    for name in sat_names:
        vars()['cut_V_'+name].to_csv(dir_sat+'cut_sat_'+name+'.csv',
                                     index=False)
    
    ###########################################################################
    ## interloper cuts ##
    ###########################################################################
    ## 1. cut on Mhost -- Interlopers
    cut_Mhost_int = perform_cut(data_int, 'Mhost', np.log10(Mcoma_vir), np.log10(3), 
                                data_int,True)
    # print('Shape after Mhost cut - Interloper: '+str(cut_Mhost_int.shape)+'\n')

    ## 2. cut on Msat -- Interlopers
    for name, lMh in zip(sat_names,log_Mh):
        temp = vars()['cut_Msat_int_'+name] = perform_cut(data_int, 'Msat', lMh, 
                                                          np.log10(3), 
                                                          cut_Mhost_int,True)
    #     print('Shape after Msat cut - Interloper for GMP '+name+': '+str(temp.shape))
    # print('\n')

    ## 4. cut on R -- Interlopers
    for name, r in zip(sat_names,R):
        temp = vars()['cut_R_int_'+name] = perform_cut(data_int, 'R', r, 0.05, 
                                                       vars()['cut_Msat_int_'+name],False)    
    #     print('Shape after R cut - Interloper for GMP '+name+': '+str(temp.shape))
    # print('\n')  

    ## 5. cut on V -- Interlopers
    for name, v in zip(sat_names,V):
        temp = vars()['cut_V_int_'+name] = perform_cut(data_int, 'V', v, 0.05, 
                                                       vars()['cut_R_int_'+name],False)
        print('Shape after V cut - Interloper for GMP '+name+': '+str(temp.shape))

    ## 6. saving data -- Interlopers
    for name in sat_names:
        vars()['cut_V_int_'+name].to_csv(dir_int+'cut_int_'+name+'.csv',
                                         index=False)


##############################################################################
        ## Performing Cuts ##
##############################################################################
## Reading Satellites and Interlopers table
data_sat = pd.read_csv('orbit_sat.csv')
data_int = pd.read_csv('orbit_int.csv')


## get Coma and sats data from tables

sat_names = ['3254','3269', '3291', '3329', '3352', '3367', '3414', '3484',
              '3534', '3565', '3639', '3664']

# Coma Mvir and Mvir +-
coma_df = pd.read_csv('Coma_props.csv')
Mcoma_vir = coma_df['Mvir'][0]
Mcoma_virp = coma_df['Mvir+'][0]
Mcoma_virm = coma_df['Mvir-'][0]

# Coma sats: Msat and Msat +-
coma_sats_df = pd.read_csv('smhm_behroozi2010.csv')
log_Mh = coma_sats_df['logMh']
Mh_sat = 10**log_Mh
log_Mhp = coma_sats_df['logMh+']
Mh_satp = 10**log_Mhp
log_Mhm = coma_sats_df['logMh-']
Mh_satm = 10**log_Mhm

# Coma sats: PPS (R and V with their +-)
coma_pps_df = pd.read_csv('Coma_pps.csv')
R = coma_pps_df['R']
Rp = coma_pps_df['R+']
Rm = coma_pps_df['R-']
V = coma_pps_df['V']
Vp = coma_pps_df['V+']
Vm = coma_pps_df['V-']

## Cut for Nominal values:
processing('Rvir_Ms', sat_names, Mcoma_vir, log_Mh, R, V)
#print('Cut for Rvir_Ms condition processed...\n')

## Cut for uncertainty boundary values
# Cut for Rvir_M+
processing('Rvir_Ms+', sat_names, Mcoma_vir, log_Mhp, R, V)
#print('Cut for Rvir_Ms+ condition processed...\n')
# Cut for Rvir_M-
processing('Rvir_Ms-', sat_names, Mcoma_vir, log_Mhm, R, V)
#print('Cut for Rvir_Ms- condition processed...\n')

# Cut for Rvir+_Ms
processing('Rvir+_Ms', sat_names, Mcoma_virp, log_Mh, Rp, Vp)
#print('Cut for Rvir+_Ms condition processed...\n')
# Cut for Rvir+_Ms+
processing('Rvir+_Ms+', sat_names, Mcoma_virp, log_Mhp, Rp, Vp)
#print('Cut for Rvir+_Ms+ condition processed...\n')
# Cut for Rvir+_Ms-
processing('Rvir+_Ms-', sat_names, Mcoma_virp, log_Mhm, Rp, Vp)
#print('Cut for Rvir+_Ms- condition processed...\n')

# Cut for Rvir-_Ms
processing('Rvir-_Ms', sat_names, Mcoma_virm, log_Mh, Rm, Vm)
#print('Cut for Rvir-_Ms condition processed...\n')
# Cut for Rvir-_Ms+
processing('Rvir-_Ms+', sat_names, Mcoma_virm, log_Mhp, Rm, Vm)
#print('Cut for Rvir-_Ms+ condition processed...\n')
# Cut for Rvir-_Ms-
processing('Rvir-_Ms-', sat_names, Mcoma_virm, log_Mhm, Rm, Vm)
#print('Cut for Rvir-_Ms- condition processed...\n')