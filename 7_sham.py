# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 03:35:04 2020

@author: amit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sat_names = ['3254','3269', '3291', '3329', '3352', '3367', '3414', '3484',
              '3534', '3565', '3639', '3664']

log_Ms= np.loadtxt('logMs_coma.m') # loading stellar mass from text file
Ms = 10**log_Ms

# uncertainty in stellar mass 25%
# log_Msp = log_Ms + np.log10(1.25)
# Msp = 10**log_Msp
# #np.savetxt('logMsp_coma.m',log_Msp)
# log_Msm = log_Ms + np.log10(0.75)
# Msm = 10**log_Msm
#np.savetxt('logMsm_coma.m',log_Msm)

# declaring dataframe to save columns
df = pd.DataFrame()
df['GMP'] = sat_names
# df['logMs-'] = log_Msm
df['logMs'] = log_Ms
# df['logMs+'] = log_Msp

# print('Satellite stellar masses (also in  log10) in solar mass: \n')
# for name, m, lm in zip(sat_names, Ms, log_Ms):
#     print('GMP'+name+' | '+'{:.2e}'.format(m)+' | '+'{:.2f}'.format(lm))
# print('\n')

# stellar masses for satellites provided by Prof. Dr. Scott Trager

# Relation for SM-HM conversion from Behroozi+ 2010 [Eqn-21, 22]
# parameter values for eqn-21 taken from Table-2
# this is direct conversion of satellite masses of all 12 galaxies with 
# correct virial definition and unit in solar mass

# scale factor
def find_a(z):
    return 1/(1+z)

def find_Mh(z, log_Ms):
    a = find_a(z)

    ## redshift dependent expressions
    # constants 
    log_M10 = 12.35
    log_M1a = 0.28

    log_Ms00 = 10.72
    log_Ms0a = 0.55

    beta0 = 0.44
    betaa = 0.18

    delta0 = 0.57
    deltaa = 0.17

    gamma0 = 1.56
    gammaa = 2.51

    # expressions
    log_M1 = log_M10 + log_M1a * (a-1)
    log_Ms0 = log_Ms00 + log_Ms0a * (a-1)
    beta = beta0 + betaa * (a-1)
    delta = delta0 + deltaa * (a-1)
    gamma = gamma0 + gammaa * (a-1)

    Ms0 = 10**log_Ms0

    k = Ms/Ms0
    log_Mh = log_M1 + beta*log_Ms - beta*log_Ms0 + (k**delta)/(1+k**(-gamma)) - 1/2
    
    #np.savetxt('logMh_coma1.m',log_Mh)
    return log_Mh

# uncertainty in halo mass
# k = Msp/Ms0
# log_Mhp = log_M1 + beta*log_Ms - beta*log_Ms0 + (k**delta)/(1+k**(-gamma)) - 1/2
# Mh_satp = 10**log_Mhp
#np.savetxt('logMhp_coma1.m',log_Mhp)

# k = Msm/Ms0
# log_Mhm = log_M1 + beta*log_Ms - beta*log_Ms0 + (k**delta)/(1+k**(-gamma)) - 1/2
# Mh_satm = 10**log_Mhm
#np.savetxt('logMhm_coma1.m',log_Mhm)


z = 0
log_Mh = find_Mh(z, log_Ms)
Mh_sat = 10**log_Mh
# uncertainty in halo mass
log_Mhp = log_Mh + np.log10(3)
Mhp = 10**log_Mhp
#np.savetxt('logMsp_coma.m',log_Msp)
log_Mhm = log_Mh - np.log10(3)
Mhm = 10**log_Mhm
#np.savetxt('logMsm_coma.m',log_Msm)

df['logMh-'] = log_Mhm
df['logMh'] = log_Mh
df['logMh+'] = log_Mhp

# print('Satellite halo masses (also in  log10) in solar mass: \n')
# for name, m, lm in zip(sat_names, Mh_sat, log_Mh):
#     print('GMP'+name+' | '+'{:.2e}'.format(m)+' | '+'{:.2f}'.format(lm))
# print('\n')

df.to_csv('smhm_behroozi2010.csv',index=False)

## plotting at different z
z_arr = np.linspace(0.,1.,6)
fig, ax = plt.subplots(figsize=(10,7))
for z in z_arr:
    log_Ms_sorted = np.sort(log_Ms)
    log_Mh_sorted = find_Mh(z, log_Ms_sorted)
    ax.plot(log_Ms_sorted[:-1], 
            np.log10(np.divide(10**log_Mh_sorted[:-1], 10**log_Ms_sorted[:-1])))