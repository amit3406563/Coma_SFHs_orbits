# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 03:35:04 2020

@author: amit
"""

import numpy as np
import pandas as pd

sat_names = ['3254','3269', '3291', '3329', '3352', '3367', '3414', '3484',
              '3534', '3565', '3639', '3664']

log_Ms= np.loadtxt('logMs_coma.m') # loading stellar mass from text file
Ms = 10**log_Ms

# uncertainty in stellar mass 25%
log_Msp = log_Ms + np.log10(1.25)
Msp = 10**log_Msp
#np.savetxt('logMsp_coma.m',log_Msp)
log_Msm = log_Ms + np.log10(0.75)
Msm = 10**log_Msm
#np.savetxt('logMsm_coma.m',log_Msm)

# declaring dataframe to save columns
df = pd.DataFrame()
df['GMP'] = sat_names
df['logMs-'] = log_Msm
df['logMs'] = log_Ms
df['logMs+'] = log_Msp

# print('Satellite stellar masses (also in  log10) in solar mass: \n')
# for name, m, lm in zip(sat_names, Ms, log_Ms):
#     print('GMP'+name+' | '+'{:.2e}'.format(m)+' | '+'{:.2f}'.format(lm))
# print('\n')

# stellar masses for satellites provided by Prof. Dr. Scott Trager

# Relation for SM-HM conversion from Behroozi+ 2010 [Eqn-21]
# parameter values for eqn-21 taken from Table-2
# this is direct conversion of satellite masses of all 12 galaxies with 
# correct virial definition and unit in solar mass
log_M1 = 12.35
log_Ms0 = 10.72
beta = 0.44
delta = 0.57
gamma = 1.56
Ms0 = 10**log_Ms0

k = Ms/Ms0
log_Mh = log_M1 + beta*log_Ms - beta*log_Ms0 + (k**delta)/(1+k**(-gamma)) - 1/2
Mh_sat = 10**log_Mh
#np.savetxt('logMh_coma1.m',log_Mh)

# uncertainty in halo mass
k = Msp/Ms0
log_Mhp = log_M1 + beta*log_Ms - beta*log_Ms0 + (k**delta)/(1+k**(-gamma)) - 1/2
Mh_satp = 10**log_Mhp
#np.savetxt('logMhp_coma1.m',log_Mhp)

k = Msm/Ms0
log_Mhm = log_M1 + beta*log_Ms - beta*log_Ms0 + (k**delta)/(1+k**(-gamma)) - 1/2
Mh_satm = 10**log_Mhm
#np.savetxt('logMhm_coma1.m',log_Mhm)

df['logMh-'] = log_Mhm
df['logMh'] = log_Mh
df['logMh+'] = log_Mhp

# print('Satellite halo masses (also in  log10) in solar mass: \n')
# for name, m, lm in zip(sat_names, Mh_sat, log_Mh):
#     print('GMP'+name+' | '+'{:.2e}'.format(m)+' | '+'{:.2f}'.format(lm))
# print('\n')

df.to_csv('smhm_behroozi2010.csv',index=False)