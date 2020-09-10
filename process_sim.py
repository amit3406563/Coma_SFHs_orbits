# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:13:20 2020

@author: amit
"""


import os
import shutil
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP7

## creating directory for saving output files
dir_sat = './cuts_df/cuts_sat/'
if os.path.exists(dir_sat):
    shutil.rmtree(dir_sat)
os.makedirs(dir_sat)

dir_int = './cuts_df/cuts_int/'
if os.path.exists(dir_int):
    shutil.rmtree(dir_int)
os.makedirs(dir_int)

## Reading Satellites and Interlopers table
data_sat = pd.read_csv('orbit_sat.csv')
data_int = pd.read_csv('orbit_int.csv')

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

def coord_icrs(ra,dec):
    return SkyCoord(ra, dec, frame = 'icrs')    


##############################################################################
## satellite cuts
##############################################################################

## 1. cut on Mhost
Mcoma = 1.4*10**15
print('Mass of Coma cluster in solar mass: '+'{:.2e}'.format(Mcoma)+'\n')
# actual mass of Coma cluster: Mcoma = 1.4*10**15 h70^-1 M_sol 
Mcoma_vir = 0.73**3 * 360*0.27/200 * Mcoma 
print('Virial mass of Coma cluster in solar mass: '+'{:.2e}'.format(Mcoma_vir)
      +'\n')
# converting actual mass of Coma cluster in terms of  Virial mass

cut_Mhost = perform_cut(data_sat, 'Mhost', np.log10(Mcoma_vir), np.log10(3), 
                        data_sat,True)
print('Shape after Mhost cut: '+str(cut_Mhost.shape)+'\n')


## 2. cut on Msat
sat_names = ['3254','3269', '3291', '3329', '3352', '3367', '3414', '3484',
              '3534', '3565', '3639', '3664']
log_Ms= np.loadtxt('logMs_coma.m') # loading stellar mass from text file
Ms = 10**log_Ms
print('Satellite stellar masses (also in  log10) in solar mass: \n')
for name, m, lm in zip(sat_names, Ms, log_Ms):
    print('GMP'+name+' | '+'{:.2e}'.format(m)+' | '+'{:.2f}'.format(lm))
print('\n')
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
np.savetxt('logMh_coma.m',log_Mh)
print('Satellite halo masses (also in  log10) in solar mass: \n')
for name, m, lm in zip(sat_names, Mh_sat, log_Mh):
    print('GMP'+name+' | '+'{:.2e}'.format(m)+' | '+'{:.2f}'.format(lm))
print('\n')

# for name, lMh in zip(sat_names,log_Mh):
#     temp = vars()['cut_Msat_'+name] = perform_cut(data_sat, 'Msat', lMh, np.log10(3), 
#                                            cut_Mhost,True)
#     print('Shape after Msat cut for GMP '+name+': '+str(temp.shape))
# print('\n')


## 3. cut on Mmax
for name, lMh in zip(sat_names,log_Mh):
    temp = vars()['cut_Mmax_'+name] = perform_cut(data_sat, 'm_max', lMh, np.log10(3), 
                                            cut_Mhost,True)
    print('Shape after Mmax cut for GMP '+name+': '+str(temp.shape))
print('\n')  
  

## 4. cut on R
coord_coma = pd.read_csv('Coma_coords.csv')
coord_coma.index = sat_names

coma_center_coord = ['12h59m46.7s', '+27d57m00s']
coord_coma_centre = coord_icrs(coma_center_coord[0],coma_center_coord[1])

print('Coma center coordinates: \n'+'RA: '+coma_center_coord[0]+'\n'+
      'DEC: '+coma_center_coord[1]+'\n')

ang_sep_rad = np.array([])
ang_sep_deg = np.array([])
print('Satellites coordinates (RA & DEC): \n')
for name in sat_names:
    ra = coord_coma.loc[name]['RA']
    dec = coord_coma.loc[name]['DEC']
    print('GMP'+name+' | '+ra+' | '+dec)
    temp1 = vars()['theta_rad_'+name] = coord_coma_centre.separation(
        coord_icrs(ra,dec)).rad
    ang_sep_rad = np.append(ang_sep_rad,temp1)
    temp2 = vars()['theta_deg_'+name] = temp1 * (180/np.pi)
    ang_sep_deg = np.append(ang_sep_deg,temp2)
print('\n')    

print('Angular separation of satellites from Coma center (rad. & deg.): \n')
for name, ar, ad in zip(sat_names, ang_sep_rad, ang_sep_deg):
    print('GMP'+name+' | '+'{:.5f}'.format(ar)+' | '+'{:.3f}'.format(ad))
print('\n')

z_coma = 0.0231 # from NASA/IPAC EXTRAGALACTIC DATABASE
print('Redshift of Coma cluster: '+str(z_coma)+'\n')
z = np.loadtxt('redshift.z') # reading redshift
print('Redshift of satellites: \n')
for name, red in zip(sat_names,z):
    print('GMP'+name+' | '+str(red))
print('\n')

dA = WMAP7.angular_diameter_distance(z_coma)
print('Angular diameter distance of Coma cluster in Mpc: '+
      '{:.2f}'.format(dA.value)+'\n') 
# angular diameter distance of Coma in Mpc
# computing R [R/r_vir_coma] for each satellite galaxy
r_vir_coma = 2.9/0.73 # in Mpc from Lokas and Mamon 2003
print('Virial radius of Coma cluster in Mpc: '+'{:.2f}'.format(r_vir_coma)+'\n')
R = ang_sep_rad * dA.value / r_vir_coma
np.savetxt('ang_sep_rad.rad',ang_sep_rad)
np.savetxt('ang_sep_deg.deg',ang_sep_rad*(180/np.pi))
np.savetxt('R.vir',R)
    
for name, r in zip(sat_names,R):
    temp = vars()['cut_R_'+name] = perform_cut(data_sat, 'R', r, 0.05, 
                                            vars()['cut_Mmax_'+name],False)    
    print('Shape after R cut for GMP '+name+': '+str(temp.shape))
print('\n')


## 5. cut on V

sig_1D_coma = 1154 # in km/s from Jorgensen et al 2018 Table-1
sig_3D_coma = sig_1D_coma * np.sqrt(3)
print('sigma_1D_Coma in km/s: '+str(sig_1D_coma)+'\n')
print('sigma_3D_Coma in km/s: '+'{:.2f}'.format(sig_3D_coma)+'\n')
c = 299792458/1000 # speedof light in km/s
V = (c * np.abs(z - z_coma)) / ((1 + z_coma) * sig_3D_coma)
np.savetxt('V.sig3d',V)

print('R/r_vir & V/sigma_3D for all satellites: \n')
for name, r, v in zip(sat_names, R, V):
    print('GMP'+name+' | '+'{:.3f}'.format(r)+' | '+'{:.2f}'.format(v))
print('\n')

for name, v in zip(sat_names,V):
    temp = vars()['cut_V_'+name] = perform_cut(data_sat, 'V', v, 0.05, 
                                            vars()['cut_R_'+name],False)
    print('Shape after V cut for GMP '+name+': '+str(temp.shape))
print('\n')


## 6. saving data
for name in sat_names:
    vars()['cut_V_'+name].to_csv(dir_sat+'cut_sat_'+name+'_mmax.csv',
                                  index=False)
    

# ##############################################################################
# ## interloper cuts
# ##############################################################################

## 1. cut on Mhost -- Interlopers
cut_Mhost_int = perform_cut(data_int, 'Mhost', np.log10(Mcoma_vir), np.log10(3), 
                        data_int,True)
print('Shape after Mhost cut - Interloper: '+str(cut_Mhost_int.shape)+'\n')


## 2. cut on Msat -- Interlopers
for name, lMh in zip(sat_names,log_Mh):
    temp = vars()['cut_Msat_int_'+name] = perform_cut(data_int, 'Msat', lMh, 
                                                      np.log10(3), 
                                                      cut_Mhost_int,True)
    print('Shape after Msat cut - Interloper for GMP '+name+': '+str(temp.shape))
print('\n')


## 4. cut on R -- Interlopers
for name, r in zip(sat_names,R):
    temp = vars()['cut_R_int_'+name] = perform_cut(data_int, 'R', r, 0.05, 
                                            vars()['cut_Msat_int_'+name],False)    
    print('Shape after R cut - Interloper for GMP '+name+': '+str(temp.shape))
print('\n')  


## 5. cut on V -- Interlopers
for name, v in zip(sat_names,V):
    temp = vars()['cut_V_int_'+name] = perform_cut(data_int, 'V', v, 0.05, 
                                            vars()['cut_R_int_'+name],False)
    print('Shape after V cut - Interloper for GMP '+name+': '+str(temp.shape))
print('\n')


## 6. saving data -- Interlopers
for name in sat_names:
    vars()['cut_V_int_'+name].to_csv(dir_int+'cut_int_'+name+'_mmax.csv',
                                  index=False)