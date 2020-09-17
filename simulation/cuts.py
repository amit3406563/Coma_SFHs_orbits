import os
import shutil
import numpy as np
import pandas as pd

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

log_Mh = np.loadtxt('logMh_coma.m')
Mh_sat = 10**log_Mh

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
R = np.loadtxt('R.vir') 
   
for name, r in zip(sat_names,R):
    temp = vars()['cut_R_'+name] = perform_cut(data_sat, 'R', r, 0.05, 
                                            vars()['cut_Mmax_'+name],False)    
    print('Shape after R cut for GMP '+name+': '+str(temp.shape))
print('\n')


## 5. cut on V
V = np.loadtxt('V.sig3d')

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
