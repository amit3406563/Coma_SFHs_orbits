# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 21:03:12 2020

@author: amit
"""


import numpy as np
import pandas as pd
from astropy.cosmology import Planck13 # Planck 13 cosmology parameters
import astropy.units as u
from astropy.constants import M_sun # value of M_sun as constant in kg
Msun = u.def_unit('Msun', M_sun) # Msun as unit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def format_float_arr(arr):
    new_arr = np.array([])
    for ar in arr:
        new_arr = np.append(new_arr,'{:.2f}'.format(ar))
    return new_arr

def format_exp_arr(arr):
    new_arr = np.array([])
    for ar in arr:
        new_arr = np.append(new_arr,'{:.2e}'.format(ar))
    return new_arr

z = np.loadtxt('z.txt')
coma_z = z[3]

H0 = Planck13.H0.value # 67.77 km/s
Hz = Planck13.H(coma_z).value # H at Coma redshift -> 68.53 km/s
Om0 = Planck13.Om(0) # 0.307
Omz = Planck13.Om(coma_z) # Om at Coma redshift -> 0.322
rhoc_0 = Planck13.critical_density(0).to(u.Msun / u.Mpc ** 3).value
# critical density at z = 0 -> 1.27e+11 Msun/Mpc^3
rhoc_z = Planck13.critical_density(coma_z).to(u.Msun / u.Mpc ** 3).value
# critical density at Coma redshift -> 1.30e+11 Msun/Mpc^3
# Bryan and Norman 1998
x = Omz - 1 # x -> -0.678
Delta_c_z = 18*(np.pi**2) + 82*x - 39*(x**2) # 104.11 in units of critical density
Delta_vir_z = Delta_c_z/Omz # 323.15 in units of background density
y = Om0 - 1 # x -> -0.693
Delta_c_0 = 18*(np.pi**2) + 82*y - 39*(y**2) # 102.12 in units of critical density
Delta_vir_0 = Delta_c_0/Omz # 316.74 in units of background density


Rvir_70 = 2.9 # virial radius in Mpc of Coma cluster from Lokas and Mamon 2003
# They assume H0 = 70 km/s, so we need to convert it to Planck 13 cosmology
# if virial is defined in 200c instead of 360b, this conversion is used:
# r200c/r360b = 0.73
Rvir = Rvir_70/((H0/100)/(70/100)) # 3.0 Mpc as per Planck 13 cosmology 

# uncertainty in Rvir
uRvir = 0.3 # in Mpc
Rvirp = Rvir + uRvir
Rvirm = Rvir - uRvir

def comp_props(R):
    M_vir = 4*np.pi/3 * Delta_vir_z * Omz * rhoc_z * R**3
    
    sig1d = (0.0165/np.sqrt(3)) * (M_vir)**(1/3) * (Delta_vir_z/Delta_vir_0)**(1/6) * (1+coma_z)**(1/2)
    sig3d = sig1d * np.sqrt(3)
    
    return M_vir, sig1d, sig3d

Mvir,sig1d,sig3d = comp_props(Rvir)
Mvirp,sig1dp,sig3dp = comp_props(Rvirp)
Mvirm,sig1dm,sig3dm = comp_props(Rvirm)

data = {'Rvir':[Rvir], 'Rvir+':[Rvirp], 'Rvir-':[Rvirm], 
        'Mvir':[Mvir], 'Mvir+':[Mvirp], 'Mvir-':[Mvirm], 
        'sig1d':[sig1d], 'sig1d+':[sig1dp], 'sig1d-':[sig1dm], 
        'sig3d':[sig3d], 'sig3d+':[sig3dp], 'sig3d-':[sig3dm]}

df = pd.DataFrame(data, columns = ['Rvir', 'Rvir+', 'Rvir-', 
                                   'Mvir', 'Mvir+', 'Mvir-', 
                                   'sig1d', 'sig1d+', 'sig1d-', 
                                   'sig3d', 'sig3d+', 'sig3d-'])

df.to_csv('Coma_props.csv', index = False)


df1 = pd.DataFrame()
mass_cols = ['Mvir', 'Mvir+', 'Mvir-']
for col in df.columns.tolist():
    if col in mass_cols:
        df1[col] = format_exp_arr(np.array(df[col]))
    else:
        df1[col] = format_float_arr(np.array(df[col]))

fig, ax = plt.subplots(figsize=(3,1))
#ax.axis('tight')
ax.axis('off')
tab = ax.table(cellText=df1.values,colLabels=df1.columns,loc='center')
tab.auto_set_font_size(False)
tab.set_fontsize(6)
tab.auto_set_column_width(col=list(range(len(df.columns))))
pp = PdfPages("Coma_props.pdf")
pp.savefig(fig, bbox_inches='tight')
pp.close()