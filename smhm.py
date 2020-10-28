# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 11:32:00 2020

@author: amit
"""


import numpy as np
import pandas as pd
from scipy.interpolate import interp1d as ip
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sat_names = np.array(['3254','3269', '3291', '3329', '3352', '3367', '3414',
                      '3484', '3534', '3565', '3639', '3664'])

## Finding scale factor 'a'
# z: redshift; a: scale factor
def finda(z):
    return 1 / (1 + z)


## defining parameters as function of redshift
def gen_params(z):
    a = finda(z)
    # characteristic mass in solar masses
    M0 = 12.013
    Ma = 4.597
    Mlna = 4.47
    Mz = -0.737
    log10_M1 = M0 + Ma*(a-1) - Mlna*np.log(a) + Mz*z

    # characteristic SFR (epsilon - e)
    e0 = -1.466
    ea = 1.852
    elna = 1.439
    ez = -0.227
    e = e0 + ea*(a-1) - elna*np.log(a) + ez*z

    # faint-end slope (alpha - a)
    a0 = 1.965
    aa = -2.137
    alna = -1.607
    az = 0.161
    a = a0 + aa*(a-1) - alna*np.log(a) + az*z

    # steep-end slope (beta - b)
    b0 = 0.564
    ba = -0.835
    bz = -0.478
    b = b0 + ba*(a-1) + bz*z

    # width of Gaussian SFR efficiency boost (delta - d)
    d0 = 0.411
    d = d0
    
    # strength of Gaussian SFR efficiency boost (gamma - g)
    g0 = -0.937
    ga = -2.810
    gz = -0.983
    log10_g = g0 + ga*(a-1) + gz*z
    g = 10**log10_g

    return log10_M1, e, a, b, d, g

## SM-HM relation from Behroozi 2019
# log10(Ms/M1) = e - log10(10^-ax + 10^-bx) + g * exp(-0.5 * (x/d)^2)
# x = log10(Mpeak/M1)
def smhm(red,logms):
    log10_M1, e, a, b, d, g = gen_params(red)
    lmh = np.linspace(10.,14.,10000)
    x = lmh - log10_M1
    y = e - np.log10(10**(-a*x) + 10**(-b*x)) + g * np.exp(-0.5 * (x/d)**2)
    lms = y + log10_M1
    f = ip(10**lms,10**lmh,fill_value='extrapolate')
    return np.log10(f(10**logms))


def format_float_arr(arr):
    new_arr = np.array([])
    for ar in arr:
        new_arr = np.append(new_arr,'{:.2f}'.format(ar))
    return new_arr
        

## SM-HM computation
log10Ms = np.loadtxt('logMs_coma.m') 

#z = np.loadtxt('redshift.z')


# Case-1: z = 0; M*
z = 0
log10Mh_1 = np.array([])
for i in range(len(sat_names)):
    log10Mh_1 = np.append(log10Mh_1,smhm(z,log10Ms[i]))

# Case-2: z = 0; M*/2
z = 0
log10Mh_2 = np.array([])
for i in range(len(sat_names)):
    log10Mh_2 = np.append(log10Mh_2,smhm(z,log10Ms[i]/2)) 

# Case-3: z = 1; M*
z = 1
log10Mh_3 = np.array([])
for i in range(len(sat_names)):
    log10Mh_3 = np.append(log10Mh_3,smhm(z,log10Ms[i]))


### Method1: Behroozi 2010
# Relation for SM-HM conversion from Behroozi+ 2010 [Eqn-21]
# parameter values for eqn-21 taken from Table-2
# this is direct conversion of satellite masses of all 12 galaxies with 
# correct virial definition and unit in solar mass
log_M1 = 12.35
log_Ms0 = 10.72
beta = 0.44
delta = 0.57
gamma = 1.56
Ms = 10**log10Ms
Ms0 = 10**log_Ms0
k = Ms/Ms0

log_Mh1 = log_M1 + beta*log10Ms - beta*log_Ms0 + (k**delta)/(1+k**(-gamma)) - 1/2
Mh1 = 10**log_Mh1

    
    
## Tabulating the results
df = pd.DataFrame()
df['GMP'] = sat_names
df['log10Ms'] = log10Ms
df['log10Mh (Behroozi2010)'] = format_float_arr(log_Mh1)
df['log10Mh (Behroozi2019)'] = format_float_arr(log10Mh_1)
df['Case2_z0Ms/2 (Behroozi2019)'] = format_float_arr(log10Mh_2)
df['Case3_z1Ms (Behroozi2019)'] = format_float_arr(log10Mh_3)
np.savetxt('logMh_coma.m',log10Mh_1)
np.savetxt('logMh_coma1.m',log_Mh1)
df.to_csv('SMHM_table.csv',index=False)

fig, ax = plt.subplots(figsize=(7,4))
#ax.axis('tight')
ax.axis('off')
tab = ax.table(cellText=df.values,colLabels=df.columns,loc='center')
tab.auto_set_font_size(False)
tab.set_fontsize(6)
tab.auto_set_column_width(col=list(range(len(df.columns))))
pp = PdfPages("smhm_table.pdf")
pp.savefig(fig, bbox_inches='tight')
pp.close()