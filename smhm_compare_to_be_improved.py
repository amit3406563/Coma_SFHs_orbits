# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 14:59:29 2020

@author: amit
"""


import numpy as np
from scipy.interpolate import interp1d as ip
import matplotlib.pyplot as plt
#from matplotlib.ticker import AutoMinorLocator

log_Ms= np.linspace(8.,11.5,1000)

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
Ms = 10**log_Ms
Ms0 = 10**log_Ms0
k = Ms/Ms0

log_Mh1 = log_M1 + beta*log_Ms - beta*log_Ms0 + (k**delta)/(1+k**(-gamma)) - 1/2
Mh1 = 10**log_Mh1



### Method2: Behroozi 2019
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
    lmh = np.arange(10.5,15.,0.2)
    x = lmh - log10_M1
    y = e - np.log10(10**(-a*x) + 10**(-b*x)) + g * np.exp(-0.5 * (x/d)**2)
    lms = y + log10_M1
    f = ip(10**lms,10**lmh,fill_value='extrapolate')
    return np.log10(f(10**logms))

## SM-HM computation
z = 0
log10_M1, e, a, b, d, g = gen_params(z)
log_Mh2 = np.array([])
for lms in log_Ms:
    log_Mh2 = np.append(log_Mh2,smhm(z,lms))

Mh2 = 10**log_Mh2

### Plotting    
fig, ax = plt.subplots(figsize=(7,7))
ax.plot(Ms,Mh1,label='Behroozi 2010')
ax.plot(Ms,Mh2,label='Behroozi 2019')
ax.axvline(x=10**11.43,color='r',linestyle='dotted',label='GMP 3329')
ax.set_xscale('log')
ax.set_yscale('log')
#ax.set_xlim(11.,11.0)
#ax.set_ylabel(r'${\rm SM-HM \: ratio} \: (M_\star/M_{h})$',fontsize=18)
ax.set_xlabel(r'$M_\star/{\rm M}_\odot$',fontsize=18)
ax.set_ylabel(r'$M_{h}/{\rm M}_\odot$',fontsize=18)
ax.legend(fontsize=18)
# ax.xaxis.set_minor_locator(AutoMinorLocator())
# ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both',which='both',direction='in', bottom = True, 
                   top = True,left = True, right = True,labelsize=18)
ax.grid(False)
ax.set_facecolor('w')
plt.savefig('SMHM_compare.pdf',dpi=500)