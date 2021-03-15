# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 12:17:45 2021

@author: amit
"""


import numpy as np
import pandas as pd
import astropy.units as u
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.lines as mlines
from astropy.cosmology import Planck13 # Planck 13 cosmology parameters
from astropy.constants import M_sun # value of M_sun as constant in kg
Msun = u.def_unit('Msun', M_sun) # Msun as unit
from astropy.constants import G

## dynamical timescale with redshift computation
z = np.linspace(0.,1.6,17)

#H0 = Planck13.H0.value # 67.77 km/s
#Hz = Planck13.H(z).value # H at Coma redshift -> 68.53 km/s
#Om0 = Planck13.Om(0) # 0.307
Omz = Planck13.Om(z) 
#rhoc_0 = Planck13.critical_density(0).to(u.Msun / u.Mpc ** 3).value
rhoc_z = Planck13.critical_density(z).to(u.kg / u.m**3)
rho_z = 360 * Omz * rhoc_z
#rho_z = 360 * rhoc_z

# coma = pd.read_csv('Coma_props.csv')

# coma_Rvir = u.Mpc.to(u.km, coma['Rvir'][0]) # Rvir,coma in km (conv. from pc)
# coma_sig3d = coma['sig3d'][0] # in km/s
# coma_sig1d = coma['sig1d'][0] # in km/s

# coma_tcross = u.s.to(u.Gyr, coma_Rvir/coma_sig1d)

t_dyn = (1 / np.sqrt(G * rho_z)).to(u.Gyr).value # in Gyr

#plt.plot(z,t_dyn) 

## depletion timescale with redshift computation
# taconni 18 expession: t_depl = (1+z)^-0.6 * del_MS^-0.44
# for stellar mass range 9-11.8: del_MS ~ 10^-1.3 - 10^2.2
t_depl_low_mass = ((1+z)**(-0.6)) * ((10**(-1.3))**(-0.44))
t_depl_high_mass = ((1+z)**(-0.6)) * ((10**(2.2))**(-0.44))  

# # from Fig-5 Taconni 2018
# log1pz = np.linspace(0.,0.7,8)
# logtdepl = -0.8 * log1pz + 0.09

# #plt.plot(log1pz, logtdepl)

# z_arr = 10**log1pz - 1 
# tdepl_arr = 10**logtdepl
# f = interp1d(z_arr, tdepl_arr, fill_value='extrapolate')
# t_depl = f(z)

## quenching timescales from other studies at different redshifts
# Taranu 14
z_t14 = 0
tq_t14 = 4
tqe_t14 = 0.5

# Wetzel 13
z_w13 = 0
tq_w13 = 4.4
tqe_w13 = 0.4

# Balogh 16
z_b16 = 1
tq_b16 = 1.5
tqe_b16 = 0.5

# Haines 15
z_h15 = 0.2
tq_h15 = 3.7
tqe_h15 = 0.5

# Muzzin 14
z_m14 = 1
tq_m14 = 1
tqe_m14 = 0.25

# Foltz 18
z1_f18 = 1
tq1_f18 = 1.3
tqe1_f18 = 0.5

z2_f18 = 1.5
tq2_f18 = 1.1
tqe2_f18 = 0.5

## Plotting
c = 'k'
fig, ax = plt.subplots(figsize=(8,7))
ax.plot(z, t_dyn,'k--') # t_dyn
ax.plot(z, t_depl_low_mass,'k:') # t_depl
#ax.plot(z, t_depl_high_mass,'k-.') # t_depl
# Foltz 18
ax.errorbar(z1_f18, tq1_f18, yerr=tqe1_f18, elinewidth=1, capsize=5, 
            ecolor=c, marker='o', mec=c, mfc=c, markersize=10)
ax.errorbar(z2_f18, tq2_f18, yerr=tqe2_f18, elinewidth=1, capsize=5, 
            ecolor=c, marker='o', mec=c, mfc=c, markersize=10)
# Muzzin 14
ax.errorbar(z_m14, tq_m14, yerr=tqe_m14, elinewidth=1, capsize=5, 
            ecolor=c, marker='D', mec=c, mfc=c, markersize=8)
# Haines 15
ax.errorbar(z_h15, tq_h15, yerr=tqe_h15, elinewidth=1, capsize=5, 
            ecolor=c, marker='s', mec=c, mfc=c, markersize=9)
# Balogh 16
ax.errorbar(z_b16, tq_b16, yerr=tqe_b16, elinewidth=1, capsize=5, 
            ecolor=c, marker='*', mec=c, mfc=c, markersize=10)
# Wetzel 13
ax.errorbar(z_w13, tq_w13, yerr=tqe_w13, elinewidth=1, capsize=5, 
            ecolor=c, marker='P', mec=c, mfc=c, markersize=10)
# Taranu 14
ax.errorbar(z_t14, tq_t14, yerr=tqe_t14, elinewidth=1, capsize=5, 
            ecolor=c, marker='X', mec=c, mfc=c, markersize=10)
# plot settings
ax.set_xlabel(r'$z$', fontsize=18)
ax.set_ylabel(r'$t_\mathrm{q}\,\mathrm{[Gyr]}$', fontsize=18)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both',which='major',direction='in', bottom = True, 
                top = True, left = True, right = True, length = 10, 
                labelsize=18)
ax.tick_params(axis='both',which='minor',direction='in', bottom = True, 
                top = True, left = True, right = True, length = 5, 
                labelsize=18)
dyn = mlines.Line2D([], [], color=c, marker='None', linestyle='--', 
                      label=r'$t_\mathrm{dyn}$')
dep = mlines.Line2D([], [], color=c, marker='None', linestyle=':', 
                      label=r'$t_\mathrm{depl}$')
circ = mlines.Line2D([], [], color=c, marker='o', linestyle='None',
                              markersize=10, label='Foltz+18')
diam = mlines.Line2D([], [], color=c, marker='D', linestyle='None',
                              markersize=8, label='Muzzin+14')
sqar = mlines.Line2D([], [], color=c, marker='s', linestyle='None',
                              markersize=9, label='Haines+15')
star = mlines.Line2D([], [], color=c, marker='*', linestyle='None',
                              markersize=10, label='Balogh+16')
plus = mlines.Line2D([], [], color=c, marker='P', linestyle='None',
                              markersize=10, label='Wetzel+13')
cros = mlines.Line2D([], [], color=c, marker='X', linestyle='None',
                              markersize=10, label='Taranu+14')
h = [plus, cros, diam, dyn, sqar, star, circ, dep]
ax.legend(handles=h, ncol=2, loc=1, fontsize=15, frameon=True, framealpha=1.0, 
          edgecolor='k', bbox_to_anchor=(0.98,0.98), 
          bbox_transform=ax.transAxes)
#plt.savefig('tq.pdf',dpi=500)