# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 04:58:54 2021

@author: amit
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from astropy.cosmology import Planck13
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.cm import ScalarMappable, viridis as cmap
from matplotlib.colors import Normalize
from matplotlib.ticker import AutoMinorLocator

ageU = Planck13.age(0).value # age of the Universe as per Planck 13 cosmology

## functions
def cdf_sfr(sfr_bins, sfr):
    ms = np.array([])
    for i in range(len(sfr_bins)-1):
        mi = 0.5 * (sfr[i+1] + sfr[i]) * (sfr_bins[i+1] - sfr_bins[i])
        ms = np.append(ms, mi)
    new_sfr_bins = np.array([])
    for i in range(len(sfr_bins)-1):
        new_sfr_bins = np.append(new_sfr_bins, 
                                 0.5*(sfr_bins[i+1] + sfr_bins[i]))
    m_cumsum = np.cumsum(ms)
    min_m_cumsum = min(m_cumsum)
    minmax_m_cumsum = max(m_cumsum) - min(m_cumsum)
    m_cdf = 1 - ((m_cumsum - min_m_cumsum) / minmax_m_cumsum)
    return new_sfr_bins, m_cdf

def cdf_time(time_bins, time, int_frac):
    h, be = np.histogram(time, bins=time_bins, density=True)
    del_be = time_bins[1]-time_bins[0]
    time_cdf = np.cumsum(h)*del_be
    new_time_bins = np.array([])
    for i in range(len(time_bins)-1):
        new_time_bins = np.append(new_time_bins, 
                                  0.5*(time_bins[i+1]+time_bins[i]))
    time_cdf = 1 - time_cdf
    time_cdf = time_cdf * (1-int_frac)
    return new_time_bins, time_cdf
    
def error_bars(time_bins,time_cdf,sfr_bins,sfr_cdf):
    f = interp1d(time_cdf, time_bins, fill_value='extrapolate')
    g = interp1d(sfr_bins, sfr_cdf, fill_value='extrapolate')
    f50 = f(0.5)
    g50 = g(f50)
    if g50 >= 1.:
        g50 = 1.
    f16 = f(0.16)
    g16 = g(f16)
    f84 = f(0.84)
    if f84 <= time_bins[0]:
        f84 = time_bins[0]
    if f84 >= 0:
        g84 = g(f84)
    else:
        g84 = 1.0
    xerr = [[abs(f84-f50)],[abs(f50-f16)]]
    yerr = [[abs(g50-g16)],[abs(g84-g50)]]
    return f50, g50, f84, g84, f16, g16, xerr, yerr

###############################################################################

## reading 3254 infall time and splitting
tinf = np.sort(np.array(pd.read_csv('./inf_peri_files/Rvir_Ms_inf_time.csv')['3254'].dropna()))
## median of infall
inf_med_idx = np.where(tinf > np.median(tinf))[0][0]
## split about median
tinf1 = tinf[0:inf_med_idx]
tinf2 = tinf[inf_med_idx:]

## reading 3254 pericenter time and splitting
tperi = np.sort(np.array(pd.read_csv('./inf_peri_files/Rvir_Ms_peri_time.csv')['3254'].dropna()))
## median of infall
peri_med_idx = np.where(tperi > np.median(tperi))[0][0]
## split about median
tperi1 = tperi[0:peri_med_idx]
tperi2 = tperi[peri_med_idx:]


log_Ms = np.array(pd.read_csv('smhm_behroozi2010.csv')['logMs'])[0]
log_Mh = np.array(pd.read_csv('smhm_behroozi2010.csv')['logMh'])[0]

int_frac = np.array(pd.read_csv('./int_frac_files/Rvir_Ms_int_frac.csv')['int_frac'])[0]

sfr_df = pd.read_csv('./sfr_ssfr_tables/miles/corr_sfr.csv')

norm = Normalize(vmin=9, vmax=11)
smap = ScalarMappable(cmap=cmap, norm=norm)
smap.set_array([])

fig, ax = plt.subplots(figsize=(10,7))

be = np.linspace(0.,ageU,20)
a = sfr_df['Age_Gyr'] 
s = sfr_df['3254']
f = interp1d(a, s, fill_value='extrapolate') 
b = be[np.where(be>=0.)[0]] # bin edges for sfr, >= 0.5Gyr
s_b = f(b)
sfrbins, mcdf = cdf_sfr(b,s_b)

## plotting infall time and infall time splits
timebinsinf, tinfcdf = cdf_time(be, tinf, int_frac)
f50i,g50i,f84i,g84i,f16i,g16i,xerri,yerri = error_bars(timebinsinf, 
                                                               tinfcdf, 
                                                               sfrbins, mcdf)

timebinsinf1, tinfcdf1 = cdf_time(be, tinf1, int_frac)
timebinsinf2, tinfcdf2 = cdf_time(be, tinf2, int_frac)
f50i1,g50i1,f84i1,g84i1,f16i1,g16i1,xerri1,yerri1 = error_bars(timebinsinf1, 
                                                               tinfcdf1, 
                                                               sfrbins, mcdf)
f50i2,g50i2,f84i2,g84i2,f16i2,g16i2,xerri2,yerri2 = error_bars(timebinsinf2, 
                                                               tinfcdf2, 
                                                               sfrbins, mcdf)

timebinsperi, tpericdf = cdf_time(be, tperi, int_frac)
f50p,g50p,f84p,g84p,f16p,g16p,xerrp,yerrp = error_bars(timebinsperi, 
                                                               tpericdf, 
                                                               sfrbins, mcdf)

timebinsperi1, tpericdf1 = cdf_time(be, tperi1, int_frac)
timebinsperi2, tpericdf2 = cdf_time(be, tperi2, int_frac)
f50p1,g50p1,f84p1,g84p1,f16p1,g16p1,xerrp1,yerrp1 = error_bars(timebinsperi1, 
                                                               tpericdf1, 
                                                               sfrbins, mcdf)
f50p2,g50p2,f84p2,g84p2,f16p2,g16p2,xerrp2,yerrp2 = error_bars(timebinsperi2, 
                                                               tpericdf2, 
                                                               sfrbins, mcdf)
c1 = 'r'
c2 = 'k'
c = cmap(norm(log_Ms))

ax.errorbar(f50i, g50i, xerr=xerri, yerr=yerri, elinewidth=1, capsize=5, 
            ecolor=c, marker='o', mec=c, mfc='w', markersize=16)

ax.errorbar(f50i1, g50i1, xerr=xerri1, yerr=yerri1, elinewidth=1, capsize=5, 
            ecolor=c1, marker='o', mec=c1, mfc='w', markersize=16)
ax.errorbar(f50i2, g50i2, xerr=xerri2, yerr=yerri2, elinewidth=1, capsize=5, 
            ecolor=c2, marker='o', mec=c2, mfc='w', markersize=16)


ax.errorbar(f50p, g50p, xerr=xerrp, yerr=yerrp, elinewidth=1, capsize=5, 
            ecolor=c, marker='*', mec=c, mfc='w', markersize=20)

ax.errorbar(f50p1, g50p1, xerr=xerrp1, yerr=yerrp1, elinewidth=1, capsize=5, 
            ecolor=c1, marker='*', mec=c1, mfc='w', markersize=20)
ax.errorbar(f50p2, g50p2, xerr=xerrp2, yerr=yerrp2, elinewidth=1, capsize=5, 
            ecolor=c2, marker='*', mec=c2, mfc='w', markersize=20)

ax.set_ylim(0.50,1.05)
ax.set_yticks(ax.get_yticks()[1:-1]) # Remove first and last ticks
ax.set_xlim(-1,11.0)
ax.set_ylabel(r'Fraction of cumulative $M_\star$ formed',fontsize=18)
ax.set_xlabel('Lookback time [Gyr]',fontsize=18)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both',which='major',direction='in', bottom = True, 
                top = True,left = True, right = True, length=10, labelsize=18)
ax.tick_params(axis='both',which='minor',direction='in', bottom = True, 
               top = True,left = True, right = True, length=5, labelsize=18)
t = mlines.Line2D([], [], color=c, marker='o', linestyle='None',
                              markersize=8, label='Infall')
t1 = mlines.Line2D([], [], color=c1, marker='o', linestyle='None',
                              markersize=8, label='Infall (later peak)') 
t2 = mlines.Line2D([], [], color=c2, marker='o', linestyle='None',
                              markersize=10, label='Infall (earlier peak)')
p = mlines.Line2D([], [], color=c, marker='*', linestyle='None',
                              markersize=8, label='Pericenter')
p1 = mlines.Line2D([], [], color=c1, marker='*', linestyle='None',
                              markersize=8, label='Pericenter (later peak)') 
p2 = mlines.Line2D([], [], color=c2, marker='*', linestyle='None',
                              markersize=10, label='Pericenter (earlier peak)')
ax.legend(handles=[t, p, t1, p1, t2, p2],fontsize=15, loc=3, 
          bbox_to_anchor=(0.02,0.02), bbox_transform=ax.transAxes) 
cbar = fig.colorbar(smap, ticks=[9., 9.5, 10.0, 10.5, 11.0])
cbar.set_label(r'$\log_{10}(M_\star/{\rm M}_\odot)$',fontsize=18)
cbar.ax.tick_params(axis='y', direction='in',labelsize=18)
ax.grid(False)
ax.set_facecolor('w')
plt.savefig('pdf_split.pdf', dpi=500)