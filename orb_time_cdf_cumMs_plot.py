# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 07:27:48 2020

@author: Amit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.ticker import AutoMinorLocator
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages

gmp_names = ['3254','3269', '3291', '3329', '3352', '3367', '3414', '3484',
             '3534', '3565', '3639', '3664']

log_Ms = np.loadtxt('logMs_coma.m')
Ms = 10**log_Ms


int_frac = np.array(pd.read_csv('int_frac_mmax.csv')['int_frac'])


def cum_mass(name,bin_edges,delta_bin_edges):
    age = sfr_df['Age_Gyr']*10**9
    sfr = sfr_df[name]
    t_arr = np.append(bin_edges,bin_edges[-1]+delta_bin_edges)
    sfr_lin_interp = np.interp(t_arr,age,sfr)
    # integrating and obtaining mass gain in each bin
    m = np.array([])
    for i in range(len(bin_edges)):
        m = np.append(m,0.5*(t_arr[i+1]-t_arr[i])*
                      (sfr_lin_interp[i+1]+sfr_lin_interp[i]))
#    
    m_cumsum = np.cumsum(m)
    min_m_cumsum = min(m_cumsum)
    minmax_m_cumsum = max(m_cumsum) - min(m_cumsum)
    m_cdf = (m_cumsum - min_m_cumsum) / minmax_m_cumsum
    return m_cdf,m

def cum_time(name,bins,time_bool):
    if time_bool == 'inf':
        time = inf_df[name].dropna()*10**9
    else:
        time = peri_df[name].dropna()*10**9
    time_b = pd.cut(time,bins=bins)
    time_count = time_b.value_counts(sort=False)
    time_hist = np.array(time_count).astype(float)/len(np.array(time))
    time_cdf = np.cumsum(time_hist)
    return time_cdf

def err_plots(name,time_bool,i_frac):
    bins,bin_edges,delta_bin_edges, bin_edges_neg = bin_gen(name,time_bool)
    time_cdf = cum_time(name,bins,time_bool)
    time_cdf = time_cdf * (1-i_frac)
    m_cdf,m = cum_mass(name,bin_edges,delta_bin_edges)
    if time_bool == 'inf':
        f = interpolate.interp1d(1-time_cdf,bin_edges,fill_value='extrapolate')
    else:
        f = interpolate.interp1d(1-time_cdf,bin_edges_neg,fill_value='extrapolate')
    g = interpolate.interp1d(bin_edges,1-m_cdf,fill_value='extrapolate')
    f50 = f(0.5)
    g50 = g(f50)
    f16 = f(0.16)
    g16 = g(f16)
    f84 = f(0.84)
    if f84 > 0:
        g84 = g(f84)
    else:
        g84 = 1.0
    xerr = [[abs(f84-f50)],[abs(f50-f16)]]
    yerr = [[abs(g50-g16)],[abs(g84-g50)]]
    return f50, g50, f16, f84, g16, g84, xerr, yerr, bin_edges, m_cdf, time_cdf, bin_edges_neg


def bin_gen(name,time_bool):
    if time_bool == 'inf':
        bins = np.linspace(0,13.7,20)*10**9 
        # defineing equally spaced bins from 0-13.7 Gyr
    else:
        bins = np.linspace(np.floor(min(peri_df[name])),13.7,25)*10**9
        # defineing equally spaced bins from min. of peri time -13.7 Gyr
    bin_edges = bins[1:]
    bin_edges_neg = bin_edges
    delta_bin_edges = bin_edges[1] - bin_edges[0]
    bin_edges = np.array([x for x in bin_edges if x > 0]) 
    return bins, bin_edges, delta_bin_edges, bin_edges_neg

sfr_df = pd.read_csv('corr_sfr.csv')
inf_df = pd.read_csv('inf_time_mmax.csv')
peri_df = pd.read_csv('peri_time_mmax.csv')

# plotting indiviually for sanity checks - infall time
pdf = PdfPages('ebar_inf_ind.pdf')
for i_frac, name in zip(int_frac,gmp_names):
    if name == '3329':
        continue
    else:
        fig, ax = plt.subplots(figsize=(10,7))
        f50, g50, f16, f84, g16, g84, xerr, yerr, be, m_cdf, time_cdf, be_neg = err_plots(name,'inf',i_frac)
        ax.plot(be/10**9,1-m_cdf,color='tab:red',linestyle='-',
                label=r'Cum. stellar mass ($M_\star$)')
        ax.plot(be/10**9,1-time_cdf,color='tab:orange',linestyle='-',
                label=r'cum. $t_{infall}$')
        ax.plot([f50/10**9],[0.5],marker='o',color='tab:purple',markersize=10,
                label=r'cum. $t_{infall}$:50%')
        ax.plot([f84/10**9],[0.84],marker='<',color='tab:purple',markersize=10,
                label=r'cum. $t_{infall}$:84%')
        ax.plot([f16/10**9],[0.16],marker='>',color='tab:purple',markersize=10,
                label=r'cum. $t_{infall}$:16%')
        ax.plot([f50/10**9],[g50],marker='o',color='tab:pink',markersize=10,
                label=r'$M_\star$ at cum. $t_{infall}$:50%')
        ax.plot([f84/10**9],[g84],marker='^',color='tab:pink',markersize=10,
                label=r'$M_\star$ at cum. $t_{infall}$:84%')
        ax.plot([f16/10**9],[g16],marker='v',color='tab:pink',markersize=10,
                label=r'$M_\star$ at cum. $t_{infall}$:16%')
        ax.plot([f50/10**9,f84/10**9],[0.5,0.5],linewidth=5,color='y',
                label=r'50-84% cum. $t_{infall}$')
        ax.plot([f50/10**9,f16/10**9],[0.5,0.5],linewidth=5,color='c',
                label=r'16-50% cum. $t_{infall}$')
        ax.plot([f50/10**9,f50/10**9],[g50,g84],linewidth=5,color='b',
                label=r'$\Delta M_\star$ in 50-84% cum. $t_{infall}$')
        ax.plot([f50/10**9,f50/10**9],[g50,g16],linewidth=5,color='g',
                label=r'$\Delta M_\star$ in 16-50% cum. $t_{infall}$')
        ax.set_xticks(np.linspace(np.floor(min(be/10**9)),
                                  np.ceil(max(be/10**9)),8))
        ax.set_yticks(ticks=np.linspace(0,1,11))
        ax.set_xlabel('Lookback time [Gyr]',fontsize=18)
        ax.set_ylabel(r'CDF $t_{inf}$ / Cumulative fraction of $M_\star$ formed',
                      fontsize=18)
        line1 = mlines.Line2D([], [], color='tab:red', marker='None', linestyle='-',
                              label='Cumulative fraction of stellar mass formed')
        line2 = mlines.Line2D([], [], color='tab:orange', marker='None', linestyle='-',
                              label='CDF: Infall')
        ax.legend(handles=[line1,line2],loc=3,frameon=False, framealpha=1.0,
                  fontsize=16)
        ax.grid(False)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis='both',which='both',direction='in', bottom = True, 
                       top = True,left = True, right = True,labelsize=18)
        ax.annotate('GMP '+name, xy=(0.05, 0.15), xycoords='axes fraction',
                    fontsize=16)
        pdf.savefig(fig,dpi=500)
        #plt.savefig('error_bars_plots/ebar_inf_'+name+'_280520.png',dpi=200)
        plt.savefig('error_bars_plots/ebar_inf_'+name+'.pdf',dpi=500)
pdf.close()
#

## plotting indiviually for sanity checks - pericenter time
pdf = PdfPages('ebar_peri_ind.pdf')
for i_frac, name in zip(int_frac,gmp_names):
    if name == '3329':
        continue
    else:
        fig, ax = plt.subplots(figsize=(10,7))
        f50, g50, f16, f84, g16, g84, xerr, yerr, be, m_cdf, time_cdf, be_neg = err_plots(name,'peri',i_frac)
        ax.plot(be/10**9,1-m_cdf,color='tab:red',linestyle='-',
                label=r'Cum. stellar mass ($M_\star$)')
        ax.plot(be_neg/10**9,1-time_cdf,color='tab:orange',linestyle='-',
                label=r'cum. $t_{peri}$')
        ax.plot([f50/10**9],[0.5],marker='o',color='tab:purple',markersize=10,
                label=r'cum. $t_{peri}$:50%')
        ax.plot([f84/10**9],[0.84],marker='<',color='tab:purple',markersize=10,
                label=r'cum. $t_{peri}$:84%')
        ax.plot([f16/10**9],[0.16],marker='>',color='tab:purple',markersize=10,
                label=r'cum. $t_{peri}$:16%')
        ax.plot([f50/10**9],[g50],marker='o',color='tab:pink',markersize=10,
                label=r'$M_\star$ at cum. $t_{peri}$:50%')
        ax.plot([f84/10**9],[1.0],marker='^',color='tab:pink',markersize=10,
                label=r'$M_\star$ at cum. $t_{peri}$:84%')
        ax.plot([f16/10**9],[g16],marker='v',color='tab:pink',markersize=10,
                label=r'$M_\star$ at cum. $t_{peri}$:16%')
        ax.plot([f50/10**9,f84/10**9],[0.5,0.5],linewidth=5,color='y',
                label=r'50-84% cum. $t_{peri}$')
        ax.plot([f50/10**9,f16/10**9],[0.5,0.5],linewidth=5,color='c',
                label=r'16-50% cum. $t_{peri}$')
        ax.plot([f50/10**9,f50/10**9],[g50,g84],linewidth=5,color='b',
                label=r'$\Delta M_\star$ in 50-84% cum. $t_{peri}$')
        ax.plot([f50/10**9,f50/10**9],[g50,g16],linewidth=5,color='g',
                label=r'$\Delta M_\star$ in 16-50% cum. $t_{peri}$')
        ax.set_xticks(np.linspace(np.floor(min(be_neg/10**9)),
                                  np.ceil(max(be_neg/10**9)),8))
        #ax.set_xticks(ticks=np.linspace(np.floor(min(be_neg/10**9)),14.,8))
        ax.set_yticks(ticks=np.linspace(0,1,11))
        ax.set_xlabel('Lookback time [Gyr]',fontsize=18)
        ax.set_ylabel(r'CDF $t_{peri}$ / Cumulative fraction of $M_\star$ formed',
                      fontsize=18)
        line1 = mlines.Line2D([], [], color='tab:red', marker='None', linestyle='-',
                              label='Cumulative fraction of stellar mass formed')
        line2 = mlines.Line2D([], [], color='tab:orange', marker='None', linestyle='-',
                              label='CDF: Pericenter')
        ax.legend(handles=[line1,line2],loc=3,frameon=False, framealpha=1.0,
                  fontsize=16)
        ax.grid(False)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis='both',which='both',direction='in', bottom = True, 
                       top = True,left = True, right = True,labelsize=18)
        ax.annotate('GMP '+name, xy=(0.05, 0.15), xycoords='axes fraction',
                    fontsize=16)
        pdf.savefig(fig,dpi=500)
        #plt.savefig('error_bars_plots/ebar_peri_'+name+'_280520.png',dpi=200)
        plt.savefig('error_bars_plots/ebar_peri_'+name+'.pdf',dpi=500)
pdf.close()
