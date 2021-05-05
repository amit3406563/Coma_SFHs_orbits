# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 07:22:32 2020

@author: amit
"""

import warnings
warnings.filterwarnings("ignore")

import os
import shutil
import numpy as np
import pandas as pd
#from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from astropy.cosmology import Planck13
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages

ageU = Planck13.age(0).value # age of the Universe as per Planck 13 cosmology

sat_names = ['3254','3269', '3291', '3329', '3352', '3367', '3414', '3484',
              '3534', '3565', '3639', '3664']


# gmp_names = ['3254','3269', '3291', '3352', '3367', '3414', '3484',
#               '3534', '3565', '3639', '3664']

gmp_names = ['3254']

def gen_out_dir(ext,ssp):
    out_dir = './cdf_inf_peri_vs_cumMs_files/'+ssp+'/'+ext+'/'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    return out_dir

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
    # xerr = [[abs(f84-f50)],[abs(f50-f16)]]
    # yerr = [[abs(g50-g16)],[abs(g84-g50)]]
    return f50, g50, f16, f84, g16, g84

def plotting(time_bool, t_inf, t_peri, sfr, i_frac, name):
    if time_bool == 'inf':
        time = np.array(t_inf[name].dropna())
    else:
        time = np.array(t_peri[name].dropna())

    if time_bool == 'inf':
        be = np.linspace(0.,ageU,20)
    else:
        be = np.linspace(min(time),ageU,25)
    # be = np.histogram_bin_edges(time, bins='rice') # generate best bin edges

    a = sfr['Age_Gyr'] 
    s = sfr[name]
    f = interp1d(a, s, fill_value='extrapolate') 
    b = be[np.where(be>=0.)[0]] # bin edges for sfr, >= 0.5Gyr
    s_b = f(b)

    sfrbins, mcdf = cdf_sfr(b,s_b)

    timebins, tcdf = cdf_time(be, time, i_frac)
    
    f50, g50, f16, f84, g16, g84 = error_bars(timebins, tcdf, sfrbins, mcdf)
    
    fig, ax = plt.subplots(figsize=(10,7))
    ax.plot(sfrbins, mcdf, color='r', linestyle='-')
    ax.plot(timebins, tcdf, color='k', linestyle='-')
    ax.plot([f50], [0.5], marker='o', color='k', markersize=10)
    ax.annotate('', xy = (f84,0.5), xytext = (f84,0.84), 
                arrowprops=dict(arrowstyle='->',color='k', 
                                linestyle='-',linewidth=2))
    ax.annotate('', xy = (f16,0.5), xytext = (f16,0.16), 
                arrowprops=dict(arrowstyle='->',color='k', 
                                linestyle='-',linewidth=2))
    ax.plot([f50],[g50],marker='o',color='r',markersize=10)
    if time_bool == 'inf':
        ax.annotate('', xy = (f50,g84), xytext = (f84,g84), 
                    arrowprops=dict(arrowstyle='->',color='r', 
                                    linestyle='-',linewidth=2))
    else:
        ax.annotate('', xy = (f50,1.0), xytext = (f84,1.0), 
                    arrowprops=dict(arrowstyle='->',color='r', 
                                    linestyle='-',linewidth=2))
    ax.annotate('', xy = (f50,g16), xytext = (f16,g16), 
                arrowprops=dict(arrowstyle='->',color='r', 
                                linestyle='-',linewidth=2))
    ax.plot([f50,f84], [0.5,0.5], linewidth=4, color='k')
    ax.plot([f50,f16], [0.5,0.5], linewidth=4, color='k')
    ax.plot([f50,f50], [g50,g84], linewidth=4, color='r')
    ax.plot([f50,f50], [g50,g16], linewidth=4, color='r')
    ax.annotate('', xy = (f84,0.84), xytext = (f84,g84), 
                arrowprops=dict(arrowstyle='-',color='r', 
                                linestyle='--',linewidth=2))
    ax.annotate('', xy = (f16,0.5), xytext = (f16,g16), 
                arrowprops=dict(arrowstyle='-',color='r', 
                                linestyle='--',linewidth=2))
    # ax.annotate('GMP '+name, xy=(0.5, 0.95), xycoords='axes fraction',
    #             fontsize=16)
    ax.set_xticks(np.linspace(np.floor(min(timebins)), 
                              np.ceil(max(timebins)),8))
    ax.set_ylim(0.,1.)
    ax.set_yticks([0.,0.16,0.5,0.84,1.])
    ax.set_xlim(0.,ageU)
    ax.set_xticks([2.,4.,6.,8.,10.,12.])
    ax.set_xlabel('Lookback time [Gyr]',fontsize=18)
    if time_bool == 'inf':
        ax.set_ylabel(r'$t_\mathrm{inf}$ CDF ; Cumulative SFH', fontsize=18)
        line2 = mlines.Line2D([], [], color='k', marker='None', linestyle='-', 
                              label=r'$t_{\mathrm{inf}}$ CDF')
    else:
        ax.set_ylabel(r'$t_\mathrm{peri}$ CDF ; Cumulative SFH', fontsize=18)
        line2 = mlines.Line2D([], [], color='k', marker='None', linestyle='-', 
                              label=r'$t_{\mathrm{peri}}$ CDF')
    line1 = mlines.Line2D([], [], color='r', marker='None', linestyle='-', 
                          label='Cumulative SFH')
    ax.legend(handles=[line1,line2], frameon=True, framealpha=1.0, 
              edgecolor='k', fontsize=14, loc=1, 
              bbox_to_anchor=(0.97,0.97), bbox_transform=ax.transAxes)
    ax.grid(False)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    #ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both',which='major',direction='in', bottom = True, 
                   top = True,left = True, right = True, length=10, 
                   labelsize=18)
    ax.tick_params(axis='both',which='minor',direction='in', bottom = True, 
                   top = True,left = True, right = True, length=5, 
                   labelsize=18)
    
    return fig

def plots(ssp,ext):
    int_frac = np.array(pd.read_csv('./int_frac_files/'+ext+'_'+'int_frac.csv')['int_frac'])
    int_frac_df = pd.DataFrame()
    int_frac_df['GMP'] = sat_names
    int_frac_df['int_frac'] = int_frac
    int_frac_df.set_index('GMP', inplace=True)

    sfr = pd.read_csv('./sfr_ssfr_tables/'+ssp+'/corr_sfr.csv')
    t_inf = pd.read_csv('./inf_peri_files/'+ext+'_'+'inf_time.csv')
    t_peri = pd.read_csv('./inf_peri_files/'+ext+'_'+'peri_time.csv')
    
    out_dir = gen_out_dir(ext,ssp)
    
    pdf = PdfPages(out_dir+ext+'_'+ssp+'_'+'cdf_tinf_sfr.pdf')
    for name in gmp_names:
        i_frac = int_frac_df.loc[name]['int_frac']
        fig = plotting('inf', t_inf, t_peri, sfr, i_frac, name)
        pdf.savefig(fig,dpi=500)
    pdf.close()
    
    pdf = PdfPages(out_dir+ext+'_'+ssp+'_'+'cdf_tperi_sfr.pdf')
    for name in gmp_names:
        i_frac = int_frac_df.loc[name]['int_frac']
        fig = plotting('peri', t_inf, t_peri, sfr, i_frac, name)
        pdf.savefig(fig,dpi=500)
    pdf.close()
    
uncert_ext = ['Rvir_Ms','Rvir_Ms+','Rvir_Ms-','Rvir+_Ms','Rvir-_Ms']
ssps = ['miles','bc03','phr']
# ssps = ['miles']
# uncert_ext = ['Rvir_Ms']
for ssp in ssps:
    for ext in uncert_ext:
        plots(ssp,ext)

