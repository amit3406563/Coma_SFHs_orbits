# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 22:01:08 2020

@author: amit
"""


import warnings
warnings.filterwarnings("ignore")

import os
import shutil
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from astropy.cosmology import Planck13
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.cm import ScalarMappable, viridis as cmap
from matplotlib.colors import Normalize
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_pdf import PdfPages

# output directory
def gen_out_dir(ext,ssp):
    out_dir = './cumMs_at_inf_peri/'+ssp+'/'+ext+'/'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    return out_dir

ageU = Planck13.age(0).value # age of the Universe as per Planck 13 cosmology

gmp_names = ['3254','3269', '3291', '3329', '3352', '3367', '3414', '3484',
              '3534', '3565', '3639', '3664']

# gmp_names = ['3254']

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

def plot_all(ext,ssp):
    Ms_sign = ext.split('s')[1]
    
    log_Ms = np.array(pd.read_csv('smhm_behroozi2010.csv')['logMs'])
    #Ms = 10**log_Ms

    log_Mh = np.array(pd.read_csv('smhm_behroozi2010.csv')['logMh'+Ms_sign])
    #Mh = 10**log_Mh

    int_frac = np.array(pd.read_csv('./int_frac_files/'+ext+'_'+'int_frac.csv')['int_frac'])
    sfr_df = pd.read_csv('./sfr_ssfr_tables/'+ssp+'/corr_sfr.csv')
    inf_df = pd.read_csv('./inf_peri_files/'+ext+'_'+'inf_time.csv')
    peri_df = pd.read_csv('./inf_peri_files/'+ext+'_'+'peri_time.csv')

    norm = Normalize(vmin=9, vmax=11)
    smap = ScalarMappable(cmap=cmap, norm=norm)
    smap.set_array([])

    ## declaring empty arrays for Tabulatingng the results
    gmp = np.array([])
    logMs = np.array([])
    logMh = np.array([])
    intf = np.array([])
    tinf = np.array([]) # tinf CDF 50th percentile
    tinfp = np.array([]) # tinf CDF 16 - 50 th percentile: tinf +
    tinfm = np.array([]) # tinf CDF 50 - 84 th percentile: tinf -
    tperi = np.array([]) # tperi CDF 50th percentile
    tperip = np.array([]) # tperi CDF 16 - 50 th percentile: tperi +
    tperim = np.array([]) # tperi CDF 50 - 84 th percentile: tperi -
    Mspti = np.array([]) # %age of M* formed at 50th percentile of tinf CDF
    Msptip = np.array([]) # %age of M* formed at 84 - 50 th percentile of tinf CDF
    Msptim = np.array([]) # %age of M* formed at 50 - 16 th percentile of tinf CDF
    Msptp = np.array([]) # %age of M* formed at 50th percentile of tperi CDF
    Msptpp = np.array([]) # %age of M* formed at 84 - 50 th percentile of tperi CDF
    Msptpm = np.array([]) # %age of M* formed at 50 - 16 th percentile of tperi CDF

    # plotting all error bars for both pericenter time and infall time
    fig, ax = plt.subplots(figsize=(10,7))
    for name,log_Mi,i_frac,log_Hi in zip(gmp_names,log_Ms,int_frac,log_Mh):
        if name == '3329':
            continue
        else:
            time = np.array(inf_df[name].dropna())
            be = np.linspace(0.,ageU,20)
            a = sfr_df['Age_Gyr'] 
            s = sfr_df[name]
            f = interp1d(a, s, fill_value='extrapolate') 
            b = be[np.where(be>=0.)[0]] # bin edges for sfr, >= 0.5Gyr
            s_b = f(b)
            sfrbins, mcdf = cdf_sfr(b,s_b)
            timebins, tcdf = cdf_time(be, time, i_frac)
            f50i,g50i,f84i,g84i,f16i,g16i,xerri,yerri = error_bars(timebins, 
                                                                  tcdf, 
                                                                  sfrbins, 
                                                                  mcdf)
            c = cmap(norm(log_Mi))
            ax.errorbar(f50i, g50i, xerr=xerri, yerr=yerri, elinewidth=1, 
                        capsize=5, ecolor=c, marker='o', mec=c, mfc=c, 
                        markersize=8)
            gmp = np.append(gmp,name)
            logMs = np.append(logMs,'{:.2f}'.format(log_Mi))
            logMh = np.append(logMh,'{:.2f}'.format(log_Hi))
            intf = np.append(intf,'{:.2f}'.format(i_frac*100))
            tinf = np.append(tinf,'{:.2f}'.format(f50i))
            tinfp = np.append(tinfp,'{:.2f}'.format(abs(f16i-f50i)))
            tinfm = np.append(tinfm,'{:.2f}'.format(abs(f50i-f84i)))
            Mspti = np.append(Mspti,'{:.2f}'.format(g50i*100))
            Msptip = np.append(Msptip,'{:.2f}'.format(abs(g84i-g50i)*100))
            Msptim = np.append(Msptim,'{:.2f}'.format(abs(g50i-g16i)*100))

    for name,log_Mi in zip(gmp_names,log_Ms):
        if name == '3329':
            continue
        else:
            time = np.array(peri_df[name].dropna())
            be = np.linspace(min(time),ageU,25)
            a = sfr_df['Age_Gyr'] 
            s = sfr_df[name]
            f = interp1d(a, s, fill_value='extrapolate') 
            b = be[np.where(be>=0.)[0]] # bin edges for sfr, >= 0.5Gyr
            s_b = f(b)
            sfrbins, mcdf = cdf_sfr(b,s_b)
            timebins, tcdf = cdf_time(be, time, i_frac)
            f50p,g50p,f84p,g84p,f16p,g16p,xerrp,yerrp = error_bars(timebins, 
                                                                  tcdf, 
                                                                  sfrbins, 
                                                                  mcdf)
            c = cmap(norm(log_Mi))
            ax.errorbar(f50p, g50p, xerr=xerrp, yerr=yerrp, elinewidth=1, 
                        capsize=5, ecolor=c, marker='*', mec=c, mfc=c, 
                        markersize=10)
            tperi = np.append(tperi,'{:.2f}'.format(f50p))
            tperip = np.append(tperip,'{:.2f}'.format(abs(f16p-f50p)))
            tperim = np.append(tperim,'{:.2f}'.format(abs(f50p-f84p)))
            Msptp = np.append(Msptp,'{:.2f}'.format(g50p*100))
            Msptpp = np.append(Msptpp,'{:.2f}'.format(abs(g84p-g50p)*100))
            Msptpm = np.append(Msptpm,'{:.2f}'.format(abs(g50p-g16p)*100))
    
    ax.set_ylim(0.50,1.05)
    ax.set_yticks(ax.get_yticks()[1:-1]) # Remove first and last ticks
    ax.set_xlim(-1,11.0)
    ax.set_ylabel(r'Fraction of cumulative $M_\star$ formed',fontsize=18)
    ax.set_xlabel('Lookback time [Gyr]',fontsize=18)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both',which='both',direction='in', bottom = True, 
                       top = True,left = True, right = True,labelsize=18)
    circ = mlines.Line2D([], [], color='grey', marker='o', linestyle='None',
                              markersize=8, label='Infall time') 
    star = mlines.Line2D([], [], color='grey', marker='*', linestyle='None',
                              markersize=10, label='Pericenter time')
    ax.legend(handles=[circ, star],fontsize=15, loc=3, bbox_to_anchor=(0.02,0.02), 
              bbox_transform=ax.transAxes) 
    cbar = fig.colorbar(smap, ticks=[9., 9.5, 10.0, 10.5, 11.0])
    cbar.set_label(r'$\log_{10}(M_\star/{\rm M}_\odot)$',fontsize=18)
    cbar.ax.tick_params(axis='y', direction='in',labelsize=18)
    ax.grid(False)
    ax.set_facecolor('w')
    out_dir = gen_out_dir(ext,ssp)
    plt.savefig(out_dir+ext+'_'+ssp+'_'+'cum_Ms_at_exp_orb_time.pdf',dpi=500)
    #plt.savefig('ebar_inf_peri.png',dpi=200)

    ## Tabulating results
    df = pd.DataFrame()
    df['GMP'] = gmp
    df['log_Ms'] = logMs
    df['log_Mh'] = logMh
    df['int_frac'] = intf
    df['tinf'] = tinf
    df['tinf+'] = tinfp
    df['tinf-'] = tinfm
    df['tperi'] = tperi
    df['tperi+'] = tperip
    df['tperi-'] = tperim
    df['%Ms_tinf'] = Mspti
    df['%Ms_tinf+'] = Msptip
    df['%Ms_tinf-'] = Msptim
    df['%Ms_tperi'] = Msptp
    df['%Ms_tperi+'] = Msptpp
    df['%Ms_tperi-'] = Msptpm
    #df.to_csv('rvir_m.csv',index=False)
    df.to_csv(out_dir+ext+'_'+ssp+'_'+'cum_Ms_at_exp_orb_time_table.csv',
              index=False)

    fig, ax = plt.subplots(figsize=(7,4))
    #ax.axis('tight')
    ax.axis('off')
    tab = ax.table(cellText=df.values,colLabels=df.columns,loc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(6)
    tab.auto_set_column_width(col=list(range(len(df.columns))))
    pp = PdfPages(out_dir+ext+'_'+ssp+'_'+'cum_Ms_at_exp_orb_time_table.pdf')
    pp.savefig(fig, bbox_inches='tight')
    pp.close()

    
###############################################################################
## Plotting Cum %age of Ms formed at expected Infall and Pericenter ##
###############################################################################

# uncert_ext = ['Rvir_Ms','Rvir_Ms+','Rvir_Ms-','Rvir+_Ms','Rvir+_Ms+','Rvir+_Ms-',
#           'Rvir-_Ms','Rvir-_Ms+','Rvir-_Ms-']
# uncert_ext = ['Rvir_Ms','Rvir_Ms+','Rvir_Ms-','Rvir+_Ms','Rvir-_Ms']
# ssps = ['miles','bc03','phr']
ssps = ['miles']
uncert_ext = ['Rvir_Ms']
for ssp in ssps:
    for ext in uncert_ext:
        plot_all(ext,ssp)