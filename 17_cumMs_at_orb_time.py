# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:55:50 2020

@author: Amit
"""

import warnings
warnings.filterwarnings("ignore")

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.cm import ScalarMappable, viridis as cmap
from matplotlib.colors import Normalize
from scipy import interpolate
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_pdf import PdfPages

# output directory
def gen_out_dir(ext,ssp):
    out_dir = './cumMs_at_inf_peri/'+ssp+'/'+ext+'/'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    return out_dir

# sat names
gmp_names = ['3254','3269', '3291', '3329', '3352', '3367', '3414', '3484',
            '3534', '3565', '3639', '3664']


def cum_mass(name,bin_edges,delta_bin_edges,sfr_df):
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

def cum_time(name,bins,time_bool,inf_df,peri_df):
    if time_bool == 'inf':
        time = inf_df[name].dropna()*10**9
    else:
        time = peri_df[name].dropna()*10**9
    time_b = pd.cut(time,bins=bins)
    time_count = time_b.value_counts(sort=False)
    time_hist = np.array(time_count).astype(float)/len(np.array(time))
    time_cdf = np.cumsum(time_hist)
    return time_cdf

def err_plots(name,time_bool,i_frac,inf_df,peri_df,sfr_df):
    bins,bin_edges,delta_bin_edges, bin_edges_neg = bin_gen(name,time_bool,inf_df,peri_df)
    time_cdf = cum_time(name,bins,time_bool,inf_df,peri_df)
    # time_cdf = (1-time_cdf) * (1-i_frac)
    # time_cdf = 1-time_cdf
    time_cdf = time_cdf * (1-i_frac)
    m_cdf,m = cum_mass(name,bin_edges,delta_bin_edges,sfr_df)
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
    return f50, g50, f16, f84, g16, g84, xerr, yerr

def bin_gen(name,time_bool,inf_df,peri_df):
    if time_bool == 'inf':
        bins = np.linspace(0,13.7,20)*10**9 
        # defineing equally spaced bins from 0-13.7 Gyr
    else:
        bins = np.linspace(np.floor(min(peri_df[name].dropna())),13.7,25)*10**9
        # defineing equally spaced bins from min. of peri time -13.7 Gyr
    bin_edges = bins[1:]
    bin_edges_neg = bin_edges
    delta_bin_edges = bin_edges[1] - bin_edges[0]
    bin_edges = np.array([x for x in bin_edges if x > 0]) 
    return bins, bin_edges, delta_bin_edges, bin_edges_neg


def plot_all(ext,ssp):
    Ms_sign = ext.split('s')[1]
    
    log_Ms = np.array(pd.read_csv('smhm_behroozi2010.csv')['logMs'+Ms_sign])
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
            f50i,g50i,f16i,f84i,g16i,g84i,xerri,yerri=err_plots(name,'inf',i_frac,inf_df,peri_df,sfr_df)
            c = cmap(norm(log_Mi))
            xerri = [[j/10**9 for j in i] for i in xerri] # to put x error in Gyrs
            ax.errorbar(f50i/10**9,g50i,xerr=xerri,yerr=yerri,elinewidth=1, 
                        capsize=5, ecolor=c, marker='o', mec=c, mfc=c,markersize=8)
            gmp = np.append(gmp,name)
            logMs = np.append(logMs,'{:.2f}'.format(log_Mi))
            logMh = np.append(logMh,'{:.2f}'.format(log_Hi))
            intf = np.append(intf,'{:.2f}'.format(i_frac*100))
            tinf = np.append(tinf,'{:.2f}'.format(f50i/10**9))
            tinfp = np.append(tinfp,'{:.2f}'.format(abs(f16i-f50i)/10**9))
            tinfm = np.append(tinfm,'{:.2f}'.format(abs(f50i-f84i)/10**9))
            Mspti = np.append(Mspti,'{:.2f}'.format(g50i*100))
            Msptip = np.append(Msptip,'{:.2f}'.format(abs(g84i-g50i)*100))
            Msptim = np.append(Msptim,'{:.2f}'.format(abs(g50i-g16i)*100))

    for name,log_Mi in zip(gmp_names,log_Ms):
        if name == '3329':
            continue
        else:
            f50p,g50p,f16p,f84p,g16p,g84p,xerrp,yerrp=err_plots(name,'peri',i_frac,inf_df,peri_df,sfr_df)
            c = cmap(norm(log_Mi))
            xerrp = [[j/10**9 for j in i] for i in xerrp]
            ax.errorbar(f50p/10**9,g50p,xerr=xerrp,yerr=yerrp,elinewidth=1, 
                        capsize=5, ecolor=c, marker='*', mec=c, mfc=c,markersize=10)
            tperi = np.append(tperi,'{:.2f}'.format(f50p/10**9))
            tperip = np.append(tperip,'{:.2f}'.format(abs(f16p-f50p)/10**9))
            tperim = np.append(tperim,'{:.2f}'.format(abs(f50p-f84p)/10**9))
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
    ax.legend(handles=[circ, star],frameon=False, framealpha=1.0,loc=3,fontsize=16) 
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

uncert_ext = ['Rvir_Ms','Rvir_Ms+','Rvir_Ms-','Rvir+_Ms','Rvir+_Ms+','Rvir+_Ms-',
          'Rvir-_Ms','Rvir-_Ms+','Rvir-_Ms-']
ssps = ['miles','bc03','phr']

for ssp in ssps:
    for ext in uncert_ext:
        plot_all(ext,ssp)
