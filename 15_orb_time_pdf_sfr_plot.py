# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 05:18:15 2020

@author: amit
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
from scipy.signal import savgol_filter as sf
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_pdf import PdfPages

## output directory
def gen_out_dir(ext,ssp):
    out_dir = './pdf_inf_peri_vs_sfr_files/'+ssp+'/'+ext+'/'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    return out_dir


## sat names
## GMP IDs of satellites
sat_names = ['3254','3269', '3291', '3352', '3367', '3414', '3484',
             '3534', '3565', '3639', '3664']

###############################################################################
## Functions ##
###############################################################################

def sav_gol(x,win,poly):
    return sf(x, window_length=win, polyorder=poly, mode='interp')


def hist_time(name,bins,time_bool,t_inf,t_peri):
    if time_bool == True:
        time = t_inf[name].dropna()*10**9
    else:
        time = t_peri[name].dropna()*10**9
    time_b = pd.cut(time,bins=bins)
    time_count = time_b.value_counts(sort=False)
    time_hist = np.array(time_count).astype(float)/len(np.array(time))
    return time_hist

def bin_gen(name,time_bool,t_inf,t_peri):
    if time_bool == True:
        bins = np.linspace(0,13.7,20)*10**9 
        # defineing equally spaced bins from 0-13.7 Gyr
    else:
        bins = np.linspace(np.floor(min(t_peri[name])),13.7,25)*10**9
        # defineing equally spaced bins from min. of peri time -13.7 Gyr
    bin_edges = bins[1:]
    bin_edges_neg = bin_edges
    delta_bin_edges = bin_edges[1] - bin_edges[0]
    bin_edges = np.array([x for x in bin_edges if x > 0]) 
    return bins, bin_edges, delta_bin_edges, bin_edges_neg

def plots(pdf,t,rate,sat_name,inf_bool,i_frac,t_inf,t_peri):
    fig, ax1 = plt.subplots(figsize=(10,7))
    bins, bin_edges, delta_bin_edges, bin_edges_neg = bin_gen(sat_name,inf_bool,
                                                              t_inf,t_peri)
    hist = hist_time(sat_name,bins,inf_bool,t_inf,t_peri)
    ax1.bar(bins[1:]/10**9,hist,align='center',width=0.9*delta_bin_edges/10**9,
            color='xkcd:lightgreen')
    #hist = hist * i_frac
    mids = 0.5*(bins[1:] + bins[:-1])
    mean = np.average(mids, weights=hist)
    ax1.axvline(x=mean/10**9,c='tab:purple',linestyle='--')
    # if inf_bool == True:
    #     print(name+' Inf: '+'{:.2f}'.format(mean/10**9)+' Gyr \n')
    # else:
    #     print(name+' Peri: '+'{:.2f}'.format(mean/10**9)+' Gyr \n')
    ax2 = ax1.twinx()
    rate_s2 = sav_gol(rate,9,3)
    ax2.scatter(t,rate,c='tab:olive',marker='s')
    ax2.plot(t,rate_s2,c='tab:orange',linestyle='-.')
    ax1.set_xlabel('Lookback time [Gyr]',fontsize=18)
    if inf_bool == True:
        ax1.set_ylabel('Infall PDF (normalized)',fontsize=18)
    else:
        ax1.set_ylabel('Pericenter PDF (normalized)',fontsize=18)
    ax1.legend(['<infall>','histogram'],loc=2,fontsize=16)
    ax2.set_ylabel(r'SFR [${\rm M}_\odot yr^{-1}$]',fontsize=18)
    ax2.legend(['SFR filtered','SFR'],loc=1,fontsize=16)
    ax1.grid(False)
    ax2.grid(False)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(axis='both',which='both',direction='in', bottom = True, top = True,
                    left = True, right = False,labelsize=18)
    ax2.tick_params(axis='both',which='both',direction='in', bottom = False, top = False,
                    left = False, right = True,labelsize=18)
    ax1.annotate('GMP '+sat_name, xy=(0.47, 0.95), xycoords='axes fraction',
                 fontsize=16)
    # if inf_bool == True:
    #     #plt.savefig('./obs_sim_plots/inf_sfr'+name+'.png',dpi=200)
    #     plt.savefig('./obs_sim_plots/inf_sfr'+name+'.pdf',dpi=500)
    # else:
    #     #plt.savefig('./obs_sim_plots/peri_sfr'+name+'.png',dpi=200)
    #     plt.savefig('./obs_sim_plots/peri_sfr'+name+'.pdf',dpi=500)
    #plt.show()
    return fig
    

def plot_all(ext,ssp):
    ###########################################################################
    ## Extracting Data for Plots ##
    ###########################################################################
    int_frac = np.array(pd.read_csv('./int_frac_files/'+ext+'_'+'int_frac.csv')['int_frac'])
    t_inf = pd.read_csv('./inf_peri_files/'+ext+'_'+'inf_time.csv')
    t_peri = pd.read_csv('./inf_peri_files/'+ext+'_'+'peri_time.csv')
    sfr = pd.read_csv('./sfr_ssfr_tables/'+ssp+'/corr_sfr.csv')

    ###########################################################################
    ## Plotting and saving in PDF ##
    ###########################################################################
    # Plotting t_inf and SFR
    out_dir = gen_out_dir(ext,ssp)
    pdf = PdfPages(out_dir+ext+'_'+ssp+'_'+'pdf_tinf_sfr.pdf')
    for i_frac,name in zip(int_frac,sat_names):
        t = np.array(sfr['Age_Gyr'])
        fig = plots(np.array(t_inf[name].dropna()),t,
                    np.array(sfr[name].dropna()),name,True,i_frac,t_inf,t_peri)
        pdf.savefig(fig,dpi=500)
        #print(name)
    pdf.close()

    # Plotting t_peri and SFR
    pdf = PdfPages(out_dir+ext+'_'+ssp+'_'+'pdf_tperi_sfr.pdf')
    for i_frac,name in zip(int_frac,sat_names):
        t = np.array(sfr['Age_Gyr'])
        fig = plots(np.array(t_peri[name].dropna()),t,
                    np.array(sfr[name].dropna()),name,False,i_frac,t_inf,t_peri)
        pdf.savefig(fig,dpi=500)
        #print(name)
    pdf.close()


###############################################################################
## Plotting PDFs of Infall and Pericenter vs SFR ##
###############################################################################
uncert_ext = ['Rvir_Ms','Rvir_Ms+','Rvir_Ms-','Rvir+_Ms','Rvir+_Ms+','Rvir+_Ms-',
          'Rvir-_Ms','Rvir-_Ms+','Rvir-_Ms-']
ssps = ['miles','bc03','phr']

# for ssp in ssps:
#     for ext in uncert:
#         plot_all(ext,ssp)

for ext in uncert_ext:
    ssp = ssps[2]
    plot_all(ext,ssp)
