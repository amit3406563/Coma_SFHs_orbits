# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 04:35:44 2020

@author: amit
"""

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

# gmp_names = ['3254','3269', '3291', '3352', '3367', '3414', '3484',
#               '3534', '3565', '3639', '3664']

gmp_names = ['3254']

def gen_out_dir(ext,ssp):
    out_dir = './pdf_inf_peri_vs_sfr_files/'+ssp+'/'+ext+'/'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    return out_dir

def integrate(x, y):
    area = np.trapz(y=y, x=x)
    return area

def rel_sfr(lookback, sfr):
    area = integrate(lookback, sfr)
    norm_sfr = sfr / area
    return norm_sfr

def plotting(time_bool,t_inf,t_peri,sfr,name):
    if time_bool == 'inf':
        time = np.array(t_inf[name].dropna())
    else:
        time = np.array(t_peri[name].dropna())
    
    if time_bool == 'inf':
        be = np.linspace(0.,ageU,20)
        del_be = be[1] - be[0]
    else:
        be = np.linspace(min(time),ageU,14)
        del_be = be[1] - be[0]
    
    # be = np.histogram_bin_edges(time, bins='rice')
    h, be = np.histogram(time, bins=be, density=True)
    #integ = cumtrapz(h,be[:-1])
    #print(integ)
    a = sfr['Age_Gyr']
    s = sfr[name]
    f = interp1d(a, s, fill_value='extrapolate')
    b = be[np.where(be>=0.)[0]]
    s_b = f(b)
    s_r = rel_sfr(b, s_b)
    med = np.median(time)
        
    fig, ax = plt.subplots(figsize=(10,7))
    ax.bar(be[1:], h, color='tab:gray', width=1.*del_be)
    ax.axvline(x=med, c='k', linestyle='--')
    ax1 = ax.twinx()
    ax1.plot(b, s_r, c='tab:red', linestyle='-', marker='s', markersize=10)

    ax.set_xlabel('Lookback time [Gyr]',fontsize=18)
    if time_bool == 'inf':
        ax.set_ylabel('Infall PDF (normalized)',fontsize=18)
    else:
        ax.set_ylabel('Pericenter PDF (normalized)',fontsize=18)
    ax1.set_ylabel(r'rel. SFR [$\mathrm{Gyr}^{-1}$]',fontsize=18)
    if time_bool == 'inf':
        bar_line = mlines.Line2D([], [], color='tab:gray', marker='None', 
                                 linestyle='-', linewidth=10, 
                                 label=r'$t_{\mathrm{inf}}$ PDF')
        line1 = mlines.Line2D([], [], color='k', marker='None', linestyle='--', 
                              linewidth=2, label=r'<$t_{\mathrm{inf}}$>')
    else:
        bar_line = mlines.Line2D([], [], color='tab:gray', marker='None', 
                                 linestyle='-', linewidth=10, 
                                 label=r'$t_{\mathrm{peri}}$ PDF')
        line1 = mlines.Line2D([], [], color='k', marker='None', linestyle='--', 
                              linewidth=2, label=r'<$t_{\mathrm{peri}}$>')
    line2 = mlines.Line2D([], [], color='tab:red', marker='s', markersize=10, 
                          linestyle='-', linewidth=2, label='SFR')
    ax.legend(handles=[bar_line,line1,line2], frameon=True, framealpha=1.0, 
              edgecolor='k', fontsize=14, loc=2, 
              bbox_to_anchor=(0.03,0.97), bbox_transform=ax.transAxes)
    ax.grid(False)
    ax1.grid(False)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both',which='major',direction='in', bottom = True, 
                   top = True, left = True, right = False, length=10, 
                   labelsize=18)
    ax.tick_params(axis='both',which='minor',direction='in', bottom = True, 
                   top = True, left = True, right = False, length=5, 
                   labelsize=18)
    ax1.tick_params(axis='both',which='major',direction='in', bottom = False, 
                    top = False, left = False, right = True, length=10, 
                    labelsize=18)
    ax1.tick_params(axis='both',which='minor',direction='in', bottom = False, 
                    top = False, left = False, right = True, length=5, 
                    labelsize=18)
    
    return fig

def plots(ssp,ext):
    t_inf = pd.read_csv('./inf_peri_files/'+ext+'_'+'inf_time.csv')
    t_peri = pd.read_csv('./inf_peri_files/'+ext+'_'+'peri_time.csv')
    sfr = pd.read_csv('./sfr_ssfr_tables/'+ssp+'/corr_sfr.csv')
    
    out_dir = gen_out_dir(ext,ssp)
    
    pdf = PdfPages(out_dir+ext+'_'+ssp+'_'+'pdf_tinf_sfr.pdf')
    for name in gmp_names:
        fig = plotting('inf', t_inf, t_peri, sfr, name)
        pdf.savefig(fig,dpi=500)
    pdf.close()
    
    pdf = PdfPages(out_dir+ext+'_'+ssp+'_'+'pdf_tperi_sfr.pdf')
    for name in gmp_names:
        fig = plotting('peri', t_inf, t_peri, sfr, name)
        pdf.savefig(fig,dpi=500)
    pdf.close()

uncert_ext = ['Rvir_Ms','Rvir_Ms+','Rvir_Ms-','Rvir+_Ms','Rvir-_Ms']
ssps = ['miles','bc03','phr']
# ssps = ['miles']
# uncert_ext = ['Rvir_Ms']
for ssp in ssps:
    for ext in uncert_ext:
        plots(ssp,ext)    