# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 07:22:32 2020

@author: amit
"""


import os
import shutil
import numpy as np
import pandas as pd
#from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages

# gmp_names = ['3254','3269', '3291', '3352', '3367', '3414', '3484',
#              '3534', '3565', '3639', '3664']


#gmp_names = ['3254']

name = '3254'
ssp = 'miles'
ext = 'Rvir_Ms'

def gen_out_dir(ext,ssp):
    out_dir = './cdf_inf_peri_vs_cumMs_files/'+ssp+'/'+ext+'/'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    return out_dir

def cdf_sfr(sfr_bins, sfr):
    mis = np.array([])
    for i in range(len(sfr_bins)-1):
        mi = 0.5 * (sfr[i+1] + sfr[i]) * (sfr_bins[i+1] - sfr_bins[i])
        mis = np.append(mis, mi)
    sb = sfr_bins[:-1]
    f = interp1d(sb, mis, fill_value='extrapolate')
    ms = f(sfr_bins)
    m_cumsum = np.cumsum(ms)
    min_m_cumsum = min(m_cumsum)
    minmax_m_cumsum = max(m_cumsum) - min(m_cumsum)
    m_cdf = 1 - ((m_cumsum - min_m_cumsum) / minmax_m_cumsum)
    return m_cdf

def cdf_time(time_bins, time, int_frac):
    

int_frac = np.array(pd.read_csv('./int_frac_files/'+ext+'_'+'int_frac.csv')['int_frac'])
sfr = pd.read_csv('./sfr_ssfr_tables/'+ssp+'/corr_sfr.csv')
t_inf = pd.read_csv('./inf_peri_files/'+ext+'_'+'inf_time.csv')
t_peri = pd.read_csv('./inf_peri_files/'+ext+'_'+'peri_time.csv')

time = np.array(t_peri[name].dropna())
be = np.histogram_bin_edges(time, bins='rice')
h, be = np.histogram(time, bins=be, density=True)
#integ = cumtrapz(h,be[:-1])
#print(integ)
a = sfr['Age_Gyr']
s = sfr[name]
f = interp1d(a, s, fill_value='extrapolate')
b = be[np.where(be>=0.5)[0]]
s_b = f(b)

del_be = be[1]-be[0]

mscdf = cdf_sfr(b,s_b)
