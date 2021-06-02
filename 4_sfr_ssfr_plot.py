# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 22:56:44 2020

@author: amit
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.cm import ScalarMappable, viridis as cmap
from matplotlib.colors import Normalize
import matplotlib.lines as mlines
from astropy.cosmology import Planck13, z_at_value
#cosmo = FlatLambdaCDM(H0=67.77, Om0=0.3219) # Planck 2013 params
import astropy.units as u
# from scipy.signal import savgol_filter as sf
from scipy.interpolate import interp1d

# linestyles
linestyles = {
     'solid':                 (0, ()),
     'densely dotted':        (0, (1, 1)),
     'densely dashed':        (0, (5, 1)),
     'densely dashdotted':    (0, (3, 1, 1, 1)),
     'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))}

l1 = linestyles['densely dotted']
l2 = linestyles['densely dashed']
l3 = linestyles['densely dashdotted']
l4 = linestyles['densely dashdotdotted']
l5 = linestyles['solid']

## sat names
gmp_names = ['3254','3269', '3291', '3329', '3352', '3367', '3414', '3484',
             '3534', '3565', '3639', '3664']

# linestyles assigned to each galaxy
lsty = [l1, l3, l2, l4, l1, l2, l4, l4, l2, l1, l3, l5]

log_Ms = np.loadtxt('logMs_coma.m')
Ms = 10**log_Ms

log_Ms_df = pd.DataFrame()
log_Ms_df['GMP'] = gmp_names
log_Ms_df['log_Ms'] = log_Ms
log_Ms_df.set_index('GMP', inplace=True)

norm = Normalize(vmin=9, vmax=11)
smap = ScalarMappable(cmap=cmap, norm=norm)
smap.set_array([])

def tab_dir(ssp):
    return 'sfr_ssfr_tables/'+ssp+'/'

# def group_sat_names(lst,n):
#     return [lst[i:i + n] for i in range(0, len(lst), n)]

# def sav_gol(x,win,poly):
#     return sf(x, window_length=win, polyorder=poly, mode='interp')

# def sfr_interp(sfr_df):
#     age = np.array(sfr_df['Age_Gyr'])
#     age_extended_bins = np.linspace(age[0],age[-1],100)
#     sfr_interp = pd.DataFrame()
#     sfr_interp['Age_Gyr'] = age_extended_bins
#     for name in gmp_names:
#         sfr = np.array(sfr_df[name])
#         sfr_filtered = sav_gol(sfr,9,3)
#         f = interp1d(age,sfr_filtered,fill_value='extrapolate')
#         sfr_interp[name] = sav_gol(f(age_extended_bins),9,3)
#     return sfr_interp

def integrate(x, y):
    area = np.trapz(y=y, x=x)
    return area

def rel_sfr(lookback, sfr):
    area = integrate(lookback, sfr)
    norm_sfr = sfr / area
    # norm_sfr_filtered = sav_gol(norm_sfr,9,3)
    return norm_sfr

def z_at_given_lookback(lookback):
    age_of_universe = Planck13.age(0)
    ages = age_of_universe - lookback*u.Gyr
    z_lookback = [z_at_value(Planck13.age, age) for age in ages]
    return np.array(z_lookback)

def lookback_at_z(z):
    age_of_universe = Planck13.age(0)
    lookback = [age_of_universe.value - Planck13.age(red).value for red in z]
    return np.array(lookback)

# computes cosmic SFR for a given z as per Madau+14
def cosmic_sfr(z_arr):
    csfr = np.array([])
    for z in z_arr:
        sfr_z = 0.015 * ((1+z)**2.7 / (1 + ((1+z)/2.9)**5.6))
        csfr = np.append(csfr, sfr_z)
        
    return csfr


def plots(ssp):
    table_dir = tab_dir(ssp)
    #steckmap_sfr = pd.read_csv('steckmap_sfr_tables/miles/steckmap_sfr.csv')
    #corr_sfr = sfr_interp(steckmap_sfr)
    corr_sfr = pd.read_csv(table_dir+'corr_sfr.csv')
    #ssfr = pd.read_csv(table_dir+'ssfr.csv')
    
    # grps = group_sat_names(gmp_names,4)
    
    lookback = np.array(corr_sfr['Age_Gyr'])
    

    ## SFR Plot
    fig, ax = plt.subplots(figsize=(11,8))
    handles = []
    for name, ls in zip(gmp_names, lsty):
        if name == '3329':
            continue
        else:
            rel_s = rel_sfr(lookback, np.array(corr_sfr[name]))
            c = cmap(norm(log_Ms_df.loc[name]['log_Ms']))
            ax.plot(lookback, rel_s, color=c, linestyle=ls, linewidth=3)
            line = mlines.Line2D([], [], color=c, marker='None', 
                                 linestyle=ls, linewidth=3, 
                                 label='GMP '+name)
            handles.append(line)

    ## plotting cosmic SFR from Madau+14
    # zl = z_at_given_lookback(lookback)
    # c_sfr = cosmic_sfr(zl)
    
    # ax3 = ax.twinx()
    # ax3.plot(lookback, c_sfr, color='r', linestyle='-', linewidth=4)
    
    # madau_line = mlines.Line2D([], [], color='r', marker='None', linestyle='-', 
    #                            linewidth=4, label='Madau+14')
    
    ## plotting exp SFR decay
    # ageU = Planck13.age(0).value
    lb = np.linspace(0.,lookback[-1],138)
    sfr01 = np.exp(0.1*lb) * (1/np.exp(0.1*lb[-1]))
    sfr03 = np.exp(0.3*lb) * (1/np.exp(0.3*lb[-1]))
    ax3 = ax.twinx()
    ax3.plot(lb, sfr03, color='r', linestyle='-', linewidth=1)
    ax3.plot(lb, sfr01, color='b', linestyle='-', linewidth=1)
    
    line01 = mlines.Line2D([], [], color='b', marker='None', linestyle='-', 
                                linewidth=1, 
                                label=r'$\tau=0.1\,\mathrm{Gyr}^{-1}$')
    handles1 = []
    
    handles1.append(line01)
    
    line03 = mlines.Line2D([], [], color='r', marker='None', linestyle='-', 
                                linewidth=1, 
                                label=r'$\tau=0.3\,\mathrm{Gyr}^{-1}$')
    
    handles1.append(line03)
    
    
    ## setting limits and axes lables
    ax.set_xlim(0.,max(lookback))
    xtks = [2., 4., 6., 8., 10., 12.]
    ax.set_xticks(xtks)
    ax.set_ylim(0.,0.45)
    ytks = [0., 0.1, 0.2, 0.3, 0.4]
    ax.set_yticks(ytks)
    
    ax3.set_ylim(0.,1.0)
    ax3.set_yticks([0., 0.2, 0.4, 0.6, 0.8, 1.0])
    
    ## adding 'z' redshift axes
    ax2 = ax.twiny()
    z = [0.2, 0.5, 1.0, 2.0, 7.0] # redshift values to be shown on axes
    # corresponding lookback time
    lookback_z = lookback_at_z(z)
    # relative postions of x-axis ticks
    x_min, x_max = ax.get_xlim()
    xtks_pos = [(tick - x_min)/(x_max - x_min) for tick in ax.get_xticks()]
    f = interp1d(xtks, xtks_pos, fill_value='extrapolate')
    # relative positions of z 
    ztks = f(lookback_z)
    # set ax2 tick values
    ax2.set_xticks(ztks)
    ax2.set_xticklabels(z)
    
    ax.set_xlabel('Lookback time [Gyr]',fontsize=20)
    ax.set_ylabel(r'rel. SFR [$\mathrm{Gyr}^{-1}$]',fontsize=20)
    ax2.set_xlabel(r'$z$',fontsize=20)
    #ax3.set_ylabel(r'SFR [$\mathrm{M}_\odot\,\mathrm{yr}^{-1}\,\mathrm{Mpc}^{-3}$]',fontsize=18)
    #ax3.set_ylabel(r'rel. SFR [$\mathrm{Gyr}^{-1}$]',fontsize=18)
    ax3.set_ylabel('Fraction of stars formed in the time interval',fontsize=20)
    
    ax.legend(ncol=2, handles=handles, frameon=True, framealpha=1.0, 
              edgecolor='k', fontsize=14, loc=2, bbox_to_anchor=(0.03,0.97), 
              bbox_transform=ax.transAxes)
    
    ax2.legend(ncol=1, handles=handles1, frameon=True, framealpha=1.0, 
              edgecolor='k', fontsize=14, loc=3, bbox_to_anchor=(0.03,0.45), 
              bbox_transform=ax2.transAxes)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both',which='major',direction='in', bottom = True, 
                   top = False, left = True, right = False,  length=10,
                   labelsize=18)
    ax.tick_params(axis='both',which='minor',direction='in', bottom = True, 
                   top = False, left = True, right = False,  length=5,
                   labelsize=18)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.tick_params(axis='both',which='major',direction='in', bottom = False, 
                   top = True, left = False, right = False,  length=10,
                   labelsize=18)
    ax2.tick_params(axis='both',which='minor',direction='in', bottom = False, 
                   top = True, left = False, right = False,  length=5,
                   labelsize=18)
    # ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    ax3.tick_params(axis='y',which='major',direction='in', bottom = False, 
                    top = False, left = False, right = True,  length=10,
                    labelsize=18)
    ax3.tick_params(axis='y',which='minor',direction='in', bottom = False, 
                    top = False, left = False, right = True,  length=5,
                    labelsize=18)
    
    cbar = fig.colorbar(smap, ticks=[9., 9.5, 10.0, 10.5, 11.0], pad=0.10)
    cbar.set_label(r'$\log(M_\star/\mathrm{M}_\odot)$',fontsize=20)
    cbar.ax.tick_params(axis='y', direction='in',  length=10, labelsize=18)
    plt.savefig(table_dir+'sfr_plot.pdf',dpi=500)
    
    
    ## SSFR Plot
    # fig, ax = plt.subplots(figsize=(8,6))
    # for grp, lsty in zip(grps, linestyles.keys()):
    #     for name in grp:
    #         if name == '3329':
    #             continue
    #         else:
    #             c = cmap(norm(log_Ms_df.loc[name]['log_Ms']))
    #             ax.plot(lookback, np.log10(np.array(ssfr[name])), color=c, 
    #                         linestyle=linestyles[lsty], linewidth=3, 
    #                         label='GMP '+name)

    # xaxis_labels = np.arange(0.5,14.5,2.5)
    # z_x = z_at_given_lookback(xaxis_labels)
    # z_xaxis_labels = [round(val,2) for val in z_x]
    # ax2 = ax.twiny()
    # ax.set_xticks(xaxis_labels)
    # ax.set_xticklabels(xaxis_labels)
    # ax2.set_xlim(ax.get_xlim())
    # ax2.set_xticks(xaxis_labels)
    # ax2.set_xticklabels(z_xaxis_labels)
    
    # ax.set_xlabel('Lookback time [Gyr]',fontsize=18)
    # ax.set_ylabel(r'$\log_{10}\mathrm{SSFR}\,\mathrm{[}\mathrm{yr}^{-1}\mathrm{]}$',
    #               fontsize=18)
    # ax2.set_xlabel(r'$z$',fontsize=18)
    
    # ax.legend(frameon=True, framealpha=1.0, edgecolor='k', fontsize=12, 
    #           loc=4, bbox_to_anchor=(0.98,0.02), bbox_transform=ax.transAxes)
    
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # ax.tick_params(axis='both',which='both',direction='in', bottom = True, 
    #                top = True, left = True, right = True, labelsize=18)
    # ax2.xaxis.set_minor_locator(AutoMinorLocator())
    # ax2.yaxis.set_minor_locator(AutoMinorLocator())
    # ax2.tick_params(axis='both',which='both',direction='in', bottom = False, 
    #                top = True, left = False, right = False, labelsize=18)
    
    # cbar = fig.colorbar(smap, ticks=[9., 9.5, 10.0, 10.5, 11.0], pad=0.01)
    # cbar.set_label(r'$\log_{10}(M_\star/{\rm M}_\odot)$',fontsize=18)
    # cbar.ax.tick_params(axis='y', direction='in',labelsize=18)
    # plt.savefig(table_dir+'ssfr_plot.pdf',dpi=500)

#ssps = ['miles','bc03','phr']
ssps = ['miles']

for ssp in ssps:
    plots(ssp)
