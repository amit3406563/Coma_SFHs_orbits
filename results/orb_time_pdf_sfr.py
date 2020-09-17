import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as sf
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_pdf import PdfPages

###############################################################################
## Functions ##
###############################################################################
## GMP IDs of satellites
sat_names = ['3254','3269', '3291', '3352', '3367', '3414', '3484',
             '3534', '3565', '3639', '3664']


int_frac = np.array(pd.read_csv('int_frac_mmax.csv')['int_frac'])

def sav_gol(x,win,poly):
    return sf(x, window_length=win, polyorder=poly, mode='interp')


def hist_time(name,bins,time_bool):
    if time_bool == True:
        time = t_inf[name].dropna()*10**9
    else:
        time = t_peri[name].dropna()*10**9
    time_b = pd.cut(time,bins=bins)
    time_count = time_b.value_counts(sort=False)
    time_hist = np.array(time_count).astype(float)/len(np.array(time))
    return time_hist

def bin_gen(name,time_bool):
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

def plots(pdf,t,rate,sat_name,inf_bool,i_frac):
    fig, ax1 = plt.subplots(figsize=(10,7))
    bins, bin_edges, delta_bin_edges, bin_edges_neg = bin_gen(sat_name,inf_bool)
    hist = hist_time(sat_name,bins,inf_bool)
    ax1.bar(bins[1:]/10**9,hist,align='center',width=0.9*delta_bin_edges/10**9,
            color='xkcd:lightgreen')
    #hist = hist * i_frac
    mids = 0.5*(bins[1:] + bins[:-1])
    mean = np.average(mids, weights=hist)
    ax1.axvline(x=mean/10**9,c='tab:purple',linestyle='--')
    if inf_bool == True:
        print(name+' Inf: '+'{:.2f}'.format(mean/10**9)+' Gyr \n')
    else:
        print(name+' Peri: '+'{:.2f}'.format(mean/10**9)+' Gyr \n')
    ax2 = ax1.twinx()
    rate_s2 = sav_gol(rate,9,3)
    ax2.scatter(t,rate,c='tab:olive',marker='s')
    ax2.plot(t,rate_s2,c='tab:orange',linestyle='-.')
    ax1.set_xlabel('Lookback time [Gyr]',fontsize=18)
    if inf_bool == True:
        ax1.set_ylabel('Infall PDF (normalized)',fontsize=18)
    else:
        ax1.set_ylabel('Pericenter PDF (normalized)',fontsize=18)
    ax1.legend(['expected','histogram'],loc=2,fontsize=16)
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
    ax1.annotate('GMP '+name, xy=(0.47, 0.95), xycoords='axes fraction',
                 fontsize=16)
    if inf_bool == True:
        #plt.savefig('./obs_sim_plots/inf_sfr'+name+'.png',dpi=200)
        plt.savefig('./obs_sim_plots/inf_sfr'+name+'.pdf',dpi=500)
    else:
        #plt.savefig('./obs_sim_plots/peri_sfr'+name+'.png',dpi=200)
        plt.savefig('./obs_sim_plots/peri_sfr'+name+'.pdf',dpi=500)
    #plt.show()
    return fig
    


###############################################################################
## Extracting Data for Plots ##
###############################################################################
t_inf = pd.read_csv('inf_time_mmax.csv')
t_peri = pd.read_csv('peri_time_mmax.csv')
sfr = pd.read_csv('corr_sfr.csv')


###############################################################################
## Plotting and saving in PDF ##
###############################################################################
# Plotting t_inf and SFR
pdf = PdfPages('tinf_sfr.pdf')
for i_frac,name in zip(int_frac,sat_names):
    t = np.array(sfr['Age_Gyr'])
    fig = plots(np.array(t_inf[name].dropna()),t,
                np.array(sfr[name].dropna()),name,True,i_frac)
    pdf.savefig(fig,dpi=500)
    #print(name)
pdf.close()



# Plotting t_peri and SFR
pdf = PdfPages('tperi_sfr.pdf')
for i_frac,name in zip(int_frac,sat_names):
    t = np.array(sfr['Age_Gyr'])
    fig = plots(np.array(t_peri[name].dropna()),t,
                np.array(sfr[name].dropna()),name,False,i_frac)
    pdf.savefig(fig,dpi=500)
    #print(name)
pdf.close()
