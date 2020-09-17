import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.cm import ScalarMappable, viridis as cmap
from matplotlib.colors import Normalize
from scipy import interpolate
from matplotlib.ticker import AutoMinorLocator

gmp_names = ['3254','3269', '3291', '3329', '3352', '3367', '3414', '3484',
             '3534', '3565', '3639', '3664']

log_Ms = np.loadtxt('logMs_coma.m')
Ms = 10**log_Ms


int_frac = np.array(pd.read_csv('int_frac_mmax.csv')['int_frac'])

norm = Normalize(vmin=9, vmax=11)
smap = ScalarMappable(cmap=cmap, norm=norm)
smap.set_array([])

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
    # time_cdf = (1-time_cdf) * (1-i_frac)
    # time_cdf = 1-time_cdf
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
    return f50, g50, f16, f84, g16, g84, xerr, yerr

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


# plotting all error bars for both pericenter time and infall time
fig, ax = plt.subplots(figsize=(10,7))
for name,log_Mi,i_frac in zip(gmp_names,log_Ms,int_frac):
    if name == '3329':
        continue
    else:
        f50i,g50i,f16i,f84i,g16i,g84i,xerri,yerri=err_plots(name,'inf',i_frac)
        c = cmap(norm(log_Mi))
        xerri = [[j/10**9 for j in i] for i in xerri] # to put x error in Gyrs
        ax.errorbar(f50i/10**9,g50i,xerr=xerri,yerr=yerri,elinewidth=1, 
                capsize=5, ecolor=c, marker='o', mec=c, mfc=c,markersize=8)
        # ax.annotate(name, (f50i/10**9,g50i), xytext=(-30, 5), 
        #             textcoords='offset points', color='b', fontsize=14)
        #print('logMs: '+str(log_Mi)+'\n')
        print(name+' Inf: '+'{:.2f}'.format(f50i/10**9)+' Gyr')
        print('f16i: '+'{:.2f}'.format(f16i/10**9)+' Gyr')
        print('f84i: '+'{:.2f}'.format(f84i/10**9)+' Gyr')
        print('f16i-f50i: '+'{:.2f}'.format(abs(f16i-f50i)/10**9)+' Gyr')
        print('f50i-f84i: '+'{:.2f}'.format(abs(f50i-f84i)/10**9)+' Gyr')
        print('g50i: '+'{:.2f}'.format(g50i*100))
        print('g16i: '+'{:.2f}'.format(g16i*100))
        print('g84i: '+'{:.2f}'.format(g84i*100))
        print('g50i-g84i: '+'{:.2f}'.format(abs(g84i-g50i)*100))
        print('g16i-g50i: '+'{:.2f}'.format(abs(g16i-g50i)*100))
        #print('xerri'+'{:.2f}'.format(xerri/10**9))
        #print('yerri'+'{:.2f}'.format(yerri))
        print('\n')
for name,log_Mi in zip(gmp_names,log_Ms):
    if name == '3329':
        continue
    else:
        f50p,g50p,f16p,f84p,g16p,g84p,xerrp,yerrp=err_plots(name,'peri',i_frac)
        c = cmap(norm(log_Mi))
        xerrp = [[j/10**9 for j in i] for i in xerrp]
        ax.errorbar(f50p/10**9,g50p,xerr=xerrp,yerr=yerrp,elinewidth=1, 
                    capsize=5, ecolor=c, marker='*', mec=c, mfc=c,markersize=10)
        print(name+' Peri: '+'{:.2f}'.format(f50p/10**9)+' Gyr')
        print('f16p: '+'{:.2f}'.format(f16p/10**9)+' Gyr')
        print('f84p: '+'{:.2f}'.format(f84p/10**9)+' Gyr')
        print('f16p-f50p: '+'{:.2f}'.format(abs(f16p-f50p)/10**9)+' Gyr')
        print('f50p-f84p: '+'{:.2f}'.format(abs(f84p-f50p)/10**9)+' Gyr')
        print('g50p: '+'{:.2f}'.format(g50p*100))
        print('g16p: '+'{:.2f}'.format(g16p*100))
        print('g84p: '+'{:.2f}'.format(g84p*100))
        print('g50p-g84p: '+'{:.2f}'.format(abs(g84p-g50p)*100))
        print('g16p-g50p: '+'{:.2f}'.format(abs(g16p-g50p)*100))
        #print('xerrp'+'{:.2f}'.format(xerrp/10**9))
        #print('yerrp'+'{:.2f}'.format(yerrp))
        print('\n')
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
plt.savefig('cum_Ms_at_exp_orb_time.pdf',dpi=500)
#plt.savefig('ebar_inf_peri.png',dpi=200)

#

mcdf_df = pd.DataFrame()
for name in gmp_names:
    bins,bin_edges,delta_bin_edges, bin_edges_neg = bin_gen(name,'inf')
    mcdf,m = cum_mass(name,bin_edges,delta_bin_edges)
    mcdf_df[name] = mcdf
    
mcdf_df.to_csv('Ms_cdf.csv',index=False)
