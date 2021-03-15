# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 07:31:54 2021

@author: amit
"""


import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages

df = pd.read_csv('./cumMs_at_inf_peri/miles/Rvir_Ms/Rvir_Ms_miles_cum_Ms_at_exp_orb_time_table.csv')

gmp = df['GMP'].tolist()

ssfr_avg = 10**(-11) # yr^-1

sfr_mw = 0.68 # M_solar yr^-1


def comp_tq_ssfr_avg(ms_perc):
    ms_left = 1 - ms_perc/100
    tq = (ms_left / ssfr_avg) / 10**9
    return tq

def comp_tq_sfr_mw(log_ms,ms_perc):
    ms_left = 1 - ms_perc/100
    ms = 10**log_ms * ms_left
    tq = (ms / sfr_mw) / 10**9
    return tq

df_new = pd.DataFrame()

df_new['GMP'] = gmp

df_sub = df[['%Ms_tperi','%Ms_tperi-']].apply(lambda x: x['%Ms_tperi']-x['%Ms_tperi-'], axis=1)

df_sub1 = pd.concat([df['log_Ms'],df_sub], axis=1)
df_sub1.columns = ['log_Ms','%Ms_tperi-']

tq_ssfr_avg = df['%Ms_tperi'].apply(comp_tq_ssfr_avg)
# tq_ssfr_avg_p = df['%Ms_tperi+'].apply(comp_tq_ssfr_avg)
tq_ssfr_avg_m = df_sub.apply(comp_tq_ssfr_avg)

tq_sfr_mw = df[['log_Ms','%Ms_tperi']].apply(lambda x: comp_tq_sfr_mw(x['log_Ms'],x['%Ms_tperi']), axis=1)
# tq_sfr_mw_p = df[['log_Ms','%Ms_tperi+']].apply(lambda x: comp_tq_sfr_mw(x['log_Ms'],x['%Ms_tperi+']), axis=1)
tq_sfr_mw_m = df_sub1.apply(lambda x: comp_tq_sfr_mw(x['log_Ms'],x['%Ms_tperi-']), axis=1)

df_new['tq_ssfr_avg'] = tq_ssfr_avg
# df_new['tq_ssfr_avg+'] = tq_ssfr_avg_p
df_new['tq_ssfr_avg-'] = tq_ssfr_avg_m

df_new['tq_sfr_mw'] = tq_sfr_mw
# df_new['tq_sfr_mw+'] = tq_sfr_mw_p
df_new['tq_sfr_mw-'] = tq_sfr_mw_m

df_new.iloc[:,1:] = df_new.iloc[:,1:].round(decimals=2)

fig, ax = plt.subplots(figsize=(4,4))
    #ax.axis('tight')
ax.axis('off')
tab = ax.table(cellText=df_new.values,colLabels=df_new.columns,loc='center')
tab.auto_set_font_size(False)
tab.set_fontsize(6)
tab.auto_set_column_width(col=list(range(len(df_new.columns))))
#pp = PdfPages('tq_table.pdf')
#pp.savefig(fig, bbox_inches='tight')
#pp.close()