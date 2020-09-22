# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:41:43 2020

@author: Amit
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

sat_names = ['3254','3269', '3291', '3329', '3352', '3367', '3414', '3484',
             '3534', '3565', '3639', '3664']

def list_files(path,key):
    unsorted_datafiles = [os.path.join(root, name)
            for root, dirs, files in os.walk(path)
            for name in files
            if name.endswith(key)]
    datafiles = sorted(unsorted_datafiles)
    return datafiles

df_list = list_files("./out_files","spectra.dat")

    
fig, axes = plt.subplots(nrows=12, ncols=1,figsize=(13,13))
fig.subplots_adjust(hspace=0)

for ax, file, name in zip(axes.flatten(), df_list, sat_names):
    df = pd.read_csv(file)
    x = df['Wavelength']
    y1 = df['Spectra']
    y2 = df['Best_Fitting_Model']
    y4 = df['Non_param_Adjusted_Continuum']
    ax.plot(x,y1,'r-',x,y2,'g--',x,y4,'b:')
    ax.text(5600,1.2,name)
    ax.grid(False)
    ax.set_yticks([0.5,1.0,1.5])
    ax.tick_params(axis='both',direction='in', bottom = True, top = True,
               left = True, right = True)
    ax.set_facecolor('white')


plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1
plt.ylabel('Spectra',fontsize=18)
plt.xlabel(r'Wavelength [$\rm \AA$]',fontsize=18)
#plt.savefig('spectra_plot.png',dpi=500)
plt.savefig('spectra_plot.pdf',dpi=500)