# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 21:27:55 2019

@author: Amit
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import pandas as pd
from matplotlib.ticker import AutoMinorLocator

###############################################################################
# creating directory for saving output files
###############################################################################
out_dir = './out_files_steckmap/phr/'
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir)

###############################################################################
# making list of STECKMAP output data file names
###############################################################################
datafiles = [os.path.join(root, name)
             for root, dirs, files in os.walk("./steckmap_out/miles_out/")
             for name in files
             if name.endswith(".txt")]

nFiles = len(datafiles)
#print(datafiles)
###############################################################################
# Finding number of entries
###############################################################################
nEntries = np.array([])
for i in range(nFiles):
    f = open(datafiles[i],'r')
    firstLine = f.readline()
    firstLineTxt = firstLine.rstrip().split(" ")
    ## note: use commented code below only for 'PHR' case
    # if i==5 or i==11 or i==17 or i==23 or i==29 or i==35 or i==41 or i==47 or i==53 or i==59 or i==65 or i==71:
    #     nE=firstLineTxt[7]
    # else:
    #     if firstLineTxt[8]=='':
    #         nE=firstLineTxt[9]
    #     else:
    #         nE=firstLineTxt[8]
    ## note: comment the next 4 lines of code for 'PHR' case
    if firstLineTxt[8]=='':
        nE=firstLineTxt[9]
    else:
        nE=firstLineTxt[8]
    nEntries = np.append(nEntries,float(nE))
    f.close()

###############################################################################
# title and labels
###############################################################################
title = ['Age Metallicty Relation', 'Line of Sight Velocity Distribution',
          'Stellar Mass', 'Stellar Age Distribution', 'Star Formation Rate', 
          'Spectrum']
xlabel = [r'$\log_{10}Age \: \rm [Myr]$', r'$v \: \rm [kms^{-1}]$', 
          r'$\log_{10}Age \: \rm [Myr]$', r'$\log_{10}Age \: \rm [Myr]$',
          r'$\log_{10}Age \: \rm [Myr]$', r'$Wavelength \: \rm [\AA]$']
ylabel = ['Metallicity (Z)', 'BF [g(v)]', r'$M_\star \: \rm [M_\odot]$', 
          'SAD (normalized)', r'$SFR \: \rm [M_\odot yr^{-1}]$', 'Spectra']

titles = np.array([])
xlabels = np.array([])
ylabels = np.array([])
idx = int(nFiles/6)
for i in range(idx):
    titles = np.append(titles,title)
    xlabels = np.append(xlabels,xlabel)
    ylabels = np.append(ylabels,ylabel)
    
###############################################################################
# output file names
###############################################################################
filenames = np.array([])
for i in range(nFiles):
    df = str(datafiles[i])
    temp1 = df.split('-')
    temp2 = temp1[1].split('.')
    temp3 = temp1[0].split('/')
    temp4 = temp3[3].split('_')
    filename = temp4[0]+'_'+temp2[0]
    filenames = np.append(filenames,filename)
    
###############################################################################
# Reading the data
###############################################################################
data_x = np.array([])
data_y = np.array([])
for i in range(nFiles):
    data=np.array([])
    f = open(datafiles[i],'r')
    lines = f.readlines()
    for line in lines:
        eachLine = np.fromstring(line, dtype=float, sep=' ')
        data=np.append(data,eachLine)
    N = int(nEntries[i])
    xaxis = data[1:N+1]
    if (str(datafiles[i]).find('spectra') != -1):
        yaxis1 = data[N+2:2*N+2]
        yaxis2 = data[2*N+3:3*N+3]
        yaxis3 = data[3*N+4:4*N+4]
        yaxis4 = data[4*N+5:]
        yaxis = np.concatenate((yaxis1,yaxis2,yaxis3,yaxis4))
    else:
        yaxis = data[N+2:]
        
    data_x = np.append(data_x,xaxis)
    data_y = np.append(data_y,yaxis)
    f.close()
       
###############################################################################
# Plotting the data
###############################################################################
xcount = int(0)
ycount = int(0)
legend1 = ['Data', 'Best Fitting Model', 'non-param adjusted Continuum']
legend2 = ['Scatter', 'Plot']
for i in range(nFiles):
    fig, ax = plt.subplots(figsize=(6,4))
    print(str(int(i/nFiles*100))+' %')
    n = int(nEntries[i])
       
    xaxis = data_x[xcount:xcount+n]
    
    if (str(datafiles[i]).find('spectra') != -1):
        yaxis = data_y[ycount:ycount+4*n]
    else:
        yaxis = data_y[ycount:ycount+n]
            
    if (str(datafiles[i]).find('LOSVD') != -1):
        ax.plot(xaxis,yaxis,'b-')
    elif (str(datafiles[i]).find('spectra') != -1):
        yaxis1 = yaxis[:n]
        yaxis2 = yaxis[n:2*n]
        yaxis3 = yaxis[2*n:3*n]
        yaxis4 = yaxis[3*n:4*n]
        ax.plot(xaxis, yaxis1, 'r', xaxis, yaxis2, 'g--', xaxis, yaxis4, 'b--')
        ax.legend(legend1)
    else:
        ax.plot(np.log10(xaxis),yaxis,'ro',np.log10(xaxis),yaxis,'b-')
        #plt.hist(yaxis,bins=len(xaxis),density=True,histtype='bar',color='y')
        #plt.bar(np.log10(xaxis),yaxis,width=5,align='center',color='y',edgecolor='y',
                #linewidth=12)
        #plt.legend(legend2)
        
    #plt.title(titles[i])
    if (str(datafiles[i]).find('SFR') != -1):
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    elif (str(datafiles[i]).find('MASS') != -1):
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    else:
        ax.ticklabel_format(axis="y", style="plain")
    ax.set_xlabel(xlabels[i],fontsize=18)
    ax.set_ylabel(ylabels[i],fontsize=18)
    ax.grid(False)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both', which='both', direction='in', bottom = True, 
                    top = True, left = True, right = True, labelsize=18)
    plt.tight_layout()
#    plt.show()
    #plt.savefig('./plots_500dpi/'+filenames[i]+'.png',dpi=500)
    plt.savefig(out_dir+filenames[i]+'.pdf',dpi=500)
    plt.clf()
    
    if (str(datafiles[i]).find('spectra') != -1):
        yaxis1 = yaxis[:n]
        yaxis2 = yaxis[n:2*n]
        yaxis3 = yaxis[2*n:3*n]
        yaxis4 = yaxis[3*n:4*n]
        out_dat = pd.DataFrame({'Wavelength':xaxis, 'Spectra':yaxis1,
                                'Best_Fitting_Model':yaxis2,
                                'Non_parm_Extinction_Curve': yaxis3,
                                'Non_param_Adjusted_Continuum': yaxis4})
        out_dat.to_csv(out_dir+filenames[i]+'.dat',index=None)
        #np.savetxt('./out_files/'+filenames[i]+'.dat', np.c_[xaxis,yaxis1,
        #          yaxis2,yaxis3,yaxis4])
    elif (str(datafiles[i]).find('LOSVD') != -1):
        out_dat = pd.DataFrame({'v (km/s)':xaxis, 'BF[g(v)]':yaxis})
        out_dat.to_csv(out_dir+filenames[i]+'.dat',index=None,sep=' ')
    else:
        out_dat = pd.DataFrame({'log[Age(Myr)]':np.log10(xaxis), 
                                     filenames[i]:yaxis})
        out_dat.to_csv(out_dir+filenames[i]+'.dat',index=None,sep=' ')
        #np.savetxt('./out_files/'+filenames[i]+'.dat', np.c_[xaxis,yaxis])
    
    xcount = xcount + n
    
    if (str(datafiles[i]).find('spectra') != -1):
        ycount = ycount + 4*n
    else:
        ycount = ycount + n
    