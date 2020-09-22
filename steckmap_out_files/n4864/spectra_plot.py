# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 12:22:27 2019

@author: Amit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n=3737
nEntriesPerLine=5
nLines=n/nEntriesPerLine
data=np.array([])
with open('n4864_1D_2p7_flux.res-spectra.txt','r') as f:
    lines=f.readlines()
    for line in lines:
        eachLine = np.fromstring(line, dtype=float, sep=' ')
        data=np.append(data,eachLine)
        
xaxis=data[1:n+1]
print(xaxis)
print(len(xaxis))
yaxis1=data[n+2:2*n+2]
print(yaxis1)
print(len(yaxis1))
yaxis2=data[2*n+3:3*n+3]
print(yaxis2)
print(len(yaxis2))
yaxis3=data[3*n+4:4*n+4]
print(yaxis3)
print(len(yaxis3))
yaxis4=data[4*n+5:]
print(yaxis4)
print(len(yaxis4))

data = {'Wavelength':xaxis,'Data':yaxis1,'Best Fitting Model':yaxis2,
        'non-param adjusted Continuum':yaxis4}
df = pd.DataFrame(data)
df.to_csv('spectra_3664.csv',index=False)

legend = ['Data', 'Best Fitting Model', 'non-param adjusted Continuum']
plt.plot(xaxis,yaxis1,'r',xaxis,yaxis2,'g--', xaxis,yaxis4,'b--')
plt.title("Spectra")
plt.xlabel("Wavelength (Angstrom)")
plt.ylabel("Spectra data")
plt.legend(legend)
plt.grid(True)
plt.show()
#plt.savefig('d127-spectra.png',dpi=300)