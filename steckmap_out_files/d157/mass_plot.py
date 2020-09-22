# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 12:13:40 2019

@author: Amit
"""

import numpy as np
import matplotlib.pyplot as plt

nEntries=20
nEntriesPerLine=5
nLines=nEntries/nEntriesPerLine
data=np.array([])
with open('d157_1D_2p7_flux.res-MASS.txt','r') as f:
    lines=f.readlines()
    for line in lines:
        eachLine = np.fromstring(line, dtype=float, sep=' ')
        data=np.append(data,eachLine)
        
xaxis=data[1:nEntries+1]
yaxis=data[nEntries+2:]

plt.plot(xaxis,yaxis)
plt.title("Age-Mass")
plt.xlabel("Age (Myr)")
plt.ylabel("Mass")
plt.grid(True)
#plt.show()
plt.savefig('d157-MASS.png',dpi=300)