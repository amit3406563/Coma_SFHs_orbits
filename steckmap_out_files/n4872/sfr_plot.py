# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 12:18:49 2019

@author: Amit
"""

import numpy as np
import matplotlib.pyplot as plt

nEntries=30
nEntriesPerLine=5
nLines=nEntries/nEntriesPerLine
data=np.array([])
with open('n4872_1D_2p7_fluxc.res-SFR.txt','r') as f:
    lines=f.readlines()
    for line in lines:
        eachLine = np.fromstring(line, dtype=float, sep=' ')
        data=np.append(data,eachLine)
        
xaxis=data[1:nEntries+1]
yaxis=data[nEntries+2:]

plt.plot(xaxis,yaxis)
plt.title("Age-SFR")
plt.xlabel("Age (Myr)")
plt.ylabel("Unnormalized SFR (M_sol/yr)")
plt.grid(True)
#plt.show()
plt.savefig('n4872-SFR.png',dpi=300)