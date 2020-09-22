# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 12:01:38 2019

@author: Amit
"""

import numpy as np
import matplotlib.pyplot as plt

nEntries=30
nEntriesPerLine=5
nLines=nEntries/nEntriesPerLine
data=np.array([])
with open('n4864_1D_2p7_flux.res-AMR.txt','r') as f:
    lines=f.readlines()
    for line in lines:
        eachLine = np.fromstring(line, dtype=float, sep=' ')
        data=np.append(data,eachLine)
        
xaxis=data[1:nEntries+1]
yaxis=data[nEntries+2:]

plt.plot(xaxis,yaxis)
plt.title("Age-Metallicty")
plt.xlabel("Age (Myr)")
plt.ylabel("AMR (0.02 is solar)")
plt.grid(True)
#plt.show()
plt.savefig('n4864-AMR.png',dpi=300)