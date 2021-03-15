# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 00:52:12 2020

@author: amit
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

x = np.array([1,2,3,4,5,6,7,8])
y = x**2
z = x/2

fig, ax = plt.subplots(figsize=(7,5))
ax.plot(x,y)
ax2 = ax.twiny()
ax.set_xticks(x)
ax.set_xticklabels(x)
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(x)
ax2.set_xticklabels(z)