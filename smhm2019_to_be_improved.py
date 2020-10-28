# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:30:25 2020

@author: amit
"""


import numpy as np

def create_bins(lower_bound, width, quantity):
    bins = []
    arr = np.arange(lower_bound, lower_bound + (quantity-1)*width, width)
    for low in arr:
        bins.append((low, low+width))
    return bins


zbins = create_bins(lower_bound=0., width=0.5, quantity=21)

