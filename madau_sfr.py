# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 15:37:17 2021

@author: amit
"""

import numpy as np
import pandas as pd
from astropy.cosmology import Planck13, z_at_value
import astropy.units as u
import matplotlib.pyplot as plt

ageU = Planck13.age(0).value

# z array
z = np.arange(0.,8.01,0.01)

# lookback array
l = np.arange(0.5,ageU,0.1)

# computes z at given  lookback
def z_at_given_lookback(lookback):
    age_of_universe = Planck13.age(0)
    ages = age_of_universe - lookback*u.Gyr
    z_lookback = [z_at_value(Planck13.age, age) for age in ages]
    return np.array(z_lookback)

# computes cosmic SFR for a given z as per Madau+14
def cosmic_sfr(z_arr):
    csfr = np.array([])
    for z in z_arr:
        sfr_z = 0.015 * ((1+z)**2.7 / (1 + ((1+z)/2.9)**5.6))
        csfr = np.append(csfr, sfr_z)
        
    return csfr

zl = z_at_given_lookback(l)

c_sfr = cosmic_sfr(zl)

plt.plot(l, c_sfr)