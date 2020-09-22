# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 18:43:41 2019

@author: Amit
"""

from astropy.io import fits

hdul=fits.open('d127_1D_2p7_flux.fits')
print(hdul[0].header['REDSHIFT'])
hdul.close()