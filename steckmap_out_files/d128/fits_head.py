# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 18:43:41 2019

@author: Amit
"""

from astropy.io import fits

hdul=fits.open('d128_1D_2p7_fluxc.fits')
print(hdul[0].header['REDSHIFT'])
hdul.close()