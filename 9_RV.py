# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 23:32:02 2020

@author: amit
"""

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP7

def coord_icrs(ra,dec):
    return SkyCoord(ra, dec, frame = 'icrs')

sat_names = ['3254','3269', '3291', '3329', '3352', '3367', '3414', '3484',
              '3534', '3565', '3639', '3664']

coord_coma = pd.read_csv('Coma_coords.csv')
coord_coma.index = sat_names

coma_center_coord = list(coord_coma.loc['3329',:])
#coma_center_coord = ['12h59m46.7s', '+27d57m00s']
coord_coma_centre = coord_icrs(coma_center_coord[0],coma_center_coord[1])

print('Coma center coordinates: \n'+'RA: '+coma_center_coord[0]+'\n'+
      'DEC: '+coma_center_coord[1]+'\n')

ang_sep_rad = np.array([])
ang_sep_deg = np.array([])
print('Satellites coordinates (RA & DEC): \n')
for name in sat_names:
    ra = coord_coma.loc[name]['RA']
    dec = coord_coma.loc[name]['DEC']
    print('GMP'+name+' | '+ra+' | '+dec)
    temp1 = vars()['theta_rad_'+name] = coord_coma_centre.separation(
        coord_icrs(ra,dec)).rad
    ang_sep_rad = np.append(ang_sep_rad,temp1)
    temp2 = vars()['theta_deg_'+name] = temp1 * (180/np.pi)
    ang_sep_deg = np.append(ang_sep_deg,temp2)
print('\n')    

print('Angular separation of satellites from Coma center (rad. & deg.): \n')
for name, ar, ad in zip(sat_names, ang_sep_rad, ang_sep_deg):
    print('GMP'+name+' | '+'{:.5f}'.format(ar)+' | '+'{:.3f}'.format(ad))
print('\n')

z = np.loadtxt('redshift.z') # reading redshift
print('Redshift of satellites: \n')
z_coma = z[3]
#z_coma = 0.0231 # from NASA/IPAC EXTRAGALACTIC DATABASE
print('Redshift of Coma cluster: '+str(z_coma)+'\n')
for name, red in zip(sat_names,z):
    print('GMP'+name+' | '+str(red))
print('\n')

dA = WMAP7.angular_diameter_distance(z_coma)
print('Angular diameter distance of Coma cluster in Mpc: '+
      '{:.2f}'.format(dA.value)+'\n') 
# angular diameter distance of Coma in Mpc
# computing R [R/r_vir_coma] for each satellite galaxy
r_vir_coma = 2.9/0.73 # in Mpc from Lokas and Mamon 2003
print('Virial radius of Coma cluster in Mpc: '+'{:.2f}'.format(r_vir_coma)+'\n')
R = ang_sep_rad * dA.value / r_vir_coma
np.savetxt('ang_sep_rad.rad',ang_sep_rad)
np.savetxt('ang_sep_deg.deg',ang_sep_rad*(180/np.pi))
np.savetxt('R.vir',R)

sig_1D_coma = 1154 # in km/s from Jorgensen et al 2018 Table-1
sig_3D_coma = sig_1D_coma * np.sqrt(3)
print('sigma_1D_Coma in km/s: '+str(sig_1D_coma)+'\n')
print('sigma_3D_Coma in km/s: '+'{:.2f}'.format(sig_3D_coma)+'\n')
c = 299792458/1000 # speedof light in km/s
V = (c * np.abs(z - z_coma)) / ((1 + z_coma) * sig_3D_coma)
np.savetxt('V.sig3d',V)

print('R/r_vir & V/sigma_3D for all satellites: \n')
for name, r, v in zip(sat_names, R, V):
    print('GMP'+name+' | '+'{:.3f}'.format(r)+' | '+'{:.2f}'.format(v))
print('\n')