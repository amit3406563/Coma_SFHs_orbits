import numpy as np
import pandas as pd
from scipy import interpolate

sfr_3484 = pd.read_csv('./out_files/d157_SFR.dat',delimiter=' ')

age_myr = np.array(sfr_3484['log[Age(Myr)]'])
sfr = np.array(sfr_3484['d157_SFR'])

## read some other age array from any other file
sfr_3254 = pd.read_csv('./out_files/d127_SFR.dat',delimiter=' ')
age = np.array(sfr_3254['log[Age(Myr)]'])

f = interpolate.interp1d(age_myr,sfr,fill_value='extrapolate')

sfr_exp = f(age)

sfr_df = pd.DataFrame()
sfr_df['log[Age(Myr)]'] = age
sfr_df['d157_SFR'] = sfr_exp

sfr_df.to_csv('./out_files/d157_SFR.dat',index=None,sep=' ')
