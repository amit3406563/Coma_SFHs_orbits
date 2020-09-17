import numpy as np

sat_names = ['3254','3269', '3291', '3329', '3352', '3367', '3414', '3484',
              '3534', '3565', '3639', '3664']
log_Ms= np.loadtxt('logMs_coma.m') # loading stellar mass from text file
Ms = 10**log_Ms
print('Satellite stellar masses (also in  log10) in solar mass: \n')
for name, m, lm in zip(sat_names, Ms, log_Ms):
    print('GMP'+name+' | '+'{:.2e}'.format(m)+' | '+'{:.2f}'.format(lm))
print('\n')
# stellar masses for satellites provided by Prof. Dr. Scott Trager

# Relation for SM-HM conversion from Behroozi+ 2010 [Eqn-21]
# parameter values for eqn-21 taken from Table-2
# this is direct conversion of satellite masses of all 12 galaxies with 
# correct virial definition and unit in solar mass
log_M1 = 12.35
log_Ms0 = 10.72
beta = 0.44
delta = 0.57
gamma = 1.56

Ms0 = 10**log_Ms0
k = Ms/Ms0

log_Mh = log_M1 + beta*log_Ms - beta*log_Ms0 + (k**delta)/(1+k**(-gamma)) - 1/2
Mh_sat = 10**log_Mh
np.savetxt('logMh_coma.m',log_Mh)
print('Satellite halo masses (also in  log10) in solar mass: \n')
for name, m, lm in zip(sat_names, Mh_sat, log_Mh):
    print('GMP'+name+' | '+'{:.2e}'.format(m)+' | '+'{:.2f}'.format(lm))
print('\n')
