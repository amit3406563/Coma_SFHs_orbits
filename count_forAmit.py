import h5py
import numpy as np

Mcoma = 1.53E15

with h5py.File('VVV_future/pdfs/pdf.hdf5', 'r') as f:
    Mhost = f['/satellites/Mhost'][()]
    Mmax = f['/satellites/m_max'][()]
    R = f['/satellites/R'][()]
    V = f['/satellites/V'][()]
    r = f['/satellites/r'][()]

    Hmask = np.abs(np.log10(Mhost) - np.log10(Mcoma)) <= 0.5
    print(np.unique(Mhost[Hmask]).size, 'hosts with similar masses to Coma.')

    for Mlim in 10**np.array([13.5, 14, 14.5]):
        print(
            np.unique(Mhost[
                np.logical_and.reduce((Hmask, Mmax > Mlim, R > 0))
            ]).size,
            'Coma-mass hosts with >=1 satellite of logMmax >=',
            np.log10(Mlim),
            'of which',
            np.unique(Mhost[
                np.logical_and.reduce((Hmask, Mmax > Mlim, R > 0, r < .5))
            ]).size,
            'have such a satellite within r<0.5'
        )
