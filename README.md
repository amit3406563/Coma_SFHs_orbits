# MSc Astronomy Project

## Star formation histories of Coma Cluster galaxies matched to simulated orbits hint at quenching around first pericenter
### A.K. Upadhyay, K.A. Oman, and S.C. Trager
---------------------------------------------------------------------------------------------------------------------------

*All codes related to my masters research during 2019-2020 and further work during 2020-21 for publication in A & A journal.*

| <!-- -->               | <!-- -->                                                                          |
|------------------------|-----------------------------------------------------------------------------------|
| Code Author:           | **Amit Kumar Upadhyay** (Kapteyn Astronomical Institute, University of Groningen) |
| In collaboration with: | **Prof. Dr. Scott Trager** (Kapteyn Astronomical Institute, University of Groningen) <br /> **Dr. Kyle Oman** (Department of Physics - Astronomy, Durham University) |
                       
Programming Languages: **Python** and **Yorrick** <br />

| Filenames                 | Description of code files                                                           |
|---------------------------|-------------------------------------------------------------------------------------|
| 1_steckmap_plot.py        | Reads (tables saved as '*.dat'), plots (plots saved as '*.pdf') the STECKMAP output files containing SAD, MASS, SFR, AMR, LOSVD, and Spectra information for different SSPs, namely, MILES, BC03, and PHR. <br /> STECKMAP output files repository: './steckmap_out/<ssp>/' <br /> STECKMAP attribute tables output repository: './out_files_steckmap/<ssp>/' |
| 2_steckmap_sfr_extract.py | Extracts SFR tables (SFR tables saved as '*.csv') from STECKMAP output tables <br /> STECKMAP attributes tables repository: './out_files_steckmap/<ssp>/<*_SFR.dat>/' <br /> SFR tables are saved in the repository: './steckmap_sfr_tables/<ssp>/' |
| 3_sfr_ssfr_compute.py     | SFR values are smoothed by interpolation and rel. SFRs (normalized to 1) are computed, SSFR is computed, and the SFR and SSFR tables are saved as '*.csv' <br /> SFR tables are read from the repository- './steckmap_sfr_tables/<ssp>/' <br /> Smoothed rel. SFR and computed SSFR tables are saved in the repository: './sfr_ssfr_tables/<ssp>/' |
| 4_sfr_ssfr_plot.py        | SFR and SSFR plotted and saved as '*.pdf'                                             |
| 5_read_sim_tab.py         | Reads orbit simulation tables from '*.HDF5' file and saves the output tables as '*.csv' |
