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
| 1_steckmap_plot.py        | Reads the STECKMAP output files containing SAD, MASS, SFR, AMR, LOSVD, and Spectra information for different SSPs, namely, MILES, BC03, and PHR. |
| 2_steckmap_sfr_extract.py | Extracts SFR tables from STECKMAP output tables. |
| 3_sfr_ssfr_compute.py     | SFR values are smoothed by interpolation and relative SFRs are computed, SSFR is computed, and the SFR and SSFR tables are saved. |
| 4_sfr_ssfr_plot.py        | Plots SFR and SSFR and saves plots. |
| 5_read_sim_tab.py         | Reads orbit simulation tables from file and saves the output tables. |
