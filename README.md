# Tokamak_to_Stell_PM [![DOI](https://zenodo.org/badge/755053921.svg)](https://zenodo.org/doi/10.5281/zenodo.10656979)
 Assessing the viability of converting the ISTTOK Tokamak into a stellarator using only permanent magnets.

## Requisites
 - VMEC2000: https://github.com/hiddenSymmetries/VMEC2000
 - booz_xform: https://github.com/hiddenSymmetries/booz_xform
 - SIMSOPT: 3 versions were used. https://github.com/hiddenSymmetries/simsopt commit hash 2bdc0c10a15e059035af0e1f564b958c8b2d910f of the permanent_magnet branch for the cases where the magnet grid is uniform toroidal (sections 4.1 and 4.2); hash c859dfb33e9a193a064e7221b4100b83a8950575 of the master branch for ArbVec GPMO optimizations without the need to remove dipoles or for magnetic equilibrium optimizations, and the dipole_removal branch of the fork https://github.com/MigMadeira/simsopt/tree/dipole_removal that implements multiple methods for dipole removal allowing the introduction of plasma accessing ports and the removal of dipoles inside toroidal surfaces, namely used for the shifted configuration in section 5.2 and all configurations in sections 5.3 and 5.4.
 - MAGPIE: MAGPIE is, for now, not available to everyone, so all MAGPIE files are already provided in a form which can be used as an input to SIMSOPT.
  
Other commonly used python requisites for data analysis, namely scipy, numpy and matplotlib.
It is also useful to have a way to visualize .vtk, .vtu, and .vts files, such as ParaView.

This repository is organized following the subsections of the article. 
For each subsection there is a Data, Plots and Scripts folders. These folders may be further subdivided according to the legend of the displayed figures.
WARNING: The scripts provided in this repository, namely those referent to PM optimizations often produce very large files as output due to the large number of variables.

## Data folder
In the Data folder you will find:
 - best_result_m=*.txt: a text file containing a long vector with all the components, m, of the dipoles for a given optimal solution. Is used as an input to the poincaré plot scripts.
 - biot_savart_*.json: a json containing the initialisation of the coils for a given optimization.
 - eff_vol_history_K*_nphi_ntheta.txt: a file containing the effective volume saved throughout a given optimization. Often corresponds to the x axis of the shown optimization curves.
 - R2history_K*_nphi_ntheta.txt: a file containing the MSE history saved throughout a given optimization. Often corresponds to the y axis of the shown optimization curves.
 - SIMSOPT_dipole_solution.focus: Simsopt output of the filled grid with the optimal solution. Results from running the *write_to_famus* function at the end of the optimization.
 - *.focus: Other focus files correspond to a MAGPIE output given to SIMSOPT as an input to build the magnet array.
 - *.nml: Namelist used as an input to MAGPIE to produce the focus file of the same name as an output.
 - wout_*.nc: VMEC outputs for rescaled equilibria.
 - Other text files with useful information.

## Plots folder
 In the Plots folder you will find the pdf or png files corresponding to the article figures, as well as the .vtk, .vtu, and .vts files used to make them.
 Note: obtaining exact replicas of the figures may require tweeking the settings of ParaView.

## Scripts folder
 In the Scripts folder you will find one or more python files used to do the permanent magnet optimization, a jupyter notebook that produces the articles figures excluding the ones made with paraview and python files that extract some information from the optimization results, for instance the polarization of the magnets.

In the "Equilibrium_Optimization" folder there is the information of the QA equilibrium relevant to sections 4 and 5.4 in the folder named ISTTOK_stell and the QA equilibrium relevant to the remainder of section 5 in the folder named ISTELL. 

The Poincaré plots were obtained by running the python files in the directory of the same name. Often the result is saved as a .npy file to allow plotting the data from multiple scripts on the same figure. The .npy files are not provided due to the large file size.
