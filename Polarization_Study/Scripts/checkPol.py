#Script to produce the vtk for Figure 11 and the statistics for Table 2. To see something similar to the Figure make sure to toggle "Interpret Values As Categories" in paraview.

import os
from pathlib import Path
import numpy as np
from simsopt.util import FocusData, discretize_polarizations, polarization_axes
from simsopt.util.permanent_magnet_helper_functions import *


# Make the output directory
OUT_DIR = '../Plots/'
os.makedirs(OUT_DIR, exist_ok=True)

famus_filename = '../Data/all/SIMSOPT_dipole_solution.focus' #name of the grid to find the polarizations of
vtkname = OUT_DIR + "polarizations_hfp"                      #name of the vtk file to save

#include only the polarizations in use
pols = ['face',     
        'fe_ftri', 
        'fc_ftri',
        'corner',
        'edge'
        ]

pol_axes = np.zeros((0, 3))
pol_type = np.zeros(0, dtype=int)
i = 0
 
for x in pols:
    # Determine the allowable polarization types and reject the negatives
    pol_axes_temp, pol_type_temp = polarization_axes([x])
    ntype_temp = int(len(pol_type_temp)/2)
    pol_axes_temp = pol_axes_temp[:ntype_temp, :]
    pol_type_temp = pol_type_temp[:ntype_temp] + i
    pol_axes = np.concatenate((pol_axes, pol_axes_temp), axis=0)
    pol_type = np.concatenate((pol_type, pol_type_temp))
    
    i += 1

#load focus file with the grid info
mag_data = FocusData(famus_filename)
    
#setup the polarization vectors from the magnet data in the focus file to draw a field period.
#mag_data.repeat_hp_to_fp(2)

ophi = np.arctan2(mag_data.oy, mag_data.ox) 
discretize_polarizations(mag_data, ophi, pol_axes, pol_type)

polarizations = pol_type[mag_data.pol_id-1]
data = {"polarizations": polarizations}
from pyevtk.hl import pointsToVTK
pointsToVTK(
    str(vtkname), mag_data.ox, mag_data.oy, mag_data.oz, data=data
)

for j in range(len(pols)):
    a=1
    #if j==1:
    #    a=2
    counter = polarizations.tolist().count(j+a)
    print(f"there are {counter} {pols[j]} magnets in the grid")
    