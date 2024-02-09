import os
import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from simsopt.geo import SurfaceRZFourier, PermanentMagnetGrid
from simsopt.field import DipoleField, ToroidalField
import simsoptpp as sopp
from simsopt.util.permanent_magnet_helper_functions import *


# Set some parameters
nphi = 64  # need to set this to 64 for a real run
ntheta = 64  # same as above
dr = 0.1  # dr is used when using cylindrical coordinates
dz = 0.1

input_name = '../wout_ISTELL_final.nc'
famus_filename = '../grids/ISTELL_shifted_axis/ISTELL_1cm_cubes_shifted_radial_extent=02725m_nfp=2.focus'

# Make the output directory
OUT_DIR = './test_grid/' 
os.makedirs(OUT_DIR, exist_ok=True)

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent).resolve()
surface_filename = str(TEST_DIR/input_name)

#set dummy surface
s = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

#set dummy bnormal
net_poloidal_current_Amperes = 3.7713e+6
mu0 = 4 * np.pi * 1e-7
RB = mu0 * net_poloidal_current_Amperes / (2 * np.pi)
bs = ToroidalField(R0=1, B0=RB)
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

#create the outside boundary for the PMs (corresponds to the coil limit)
s_remove = SurfaceRZFourier.from_nphi_ntheta(nphi = nphi, ntheta = ntheta, range='half period', nfp=2, stellsym=True)
s_remove.set_rc( 0, 0, 0.46)
s_remove.set_rc( 1, 0, 0.2025)
s_remove.set_zs( 1, 0, 0.2025)
s_remove.to_vtk(OUT_DIR + "surf_to_remove")

# Initialize the permanent magnet class
pm_opt = PermanentMagnetGrid.geo_setup_from_famus(s, Bnormal, famus_filename) 

#print the number of magnets and plot the magnet grid before removal
print('Number of available dipoles = ', pm_opt.ndipoles)
pm_opt.m = np.zeros(pm_opt.ndipoles*3)
pm_opt.ndipoles = int(pm_opt.ndipoles/(s.nfp))
print('Number of available dipoles = ', pm_opt.ndipoles)

b_dipole = DipoleField(pm_opt.dipole_grid_xyz[:pm_opt.ndipoles], np.zeros(pm_opt.ndipoles*3),
                       nfp=s.nfp, coordinate_flag=pm_opt.coordinate_flag, m_maxima=pm_opt.m_maxima[:pm_opt.ndipoles],)
b_dipole._toVTK(OUT_DIR + "Dipole_Fields_K_hfp")
pm_opt.ndipoles = int(pm_opt.ndipoles*(s.nfp))

#remove magnets outside a given surface s_remove
pm_opt.remove_magnets_inside_surface(s_remove)

#print the number of magnets and plot the magnet grid after removal
pm_opt.m = np.zeros(pm_opt.ndipoles*3)
print('Number of available dipoles = ', pm_opt.ndipoles)

pm_opt.ndipoles = int(pm_opt.ndipoles/(s.nfp))
b_dipole = DipoleField(pm_opt.dipole_grid_xyz[:pm_opt.ndipoles], np.zeros(pm_opt.ndipoles*3),
                       nfp=s.nfp, coordinate_flag=pm_opt.coordinate_flag, m_maxima=pm_opt.m_maxima[:pm_opt.ndipoles],)
b_dipole._toVTK(OUT_DIR + "Dipole_Fields_K_after_torus_removal_hfp")

#write solution to FAMUS-type file
pm_opt.write_to_famus(Path(OUT_DIR))
