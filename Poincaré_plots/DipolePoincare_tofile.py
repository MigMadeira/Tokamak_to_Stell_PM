import os
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.field.magneticfieldclasses import DipoleField
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo import PermanentMagnetGrid
from simsopt.field import compute_fieldlines
import pickle
import time
from simsopt.util.permanent_magnet_helper_functions import *
from simsopt.mhd.vmec import Vmec
from simsopt import load
from math import sqrt,ceil
from mpi4py import MPI
from simsopt.util import MpiPartition, FocusData, discretize_polarizations, polarization_axes
mpi = MpiPartition()
comm = MPI.COMM_WORLD

# Set some parameters
comm = None
nphi = 64 # need to set this to 64 for a real run
ntheta = 64 # same as above
#input_name = 'wout_ISTELL_final.nc'
input_name = 'wout_ISTTOK_final_rescaled.nc'
coordinate_flag = 'cartesian'


# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent).resolve()
surface_filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

vmec_final = Vmec(TEST_DIR / input_name)
ntheta_VMEC = 200

# Make the output directory
OUT_DIR = './Poincare_plots/Data/'
os.makedirs(OUT_DIR, exist_ok=True)

# Files for the desired initial coils, magnet grid and magnetizations:
coilfile = "../Different_VV/Data/circular_cs/biot_savart_opt.json"
famus_filename = "../Different_VV/Data/circular_cs/SIMSOPT_dipole_solution.focus"
dipole_file = "../Different_VV/Data/circular_cs/best_result_m=25600.txt"

# Get the Biot Savart field from the coils:
bs = load(coilfile)

# Set up correct Bnormal from the coils 
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

#load focus file with the grid info
mag_data = FocusData(famus_filename)

# Determine the allowable polarization types and reject the negatives
pol_axes = np.zeros((0, 3))
pol_type = np.zeros(0, dtype=int)
pol_axes_f, pol_type_f = polarization_axes(['face'])
ntype_f = int(len(pol_type_f)/2)
pol_axes_f = pol_axes_f[:ntype_f, :]
pol_type_f = pol_type_f[:ntype_f]
pol_axes = np.concatenate((pol_axes, pol_axes_f), axis=0)
pol_type = np.concatenate((pol_type, pol_type_f))

# Optionally add additional types of allowed orientations
PM4Stell_orientations = True
full_orientations = False
if PM4Stell_orientations:
    pol_axes_fe_ftri, pol_type_fe_ftri = polarization_axes(['fe_ftri'])
    ntype_fe_ftri = int(len(pol_type_fe_ftri)/2)
    pol_axes_fe_ftri = pol_axes_fe_ftri[:ntype_fe_ftri, :]
    pol_type_fe_ftri = pol_type_fe_ftri[:ntype_fe_ftri] + 1
    pol_axes = np.concatenate((pol_axes, pol_axes_fe_ftri), axis=0)
    pol_type = np.concatenate((pol_type, pol_type_fe_ftri))

    pol_axes_fc_ftri, pol_type_fc_ftri = polarization_axes(['fc_ftri'])
    ntype_fc_ftri = int(len(pol_type_fc_ftri)/2)
    pol_axes_fc_ftri = pol_axes_fc_ftri[:ntype_fc_ftri, :]
    pol_type_fc_ftri = pol_type_fc_ftri[:ntype_fc_ftri] + 2
    pol_axes = np.concatenate((pol_axes, pol_axes_fc_ftri), axis=0)
    pol_type = np.concatenate((pol_type, pol_type_fc_ftri))
    
    if full_orientations:
        pol_axes_corner, pol_type_corner = polarization_axes(['corner'])
        ntype_corner = int(len(pol_type_corner)/2)
        pol_axes_corner = pol_axes_corner[:ntype_corner, :]
        pol_type_corner = pol_type_corner[:ntype_corner] + 1
        pol_axes = np.concatenate((pol_axes, pol_axes_corner), axis=0)
        pol_type = np.concatenate((pol_type, pol_type_corner))
        
        pol_axes_edge, pol_type_edge = polarization_axes(['edge'])
        ntype_edge = int(len(pol_type_edge)/2)
        pol_axes_edge = pol_axes_edge[:ntype_edge, :]
        pol_type_edge = pol_type_edge[:ntype_edge] + 1
        pol_axes = np.concatenate((pol_axes, pol_axes_edge), axis=0)
        pol_type = np.concatenate((pol_type, pol_type_edge))
        


#setup the polarization vectors from the magnet data in the focus file
ophi = np.arctan2(mag_data.oy, mag_data.ox) 
discretize_polarizations(mag_data, ophi, pol_axes, pol_type)
pol_vectors = np.zeros((mag_data.nMagnets, len(pol_type), 3))
pol_vectors[:, :, 0] = mag_data.pol_x
pol_vectors[:, :, 1] = mag_data.pol_y
pol_vectors[:, :, 2] = mag_data.pol_z
print('pol_vectors_shape = ', pol_vectors.shape)

# Initialize the permanent magnet class
pm_opt = PermanentMagnetGrid.geo_setup_from_famus(s, Bnormal, 
                                                  famus_filename, pol_vectors=pol_vectors)

# Get the Biot Savart field from the magnets:
pm_opt.m = np.loadtxt(dipole_file)
b_dipole = DipoleField(pm_opt.dipole_grid_xyz, pm_opt.m.reshape(pm_opt.ndipoles * 3),
                       nfp=s.nfp, coordinate_flag=pm_opt.coordinate_flag, m_maxima=pm_opt.m_maxima)
b_dipole.set_points(s.gamma().reshape((-1, 3)))

print("Total fB = ",
      0.5 * np.sum((pm_opt.A_obj @ pm_opt.m - pm_opt.b_obj) ** 2))

bs_final = b_dipole + bs

# Do Poincaré plots from here

print("Obtaining VMEC final surfaces")
nfp = vmec_final.wout.nfp
nzeta = 4
nradius = 4
nfieldlines = nradius
zeta = np.linspace(0,2*np.pi/nfp,num=nzeta,endpoint=False)
theta = np.linspace(0,2*np.pi,num=ntheta_VMEC)
iradii = np.linspace(0,vmec_final.wout.ns-1,num=nfieldlines).round()
iradii = [int(i) for i in iradii]
R_final = np.zeros((nzeta,nradius,ntheta_VMEC))
Z_final = np.zeros((nzeta,nradius,ntheta_VMEC))
for itheta in range(ntheta_VMEC):
    for izeta in range(nzeta):
        for iradius in range(nradius):
            for imode, xnn in enumerate(vmec_final.wout.xn):
                angle = vmec_final.wout.xm[imode]*theta[itheta] - xnn*zeta[izeta]
                R_final[izeta,iradius,itheta] += vmec_final.wout.rmnc[imode, iradii[iradius]]*np.cos(angle)
                Z_final[izeta,iradius,itheta] += vmec_final.wout.zmns[imode, iradii[iradius]]*np.sin(angle)
R0 = R_final[0,:,0]
Z0 = Z_final[0,:,0]

print("Finished VMEC")
from simsopt.field import particles_to_vtk
tmax_fl= 5000
tol_poincare=1e-14
def trace_fieldlines(bfield, R0, Z0):
    t1 = time.time()
    phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=tol_poincare, comm=comm,
        phis=phis, stopping_criteria=[])
    t2 = time.time()
    print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
    #particles_to_vtk(fieldlines_tys, f'fieldlines_optimized_coils')
    return fieldlines_tys, fieldlines_phi_hits, phis

print("started tracing")
fieldlines_tys, fieldlines_phi_hits, phis = trace_fieldlines(bs_final, R0, Z0)
#Poincare_data = np.array([fieldlines_tys, fieldlines_phi_hits, phis, R_final, Z_final], dtype=object)
Poincare_data = np.empty(5, dtype=object)
Poincare_data[0] = fieldlines_tys
Poincare_data[1] = fieldlines_phi_hits
Poincare_data[2] = phis
Poincare_data[3] = R_final
Poincare_data[4] = Z_final


print("writing to file")
np.save(OUT_DIR+f"Poincare_data_tol={tol_poincare}_t={tmax_fl}_1cm_diffVV_circular",Poincare_data)
