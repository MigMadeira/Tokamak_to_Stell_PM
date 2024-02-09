import os
from pathlib import Path
import numpy as np
from simsopt.geo import SurfaceRZFourier
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo import PermanentMagnetGrid, create_equally_spaced_curves, curves_to_vtk, Surface, cylinder_to_vtk
from simsopt.field import Current, DipoleField, coils_via_symmetries
import time
from simsopt.util.permanent_magnet_helper_functions import *
from simsopt.mhd.vmec import Vmec
from simsopt.util import FocusData
from simsopt.util import discretize_polarizations, polarization_axes
import simsoptpp as sopp
from simsopt.objectives import SquaredFlux
from matplotlib import pyplot as plt
from simsopt.solve import GPMO

t_start = time.time()

# Set some parameters
comm = None
nphi = 64 # need to set this to 64 for a real run
ntheta = 64 # same as above

input_name = 'wout_ISTTOK_final_PHIEDGE=0.004823433820504875.nc'                #rescaled to 0.52m major radius and 0.5T volavgB
coordinate_flag = 'cartesian'

#famus_filename = '../Data/ISTELL_diff_VV/ISTELL_different_VV.focus'              #equilibrium cross section
famus_filename = '../Data/ISTELL_diff_VV/ISTELL_different_VV_circular_1cm.focus'  #circular cross section

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent).resolve()
surface_filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

# Make higher resolution surface for plotting Bnormal
qphi = 2 * nphi
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_wout(
    surface_filename, range="full torus",
    quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta
)

# Make the output directory
if 'circular' in famus_filename:
    OUT_DIR = 'ISTELL_different_VV_circular/'
else:
    OUT_DIR = 'ISTELL_different_VV/'
os.makedirs(OUT_DIR, exist_ok=True)

#setting radius for the circular coils
vmec = Vmec(TEST_DIR / input_name)

# Number of Fourier modes describing each Cartesian component of each coil:
order = 5
# Number of unique coil shapes, i.e. the number of coils per half field period:
ncoils = int(24/(2*s.nfp))

# Major radius for the initial circular coils:
R0 = 0.52

# Minor radius for the initial circular coils:
R1 = 0.2025

#Initialize the coils 
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=128)
base_currents = [Current(1.0) * 48e3 for i in range(ncoils)]
coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)

# fix all the coil shapes so only the currents are optimized
for i in range(ncoils):
    base_curves[i].fix_all()
    base_currents[i].fix_all()

# Initialize the coil curves and save the data to vtk
curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")

bs = BiotSavart(coils)

# Plot initial Bnormal on plasma surface from un-optimized BiotSavart coils
make_Bnormal_plots(bs, s_plot, OUT_DIR, "biot_savart_initial")

# check after-optimization average on-axis magnetic field strength
calculate_on_axis_B(bs, s)

# Set up correct Bnormal from TF coils 
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
bs.save(OUT_DIR + f"biot_savart_opt.json")

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

# Finally, initialize the permanent magnet class
pm_opt = PermanentMagnetGrid.geo_setup_from_famus(s, Bnormal, famus_filename, pol_vectors=pol_vectors) 

print('Number of available dipoles = ', pm_opt.ndipoles)

pm_opt.m = np.zeros(pm_opt.ndipoles*3)
b_dipole = DipoleField(pm_opt.dipole_grid_xyz, pm_opt.m,
                       nfp=s.nfp, coordinate_flag=pm_opt.coordinate_flag, m_maxima=pm_opt.m_maxima,stellsym=False)

b_dipole._toVTK(OUT_DIR + "Dipole_Fields_K")

#create the outside boundary for the PMs
if 'circular' not in famus_filename:
    s_out = SurfaceRZFourier.from_wout(surface_filename, nphi = nphi, ntheta = ntheta, range='half period')
    s_out.extend_via_normal(0.01)
    s_out.to_vtk(OUT_DIR + "surf_out")

    pm_opt.remove_magnets_inside_surface(s_out)

# remove any dipoles where the diagnostic ports should be

if 'circular' in famus_filename:
    cylinder_list = [sopp.Cylinder(0.52, 0.11, 2*np.pi/12, np.pi/2, 0.21, 0.0575, 2*np.pi/12, 0),
                    sopp.Cylinder(0.52, 0.11, 2*np.pi/12, 0, 0.21, 0.035, 2*np.pi/12, np.pi/2),
                    sopp.Cylinder(0.52, 0.11, 2*np.pi/12, 3*np.pi/2, 0.21, 0.035, 2*np.pi/12, np.pi),
                    sopp.Cylinder(0.52, 0.11, 4*np.pi/12, np.pi/2, 0.21, 0.035, 4*np.pi/12, 0),
                    sopp.Cylinder(0.52, 0.11, 4*np.pi/12, 0, 0.21, 0.0575, 4*np.pi/12, np.pi/2)]
else:
    cylinder_list = [sopp.Cylinder(0.52+0.015, 0.06, 2*np.pi/12, np.pi/2, 0.21, 0.0575, 2*np.pi/12, 0),
                    sopp.Cylinder(0.52, 0.03, 2*np.pi/12, np.pi/9, 0.21, 0.035, 2*np.pi/12, np.pi/2),
                    sopp.Cylinder(0.52+0.035, 0.03, 2*np.pi/12, 3*np.pi/2, 0.21, 0.035, 2*np.pi/12, np.pi),
                    sopp.Cylinder(0.52-0.035, 0.03, 4*np.pi/12, np.pi/2, 0.21, 0.035, 4*np.pi/12, 0),
                    sopp.Cylinder(0.52, 0.03, 4*np.pi/12, 6*np.pi/18 + 1*np.pi/32, 0.21, 0.0575, 4*np.pi/12, np.pi/2)]


pm_opt.remove_dipoles_inside_shapes(cylinder_list)

b_dipole = DipoleField(pm_opt.dipole_grid_xyz, pm_opt.m,
                       nfp=s.nfp, coordinate_flag=pm_opt.coordinate_flag, m_maxima=pm_opt.m_maxima,stellsym=True)

b_dipole._toVTK(OUT_DIR + "Dipole_Fields_K_after_cylinder_removal")

print('Number of available dipoles after diagnostic port and toroidal removal = ', pm_opt.ndipoles)

for i in range(len(cylinder_list)):
    cylinder_to_vtk(cylinder_list[i], OUT_DIR + f"diagnostics/diagnostic {i}")

pm_opt.write_to_famus(Path(OUT_DIR))

# Set some hyperparameters for the optimization
algorithm = 'ArbVec'  # Algorithm to use
nAdjacent = 10  # How many magnets to consider "adjacent" to one another
nHistory = 400 # How often to save the algorithm progress
thresh_angle = 7*np.pi/9 # The angle between two "adjacent" dipoles such that they should be removed
max_nMagnets = 27600 #44000 for the equilibrium cross section case #27600 for the circular case
nBacktracking = 200
kwargs = initialize_default_kwargs('GPMO')
kwargs['K'] = max_nMagnets # Maximum number of GPMO iterations to run
kwargs['nhistory'] = nHistory
if algorithm == 'backtracking' or algorithm == 'ArbVec_backtracking':
    kwargs['backtracking'] = nBacktracking  # How often to perform the backtrackinig
    kwargs['Nadjacent'] = nAdjacent
    kwargs['dipole_grid_xyz'] = np.ascontiguousarray(pm_opt.dipole_grid_xyz)
    kwargs['max_nMagnets'] = max_nMagnets
    if algorithm == 'ArbVec_backtracking':
        kwargs['thresh_angle'] = thresh_angle

# Optimize the permanent magnets greedily
t1 = time.time()
R2_history, Bn_history, m_history = GPMO(pm_opt, algorithm, **kwargs)
t2 = time.time()
print('GPMO took t = ', t2 - t1, ' s')

# plot the MSE history
iterations = np.linspace(0, max_nMagnets, len(R2_history), endpoint=False)
plt.figure()
plt.semilogy(iterations, R2_history, label=r'$f_B$')
plt.semilogy(iterations, Bn_history, label=r'$<|Bn|>$')
plt.grid(True)
plt.xlabel('K')
plt.ylabel('Metric values')
plt.legend()
plt.savefig(OUT_DIR + 'GPMO_MSE_history.png')

# Set final m to the minimum achieved during the optimization
min_ind = np.argmin(R2_history)
pm_opt.m = np.ravel(m_history[:, :, min_ind])
print("best result = ", 0.5 * np.sum((pm_opt.A_obj @ pm_opt.m - pm_opt.b_obj) ** 2))
np.savetxt(OUT_DIR + 'best_result_m=' + str(int(max_nMagnets/ (kwargs['nhistory']) * min_ind )) + '.txt', m_history[:, :, min_ind ].reshape(pm_opt.ndipoles * 3))
b_dipole = DipoleField(pm_opt.dipole_grid_xyz, m_history[:, :, min_ind ].reshape(pm_opt.ndipoles * 3),
                       nfp=s.nfp, coordinate_flag=pm_opt.coordinate_flag, m_maxima=pm_opt.m_maxima,)
b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
b_dipole._toVTK(OUT_DIR + "Dipole_Fields_K" + str(int(max_nMagnets / (kwargs['nhistory']) * min_ind)))
bs.set_points(s_plot.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
Bnormal_dipoles = np.sum(b_dipole.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
Bnormal_total = Bnormal + Bnormal_dipoles
# For plotting Bn on the full torus surface at the end with just the dipole fields
make_Bnormal_plots(b_dipole, s_plot, OUT_DIR, "only_m_optimized_K" + str(int(kwargs['K'] / (kwargs['nhistory']) * min_ind)))
pointData = {"B_N": Bnormal_total[:, :, None]}
s_plot.to_vtk(OUT_DIR + "m_optimized_K" + str(int(max_nMagnets / (kwargs['nhistory']) * min_ind)), extra_data=pointData)
    

# Print effective permanent magnet volume
B_max = 1.465
mu0 = 4 * np.pi * 1e-7
M_max = B_max / mu0 
dipoles = pm_opt.m.reshape(pm_opt.ndipoles, 3)
print('Volume of permanent magnets is = ', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))) / M_max)
print('sum(|m_i|)', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))))

save_plots = True
if save_plots:
    # Save the MSE history and history of the m vectors
    #np.savetxt(
    #    OUT_DIR / f"mhistory_K{kwargs['K']}_nphi{nphi}_ntheta{ntheta}.txt", 
    #    m_history.reshape(pm_opt.ndipoles * 3, kwargs['nhistory'] + 1)
    #)
    np.savetxt(
        OUT_DIR + f"R2history_K{max_nMagnets}_nphi{nphi}_ntheta{ntheta}.txt",
        R2_history
    )
    mu0 = 4 * np.pi * 1e-7
    Bmax = 1.465
    vol_eff = np.sum(np.sqrt(np.sum(m_history ** 2, axis=1)), axis=0) * mu0 * 2 * s.nfp / Bmax
    np.savetxt(OUT_DIR + 'eff_vol_history_K' + str(max_nMagnets) + '_nphi' + str(nphi) + '_ntheta' + str(ntheta) + '.txt', vol_eff)
    
    # Plot the SIMSOPT GPMO solution
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
    make_Bnormal_plots(bs, s_plot, OUT_DIR, "biot_savart_optimized")

    # Look through the solutions as function of K and make plots
    for k in range(0, kwargs["nhistory"] + 1, 40):
        mk = m_history[:, :, k].reshape(pm_opt.ndipoles * 3)
        np.savetxt(OUT_DIR + 'result_m=' + str(int(max_nMagnets / (kwargs['nhistory']) * k)) + '.txt', m_history[:, :, k].reshape(pm_opt.ndipoles * 3))
        b_dipole = DipoleField(
            pm_opt.dipole_grid_xyz,
            mk, 
            nfp=s.nfp,
            coordinate_flag=pm_opt.coordinate_flag,
            m_maxima=pm_opt.m_maxima,
        )
        b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
        K_save = int(max_nMagnets/ kwargs['nhistory'] * k)
        b_dipole._toVTK(OUT_DIR + f"Dipole_Fields_K{K_save}_nphi{nphi}_ntheta{ntheta}")
        print("Total fB = ", 0.5 * np.sum((pm_opt.A_obj @ mk - pm_opt.b_obj) ** 2))
        Bnormal_dipoles = np.sum(b_dipole.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
        Bnormal_total = Bnormal + Bnormal_dipoles

        # For plotting Bn on the full torus surface at the end with just the dipole fields
        make_Bnormal_plots(b_dipole, s_plot, OUT_DIR, f"only_m_optimized_K{K_save}_nphi{nphi}_ntheta{ntheta}")
        pointData = {"B_N": Bnormal_total[:, :, None]}
        s_plot.to_vtk(OUT_DIR + f"m_optimized_K{K_save}_nphi{nphi}_ntheta{ntheta}", extra_data=pointData)

    # write solution to FAMUS-type file
    pm_opt.write_to_famus(Path(OUT_DIR))

# Compute metrics with permanent magnet results
dipoles_m = pm_opt.m.reshape(pm_opt.ndipoles, 3)
num_nonzero = np.count_nonzero(np.sum(dipoles_m ** 2, axis=-1)) / pm_opt.ndipoles * 100
print("Number of possible dipoles = ", pm_opt.ndipoles)
print("% of dipoles that are nonzero = ", num_nonzero)

# Print optimized f_B and other metrics
### Note this will only agree with the optimization in the high-resolution
### limit where nphi ~ ntheta >= 64!
b_dipole = DipoleField(
    pm_opt.dipole_grid_xyz,
    pm_opt.m, 
    nfp=s.nfp,
    coordinate_flag=pm_opt.coordinate_flag,
    m_maxima=pm_opt.m_maxima,
)
b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
bs.set_points(s_plot.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
f_B_sf = SquaredFlux(s_plot, b_dipole, -Bnormal).J()
print('f_B = ', f_B_sf)
total_volume = np.sum(np.sqrt(np.sum(pm_opt.m.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1))) * s.nfp * 2 * mu0 / B_max
print('Total volume = ', total_volume)

t_end = time.time()
print('Total time = ', t_end - t_start)
