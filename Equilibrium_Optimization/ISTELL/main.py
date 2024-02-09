#!/usr/bin/env python3
import os
import glob
import shutil
import numpy as np
from pathlib import Path
from simsopt.util import MpiPartition
from simsopt.mhd import Vmec,  QuasisymmetryRatioResidual
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from simsopt.geo import CurveSurfaceDistance, curves_to_vtk, create_equally_spaced_curves, SurfaceRZFourier
from simsopt.field import Current, coils_via_symmetries
mpi = MpiPartition()
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    def pprint(*args, **kwargs):
        if comm.rank == 0:  # only print on rank 0
            print(*args, **kwargs)
except ImportError:
    comm = None
    pprint = print
######## INPUT PARAMETERS ########
ncoils=12
CS_THRESHOLD = [0.012, 0.01, 0.006, 0.002, 0.0008, 0.0006] 
CS_WEIGHT = 1e40
max_nfev = 30
iota_target = 0.122 
iota_weight = 5e3 
aspect_target = 10.8
aspect_weight = 3e2 
quasisymmetry_weight = 5e4 
max_modes = [1,2,3,3,3,3]
rel_step = 1e-5
abs_step = 1e-7
ISTTOK_R0 = 0.46
ISTTOK_R1 = 0.085
ntheta_VMEC = 100 
nphi_VMEC = 100 
numquadpoints = 151
ftol=1e-4
diff_method = 'centered'
######## END INPUT PARAMETERS ########


### Go to results folder
results_path = os.path.join(os.path.dirname(__file__), 'results')
Path(results_path).mkdir(parents=True, exist_ok=True)
os.chdir(results_path)

### Get VMEC surface
filename = 'input.ISTELL_maxmode1'
vmec = Vmec(filename, mpi=mpi, verbose=False, ntheta=ntheta_VMEC, nphi=nphi_VMEC, range_surface='half period')
s = vmec.boundary

### Create coils
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=ISTTOK_R0, R1=ISTTOK_R1, order=2, numquadpoints=numquadpoints)
curves_to_vtk(base_curves, 'curves_init')
s.to_vtk("surf_init")

### Optimize
surf = vmec.boundary
qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1),  # Radii to target
                                helicity_m=1, helicity_n=0)  # (M, N) you want in |B|

counter = 0
for max_mode in max_modes:
    Jcsdist = CurveSurfaceDistance(base_curves, surf, CS_THRESHOLD[counter])
    pprint(f' ### Max mode = {max_mode} ### ')
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")  # Major radius

    prob = LeastSquaresProblem.from_tuples([(vmec.aspect, aspect_target, aspect_weight),
                                            (qs.residuals, 0, quasisymmetry_weight),
                                            (vmec.mean_iota, iota_target, iota_weight),
                                            (Jcsdist.J, 0, CS_WEIGHT)])
    pprint("Iota before optimization:", vmec.mean_iota())
    pprint("Distance to surfaces before optimization:", Jcsdist.shortest_distance())
    pprint("Value of Jcsdist.J before optimization:", Jcsdist.J())
    pprint("Quasisymmetry objective before optimization:", qs.total())
    pprint("Aspect ratio before optimization:", vmec.aspect())
    pprint("Total objective before optimization:", prob.objective())
    mpi.comm_world.barrier()
    least_squares_mpi_solve(prob, mpi, grad=True, rel_step=rel_step, abs_step=abs_step, max_nfev=max_nfev, diff_method=diff_method, ftol=ftol)
    pprint("Final aspect ratio:", vmec.aspect())
    pprint("Final iota:", vmec.mean_iota())
    pprint("Distance to surfaces after optimization:", Jcsdist.shortest_distance())
    pprint("Value of Jcsdist.J after optimization:", Jcsdist.J())
    pprint("Quasisymmetry objective after optimization:", qs.total())
    pprint("Aspect ratio after optimization:", vmec.aspect())
    pprint("Total objective after optimization:", prob.objective())
    s.to_vtk(f"surf_maxmode{max_mode}")
    # vmec.indata.ns_array[:3]    = [  16,    51,    101]
    # vmec.indata.niter_array[:3] = [ 2000,  3000, 20000]
    # vmec.indata.ftol_array[:3]  = [1e-14, 1e-14, 1e-14]
    vmec.write_input(f'input.ISTELL_maxmode{max_mode}')
    counter += 1
### Write result
mpi.comm_world.barrier()
if mpi.proc0_world:
    vmec.indata.ns_array[:5]    = [  16,     51,   101,   151,   201]
    vmec.indata.niter_array[:5] = [ 2000,  2000,  2000,  2000, 20000]
    vmec.indata.ftol_array[:5]  = [1e-14, 1e-14, 1e-14, 1e-14, 1e-15]
    vmec.write_input(f'input.ISTELL_final')
    vmec = Vmec('input.ISTELL_final', mpi=mpi, verbose=True, ntheta=ntheta_VMEC, nphi=nphi_VMEC*2*s.nfp)
    vmec.run()

    
    shutil.move(f"wout_ISTELL_final_000_000000.nc", f"wout_ISTELL_final.nc")
    os.remove(f'input.ISTELL_final_000_000000')
    
    s = SurfaceRZFourier.from_wout("wout_ISTELL_final.nc")
    s.to_vtk("surf_final")
    # Remove spurious files
    for objective_file in glob.glob(f"jac_*"): os.remove(objective_file)
    for objective_file in glob.glob(f"jac_*"): os.remove(objective_file)
    for objective_file in glob.glob(f"objective_*"): os.remove(objective_file)
    for objective_file in glob.glob(f"residuals_*"): os.remove(objective_file)
    for objective_file in glob.glob(f"*000_*"): os.remove(objective_file)
    for objective_file in glob.glob(f"parvmec*"): os.remove(objective_file)
    for objective_file in glob.glob(f"threed*"): os.remove(objective_file)
    

