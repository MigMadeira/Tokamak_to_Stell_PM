import os
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np

from simsopt.util.permanent_magnet_helper_functions import *
from simsopt.mhd.vmec import Vmec
from simsopt import load
from math import sqrt,ceil
from mpi4py import MPI
from simsopt.util import MpiPartition
mpi = MpiPartition()

_, _, _, R_final, Z_final = np.load("./Poincare_plots/Data/Poincare_data_tol=1e-14_t=10000_2cm_no_diagnostics_no_quadrupole_fixed_I=72kA.npy", allow_pickle=True)
nfieldlines = 4

filename = f'poincare_ISTELL_diag'
OUT_DIR = "Poincare_plots/Figures/"
os.makedirs(OUT_DIR, exist_ok=True)

filelist = ["./Poincare_plots/Data/Poincare_data_tol=1e-14_t=10000_2cm_no_diagnostics_no_quadrupole_fixed_I=72kA.npy",
            "./Poincare_plots/Data/Poincare_data_tol=1e-14_t=10000_2cm_diag.npy",
            "./Poincare_plots/Data/Poincare_data_tol=1e-14_t=10000_2cm_diag_tol=1cm.npy",
            "./Poincare_plots/Data/Poincare_data_tol=1e-14_t=10000_2cm_diag_tol=2cm.npy",
            ]

legend = ["VMEC", "without ports", "with ports \ntol = 1 mm", "with ports \ntol = 1 cm", "with ports \ntol = 2 cm"]

markerlist=['o',"^","s","P","D"]
colorlist=["tab:blue","tab:orange","tab:green","tab:red","tab:purple"]

print('Creating Poincare plot R, Z')

counter = 0
for file in filelist:
    fieldlines_tys, fieldlines_phi_hits, phis, R_final, Z_final = np.load(file, allow_pickle=True)
    r = []
    z = []
    for izeta in range(len(phis)):
        r_2D = []
        z_2D = []
        for iradius in range(len(fieldlines_phi_hits)):
            lost = fieldlines_phi_hits[iradius][-1, 1] < 0
            data_this_phi = fieldlines_phi_hits[iradius][np.where(fieldlines_phi_hits[iradius][:, 1] == izeta)[0], :]
            if data_this_phi.size == 0:
                print(f'No Poincare data for iradius={iradius} and izeta={izeta}')
                continue
            r_2D.append(np.sqrt(data_this_phi[:, 2]**2+data_this_phi[:, 3]**2))
            z_2D.append(data_this_phi[:, 4])
        r.append(r_2D)
        z.append(z_2D)
    r = np.array(r, dtype=object)
    z = np.array(z, dtype=object)
    print('Plotting Poincare plot')

    if counter == 0:
        nrowcol = ceil(sqrt(len(phis)))
        fig, axs = plt.subplots(nrowcol, nrowcol, figsize=(8, 8))
        
    for i in range(len(phis)):
        row = i//nrowcol
        col = i % nrowcol
        axs[row, col].set_title(f"$\\phi={phis[i]/np.pi:.2f}\\pi$", loc='right', y=0.0, fontsize=10)
        axs[row, col].set_xlabel("$R$", fontsize=14)
        axs[row, col].set_ylabel("$Z$", fontsize=14)
        axs[row, col].set_aspect('equal')
        axs[row, col].tick_params(direction="in")
        
        for j in range(nfieldlines):
            if j == 1:
                for k in range(len(legend)):
                    legend[k] = '_nolegend_'
            if counter == 0:
                #print(i,j)
                axs[row, col].plot(R_final[i,j,:], Z_final[i,j,:], '-', linewidth=1.2, c='k', label = legend[0])
            try: axs[row, col].scatter(r[i][j], z[i][j], marker=markerlist[counter], s=1., c=colorlist[counter], label = legend[counter+1])
            except Exception as e: print(e, i, j)
            # if j == 0: axs[row, col].legend(loc="upper right")
    counter+=1

# plt.legend(bbox_to_anchor=(0.1, 0.9 ))
leg = fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=12)
plt.tight_layout()

plt.savefig(OUT_DIR + filename + ".pdf", bbox_inches = 'tight', pad_inches = 0)
plt.savefig(OUT_DIR + filename + ".png", bbox_inches = 'tight', pad_inches = 0)
