#!/usr/bin/env python3
import os
import numpy as np
import booz_xform as bx
from pathlib import Path
import matplotlib.pyplot as plt
from simsopt.mhd import Vmec, Boozer
from qsc import Qsc
#### INPUT PARAMETERS ####
max_s_for_fit = 0.6
boozxform_nsurfaces = 20
boozer_mpol = 64
boozer_ntor = 64
N_phi = 151
max_n_to_plot = 3
show_fit_plot = True
plot_boozer = True
#### RUN ####
current_path = str(Path(__file__).parent.resolve())
print('Loading ISTTOK wout file')
vmec = Vmec(os.path.join(current_path, 'results', f'wout_ISTTOK_final.nc'))
# vmec = Vmec(os.path.join(current_path, 'results', f'input.ISTTOK_final'))
# vmec.indata.mpol            = 8
# vmec.indata.ntor            = 8
# vmec.indata.ns_array[:5]    = [  16,    51,    101,   151,   201]
# vmec.indata.niter_array[:5] = [ 2000,  2000,  2000,  2000, 20000]
# vmec.indata.ftol_array[:5]  = [1e-12, 1e-13, 1e-14, 1e-15, 1e-15]
# vmec.run()
print('Creating Boozer class for vmec_final')
b1 = Boozer(vmec, mpol=boozer_mpol, ntor=boozer_ntor)
print('Defining surfaces where to compute Boozer coordinates')
booz_surfaces = np.linspace(0,1,boozxform_nsurfaces,endpoint=False)
print(f' booz_surfaces={booz_surfaces}')
b1.register(booz_surfaces)
print('Running BOOZ_XFORM')
b1.run()
# b1.bx.write_boozmn(os.path.join(current_path, 'results', "boozmn_ISTTOK_final.nc"))
if plot_boozer:
    print("Plot BOOZ_XFORM")
    fig = plt.figure(); bx.surfplot(b1.bx, js=1,  fill=False, ncontours=35)
    plt.savefig(os.path.join(current_path, "Boozxform_surfplot_1_ISTTOK_final.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); bx.surfplot(b1.bx, js=int(boozxform_nsurfaces/2), fill=False, ncontours=35)
    plt.savefig(os.path.join(current_path, 'results', "Boozxform_surfplot_2_ISTTOK_final.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); bx.surfplot(b1.bx, js=boozxform_nsurfaces-1, fill=False, ncontours=35)
    plt.savefig(os.path.join(current_path, 'results', "Boozxform_surfplot_3_ISTTOK_final.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); bx.symplot(b1.bx, helical_detail = False, sqrts=True)
    plt.savefig(os.path.join(current_path, "Boozxform_symplot_ISTTOK_final.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); bx.modeplot(b1.bx, sqrts=True); plt.xlabel(r'$s=\psi/\psi_b$')
    plt.savefig(os.path.join(current_path, 'results', "Boozxform_modeplot_ISTTOK_final.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
print('Obtaining near-axis components')
stel = Qsc(rc=vmec.wout.raxis_cc,zs=-vmec.wout.zaxis_cs,etabar=1,nphi=N_phi,nfp=vmec.wout.nfp)
nNormal = stel.iotaN - stel.iota
# Prepare coordinates for fit
s_full = np.linspace(0,1,b1.bx.ns_in)
ds = s_full[1] - s_full[0]
#s_half = s_full[1:] - 0.5*ds
s_half = s_full[b1.bx.compute_surfs+1] - 0.5*ds
mask = s_half < max_s_for_fit
s_fine = np.linspace(0,1,400)
sqrts_fine = s_fine
phi = np.linspace(0,2*np.pi / vmec.wout.nfp, N_phi)
B0  = np.zeros(N_phi)
B1s = np.zeros(N_phi)
B1c = np.zeros(N_phi)
B20 = np.zeros(N_phi)
B2s = np.zeros(N_phi)
B2c = np.zeros(N_phi)
# Perform fit
numRows=3
numCols=max_n_to_plot*2+1
fig=plt.figure(num=None, figsize=(16, 9), dpi=80, facecolor='w', edgecolor='k')
for jmn in range(len(b1.bx.xm_b)):
    m = b1.bx.xm_b[jmn]
    n = b1.bx.xn_b[jmn] / vmec.wout.nfp
    if m>2:
        continue
    doplot = (np.abs(n) <= max_n_to_plot) & show_fit_plot
    row = m
    col = n+max_n_to_plot
    if doplot:
        plt.subplot(int(numRows),int(numCols),int(row*numCols + col + 1))
        plt.plot(np.sqrt(s_half), b1.bx.bmnc_b[jmn, :],'.-')
        # plt.xlabel(r'$\sqrt{s}$')
        plt.title('bmnc(m='+str(m)+' n='+str(n)+')')
    if m==0:
        # For m=0, fit a polynomial in s (not sqrt(s)) that does not need to go through the origin.
        degree = 4
        p = np.polyfit(s_half[mask], b1.bx.bmnc_b[jmn, mask], degree)
        B0 += p[-1] * np.cos(n*vmec.wout.nfp*phi)
        B20 += p[-2] * np.cos(2*n*vmec.wout.nfp*phi)
        if doplot:
            plt.plot(np.sqrt(s_fine), np.polyval(p, s_fine),'r')
    if m==1:
        # For m=1, fit a polynomial in sqrt(s) to an odd function
        x1 = np.sqrt(s_half[mask])
        y1 = b1.bx.bmnc_b[jmn,mask]
        x2 = np.concatenate((-x1,x1))
        y2 = np.concatenate((-y1,y1))
        degree = 5
        p = np.polyfit(x2,y2, degree)
        B1c += p[-2] * (np.sin(n*vmec.wout.nfp*phi) * np.sin(nNormal*phi) + np.cos(n*vmec.wout.nfp*phi) * np.cos(nNormal*phi))
        B1s += p[-2] * (np.sin(n*vmec.wout.nfp*phi) * np.cos(nNormal*phi) - np.cos(n*vmec.wout.nfp*phi) * np.sin(nNormal*phi))
        if doplot:
            plt.plot(sqrts_fine, np.polyval(p, sqrts_fine),'r')
    if m==2:
        # For m=2, fit a polynomial in s (not sqrt(s)) that does need to go through the origin.
        x1 = s_half[mask]
        y1 = b1.bx.bmnc_b[jmn,mask]
        degree = 4
        p = np.polyfit(x1,y1, degree)
        B2c += p[-2] * (np.sin(2*n*vmec.wout.nfp*phi) * np.sin(2*nNormal*phi) + np.cos(2*n*vmec.wout.nfp*phi) * np.cos(2*nNormal*phi))
        B2s += p[-2] * (np.sin(2*n*vmec.wout.nfp*phi) * np.cos(2*nNormal*phi) - np.cos(2*n*vmec.wout.nfp*phi) * np.sin(2*nNormal*phi))
        if doplot:
            plt.plot(np.sqrt(s_fine), np.polyval(p, s_fine),'r')
plt.savefig(os.path.join('results','ISTELL_nearaxis_fit.pdf'))
# if show:
#     plt.show()
# plt.close()
# Convert expansion in sqrt(s) to an expansion in r
BBar = np.mean(B0)
Psi_a = np.abs(vmec.wout.phi[-1])
sqrt_s_over_r = np.sqrt(np.pi * BBar / Psi_a)
B1s *= -sqrt_s_over_r
B1c *= -sqrt_s_over_r
B20 *= sqrt_s_over_r*sqrt_s_over_r
B2c *= sqrt_s_over_r*sqrt_s_over_r
B2s *= sqrt_s_over_r*sqrt_s_over_r
print('Plotting B0, B1 and B2')
eta_bar = np.mean(B1c) / BBar
r_boundary  = np.sqrt(Psi_a/(np.pi*BBar))
p2 = 0
I2 = 0
sigma0 = 0
stel = Qsc(rc=vmec.wout.raxis_cc,zs=-vmec.wout.zaxis_cs,etabar=eta_bar,nphi=N_phi,nfp=vmec.wout.nfp,B0=BBar,sigma0=sigma0, I2=I2, B2c=np.mean(B2c), order='r3', p2=p2)
print('etabar from fit',eta_bar)
print('VMEC iota = ',vmec.wout.iotaf[0])
print('QSC  iota = ',stel.iota)
# stel.plot_boundary(r=r_boundary)
# exit()
numRows=1
numCols=3
fig=plt.figure(num=None, figsize=(16, 9), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(numRows,numCols,1)
plt.plot(phi,B0, label='B0')
plt.plot(phi,[BBar]*len(phi), label='BBar')
plt.plot(phi,[stel.B0]*len(phi), label='Near-Axis B0')
plt.xlabel(r'$\phi$')
plt.ylabel(r'$B_0$')
plt.legend()
plt.subplot(numRows,numCols,2)
plt.plot(phi,B1c, label='B1c')
plt.plot(phi,B1s, label='B1s')
plt.plot(phi,[stel.etabar * stel.B0]*len(phi), linewidth=2, label='Near-Axis B1c')
plt.plot(phi,[0]*len(phi), linewidth=2, label='Near-Axis B1s')
plt.plot(phi,[np.mean(B1c)]*len(phi), '--', linewidth=2, label='Mean(B1c)')
plt.plot(phi,[np.mean(B1s)]*len(phi), '--', linewidth=2, label='Mean(B1s)')
plt.xlabel(r'$\phi$')
plt.ylabel(r'$B_1$')
plt.legend()
plt.subplot(numRows,numCols,3)
plt.plot(phi,B20, label='B20')
plt.plot(phi,stel.B20, label='Near-Axis B20')
plt.plot(phi,B2c, label='B2c')
plt.plot(phi,B2s, label='B2s')
plt.plot(phi,[stel.B2c]*len(phi), linewidth=2, label='Near-Axis B2c')
plt.plot(phi,[stel.B2s]*len(phi), linewidth=2, label='Near-Axis B2s')
plt.plot(phi,[np.mean(B2c)]*len(phi), '--', linewidth=2, label='Mean(B2c)')
plt.plot(phi,[np.mean(B2s)]*len(phi), '--', linewidth=2, label='Mean(B2s)')
plt.xlabel(r'$\phi$')
plt.ylabel(r'$B_2$')
plt.legend()
plt.savefig('ISTELL_B0B1B2.pdf')
plt.legend()

# plt.show()
# exit()

print('Calculating difference between near-axis and VMEC geometries')
ntheta = 200
nradius = 12
phi0 = 0#2*np.pi/vmec.wout.nfp/2
theta = np.linspace(0,2*np.pi,num=ntheta)
iradii = np.linspace(0,vmec.wout.ns-1,num=nradius).round()
iradii = [int(i) for i in iradii]
R = np.zeros((ntheta,nradius))
Z = np.zeros((ntheta,nradius))
for itheta in range(ntheta):
    for iradius in range(nradius):
        for imode in range(len(vmec.wout.xn)):
            angle = vmec.wout.xm[imode]*theta[itheta] - vmec.wout.xn[imode]*phi0
            R[itheta,iradius] += vmec.wout.rmnc[imode,iradii[iradius]]*np.cos(angle)
            Z[itheta,iradius] += vmec.wout.zmns[imode,iradii[iradius]]*np.sin(angle)
Raxis = 0
Zaxis = 0
for n in range(vmec.wout.ntor+1):
    angle = -n*vmec.wout.nfp*phi0
    Raxis += vmec.wout.raxis_cc[n]*np.cos(angle)
    Zaxis += vmec.wout.zaxis_cs[n]*np.sin(angle)

R0axis = stel.R0_func(phi0)
Z0axis = stel.Z0_func(phi0)
iota0  = stel.iota
fig = plt.figure()
print('Getting near-axis boundary')
for i, psi in enumerate(vmec.wout.phi[iradii][1:]):
    _, _, z_2D_plot, R_2D_plot = stel.get_boundary(r=np.sqrt(abs(psi)/(np.pi*BBar)), ntheta=80, nphi=1, ntheta_fourier=20)
    plt.plot(R_2D_plot[:,0], z_2D_plot[:,0], '--k', label = f'Near-Axis surfaces' if i==0 else '_nolegend_')
print('Done, plotting the rest')
for i, iradius in enumerate(range(nradius)):
    plt.plot(R[:,iradius], Z[:,iradius], '-r', label = f'VMEC surfaces' if i==0 else '_nolegend_')
plt.plot([R0axis],[Z0axis],'*g', markersize=14, label='Near-Axis axis')
plt.plot([Raxis],[Zaxis],'.b', markersize=14, label='VMEC axis')
plt.xlabel('R', fontsize=10)
plt.ylabel('Z', fontsize=10)
plt.legend()
plt.title(f'B0={BBar:.2f}, etabar={eta_bar:.2f}, B2c={np.mean(B2c):.2f}, nfp={vmec.wout.nfp}')
plt.savefig('nearaxis_ISTELL.pdf')
plt.show()
