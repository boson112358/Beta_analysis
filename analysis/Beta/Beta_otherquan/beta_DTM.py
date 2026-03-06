import numpy as np
import caesar
import unyt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils.beta_utils import *
from plot_utils import *

# --- File lists ---
files_25 = sorted([
    '/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/CaesarFile/m25n1024/' + f
    for f in [
        'caesar_m25n1024_016.hdf5',
        'caesar_m25n1024_019.hdf5',
        'caesar_m25n1024_022.hdf5',
        'caesar_m25n1024_026.hdf5',
        'caesar_m25n1024_030.hdf5',
        'caesar_m25n1024_036.hdf5'
    ]
])

files_50 = sorted([
    '/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/CaesarFile/m50n1024/' + f
    for f in [
        'caesar_m50n1024_016.hdf5',
        'caesar_m50n1024_019.hdf5',
        'caesar_m50n1024_022.hdf5',
        'caesar_m50n1024_026.hdf5',
        'caesar_m50n1024_030.hdf5',
        'caesar_m50n1024_036.hdf5'
    ]
])

which = "m50"

files = files_25 if which == 'm25' else files_50

# --- Bands ---
bands = ["i1500", "i2300", "i2800"]
wavelengths = np.array([1500, 2300, 2800])

colors = cm.viridis(np.linspace(0, 1, len(files)))  # color gradient

# --- Figure with 1x6 subplots ---
fig, axes = plt.subplots(3, 7, figsize=(18, 12), sharey=True, sharex='row', constrained_layout=False)
# Adjust spacing
plt.subplots_adjust(left=0.05, right=0.95,bottom=0.15,wspace=0, hspace=0.3) 

for i, infile in enumerate(files):
    # Load Caesar file
    obj = caesar.load(infile)
    Z = obj.simulation.redshift

    # Magnitude cut
    if which == 'm25':
        mask = np.array([g.absmag["i1500"] for g in obj.galaxies]) < -16
    elif which == 'm50':
        mask = np.array([g.absmag["i1500"] for g in obj.galaxies]) < -17.5

    # Magnitudes for beta
    mags_cut = np.array([[g.absmag[band] for g in obj.galaxies] for band in bands])[:, mask]

    # Compute beta
    beta = Calbeta(mags_cut, wavelengths)

    # ---- Metallicity (choose gas or stellar) ----
    metallicity = np.array([g.metallicities['mass_weighted'] for g in obj.galaxies])[mask]

    # Dust mass (NEW)
    dust_mass = np.array([g.masses['dust'].to('Msun') for g in obj.galaxies])[mask]
    gas_mass = np.array([g.masses['gas'].to('Msun') for g in obj.galaxies])[mask]
    log_dustbefore = np.log10(dust_mass + 1e-12)

    # Log metalicity
    log_Zbefore = np.log10(metallicity + 1e-12)   # avoid log(0)

    # Keep only logZ >= -4
    mask_Z = log_Zbefore >= -4
    log_Z = log_Zbefore[mask_Z]
    beta_Zs = beta[mask_Z]
    
    # Mask dust because there are some extremely low dust values
    mask_dust = log_dustbefore >= 0
    log_dust = log_dustbefore[mask_dust]
    beta_dusts = beta[mask_dust]

    # dust to metal ratio
    DTM = dust_mass / (gas_mass * metallicity)
    log_DTMbefore = np.log10(DTM + 1e-12)

    mask_DTM = log_DTMbefore >= -3
    log_DTM = log_DTMbefore[mask_DTM]
    beta_DTMs = beta[mask_DTM]

    # Beta vs metallicity
    bin_Z, beta_Z, _, _, low_Z = bin_xy_median(
        x_values=log_Z,
        y_values=beta_Zs,
        N_bins=10,
        min_count=10
    )

    bin_dust,beta_dust_med, _, _, low_dust  = bin_xy_median(log_dust, beta_dusts, N_bins=10, min_count=10)
    
    bin_DTM, beta_DTM_med, _, _, low_DTM = bin_xy_median(
    x_values=log_DTM,
    y_values=beta_DTMs,
    N_bins=10,
    min_count=10
    )

    print(bin_DTM)
    print(beta_DTM_med)
    # -------------------------
    # Top row 1: beta vs stellar mass
    # -------------------------
    ax_bot = axes[0, i]
    sc=ax_bot.scatter(log_Zbefore, beta, c=mags_cut[0], cmap='viridis', s=10, alpha=0.5)
    plot_binned_line(bin_Z, beta_Z, low_Z, ax_bot, color=colors[i])
    ax_bot.set_ylim(-2.6, -1.5)
    ax_bot.set_xlim(-4, -1.5)   # adjust if needed
    if i == 0:
        ax_bot.set_ylabel(r"$\beta$")
    xticks = ax_bot.get_xticklabels()
    if len(xticks) > 0:
        xticks[-1].set_visible(False)
    ax_bot.set_xlabel(r"log$_{10}$(Z)")

    ax_last = axes[0, 6]
    plot_binned_line(bin_Z, beta_Z, low_Z, ax_last, color=colors[i])

    # --------------------------
    # ROW 2 — β vs DUST MASS (NEW)
    # --------------------------
    ax = axes[1, i]
    ax.scatter(log_dustbefore, beta, c=mags_cut[0], cmap='viridis', s=10, alpha=0.5)
    plot_binned_line(bin_dust, beta_dust_med, low_dust, ax, color=colors[i])
    ax.set_xlim(3, 8)
    ax.set_ylim(-2.6, -1.5)
    if i == 0: ax.set_ylabel(r"$\beta$")
    ax.set_xlabel(r"log$_{10}$(M$_\mathrm{dust}$/M$_\odot$)")

    ticks = ax.get_xticklabels()
    if ticks: ticks[-1].set_visible(False)

    ax_last = axes[1, 6]
    plot_binned_line(bin_dust, beta_dust_med, low_dust, ax_last, color=colors[i])

    # --------------------------
    # ROW 3 — β vs DTM
    # --------------------------
    ax_DTM = axes[2, i]  # or row index if you have more rows
    ax_DTM.scatter(log_DTMbefore, beta, c=mags_cut[0], cmap='viridis', s=10, alpha=0.5)
    if i == 5:
        plot_binned_line(bin_DTM[1:], beta_DTM_med[1:], low_DTM, ax_DTM, color=colors[i])
    else:
        plot_binned_line(bin_DTM, beta_DTM_med, low_DTM, ax_DTM, color=colors[i])
    ax_DTM.set_xlim(-2, 0)  # adjust to DTM range
    ax_DTM.set_ylim(-2.6, -1.5)
    if i == 0:
        ax_DTM.set_ylabel(r"$\beta$")
    ax_DTM.set_xlabel(r"log$_{10}$(DTM)")

    # Plot all redshifts combined in last column
    ax_last = axes[2, -1]
    if i==5:
        plot_binned_line(bin_DTM[1:], beta_DTM_med[1:], low_DTM, ax_last, color=colors[i])
    else:
        plot_binned_line(bin_DTM, beta_DTM_med, low_DTM, ax_last, color=colors[i])
# Colorbar for all subplots
fig.colorbar(sc, ax=axes.ravel().tolist(), label='i1500 magnitude', shrink=0.6)

plt.savefig(f"Beta_vs_DTM_{which}_MEDIAN_3x6.png") 
