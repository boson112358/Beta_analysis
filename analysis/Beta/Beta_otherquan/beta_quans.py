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
fig, axes = plt.subplots(4, 7, figsize=(18, 12), sharey=True, sharex='row', constrained_layout=False)
# Adjust spacing
plt.subplots_adjust(left=0.05, right=0.95,bottom=0.15,wspace=0, hspace=0.3) 

for i, infile in enumerate(files):
    # Load Caesar file
    obj = caesar.load(infile)
    Z = obj.simulation.redshift

    # Stellar mass
    stellar_mass = np.array([g.masses['stellar'] for g in obj.galaxies])

    # Magnitude cut
    if which == 'm25':
        mask = np.array([g.absmag["i1500"] for g in obj.galaxies]) < -16
    elif which == 'm50':
        mask = np.array([g.absmag["i1500"] for g in obj.galaxies]) < -17.5

    stellar_mass_cut = stellar_mass[mask]

    # Magnitudes for beta
    mags_cut = np.array([[g.absmag[band] for g in obj.galaxies] for band in bands])[:, mask]

    # Compute beta
    beta = Calbeta(mags_cut, wavelengths)

    # SFR
    sfr = np.array([g.sfr for g in obj.galaxies])[mask]  # apply same mask

    # ---- Metallicity (choose gas or stellar) ----
    metallicity = np.array([g.metallicities['mass_weighted'] for g in obj.galaxies])[mask]

    # Dust mass (NEW)
    dust_mass = np.array([g.masses['dust'] for g in obj.galaxies])[mask]
    log_dustbefore = np.log10(dust_mass + 1e-12)

    # Log stellar mass
    log_stellar_mass = np.log10(stellar_mass_cut)
    log_sfr = np.log10(sfr + 1e-10)  # avoid log(0)
    log_Zbefore = np.log10(metallicity + 1e-12)   # avoid log(0)

    # Keep only logZ >= -4
    mask_Z = log_Zbefore >= -4

    log_Z = log_Zbefore[mask_Z]
    beta_Zs = beta[mask_Z]
    
    # Mask dust because there are some extremely low dust values
    mask_dust = log_dustbefore >= 0
    log_dust = log_dustbefore[mask_dust]
    beta_dusts = beta[mask_dust]

    # Bin beta (median)
    bin_centers_mass, beta_med_mass, beta_iqr_mass, bin_count_mass, low_mass = bin_xy_median(
        x_values=log_stellar_mass,
        y_values=beta,
        N_bins=10,
        min_count=10
    )

    # --- Bin beta vs SFR (median) ---
    bin_centers_sfr, beta_med_sfr, beta_iqr_sfr, bin_count_sfr, low_sfr = bin_xy_median(
        x_values=log_sfr,
        y_values=beta,
        N_bins=10,
        min_count=10
    )

    # Beta vs metallicity
    bin_Z, beta_Z, _, _, low_Z = bin_xy_median(
        x_values=log_Z,
        y_values=beta_Zs,
        N_bins=10,
        min_count=10
    )

    bin_dust,beta_dust_med, _, _, low_dust  = bin_xy_median(log_dust, beta_dusts, N_bins=10, min_count=10)
    
    # -------------------------
    # Top row: beta vs stellar mass
    # -------------------------
    ax_top = axes[0, i]
    sc = ax_top.scatter(log_stellar_mass, beta, c=mags_cut[0], cmap='viridis', s=10, alpha=0.5)
    plot_binned_line(bin_centers_mass, beta_med_mass, low_mass, ax_top, color=colors[i])
    ax_top.set_title(f"z = {round(Z)}", fontsize=10)
    ax_top.set_xlim(7, 10)
    ax_top.set_ylim(-2.6, -1.5)
    if i == 0:
        ax_top.set_ylabel(r"$\beta$")
    # hide rightmost x-tick
    xticks = ax_top.get_xticklabels()
    if len(xticks) > 0:
        xticks[-1].set_visible(False)
    ax_top.set_xlabel(r"log$_{10}$(M$_\star$/M$_\odot$)")

    ax_last = axes[0, 6]
    plot_binned_line(bin_centers_mass, beta_med_mass, low_mass, ax_last, color=colors[i])

    # -------------------------
    # Second row: beta vs SFR
    # -------------------------
    ax_bottom = axes[1, i]
    ax_bottom.scatter(log_sfr, beta, c=mags_cut[0], cmap='viridis', s=10, alpha=0.5)
    plot_binned_line(bin_centers_sfr, beta_med_sfr, low_sfr, ax_bottom, color=colors[i])
    ax_bottom.set_ylim(-2.6, -1.5)
    ax_bottom.set_xlim(-2, 2)
    if i == 0:
        ax_bottom.set_ylabel(r"$\beta$")
    # hide rightmost x-tick
    xticks = ax_bottom.get_xticklabels()
    if len(xticks) > 0:
        xticks[-1].set_visible(False)
    ax_bottom.set_xlabel(r"log$_{10}$(SFR)")

    ax_last = axes[1, 6]
    plot_binned_line(bin_centers_sfr, beta_med_sfr, low_sfr, ax_last, color=colors[i])

    # -----------------------------------------------------
    # BOTTOM ROW — β vs metallicity
    # -----------------------------------------------------
    ax_bot = axes[2, i]
    ax_bot.scatter(log_Zbefore, beta, c=mags_cut[0], cmap='viridis', s=10, alpha=0.5)
    plot_binned_line(bin_Z, beta_Z, low_Z, ax_bot, color=colors[i])
    ax_bot.set_ylim(-2.6, -1.5)
    ax_bot.set_xlim(-4, -1.5)   # adjust if needed
    if i == 0:
        ax_bot.set_ylabel(r"$\beta$")
    xticks = ax_bot.get_xticklabels()
    if len(xticks) > 0:
        xticks[-1].set_visible(False)
    ax_bot.set_xlabel(r"log$_{10}$(Z)")

    ax_last = axes[2, 6]
    plot_binned_line(bin_Z, beta_Z, low_Z, ax_last, color=colors[i])

    # --------------------------
    # ROW 4 — β vs DUST MASS (NEW)
    # --------------------------
    ax = axes[3, i]
    ax.scatter(log_dustbefore, beta, c=mags_cut[0], cmap='viridis', s=10, alpha=0.5)
    plot_binned_line(bin_dust, beta_dust_med, low_dust, ax, color=colors[i])
    ax.set_xlim(3, 8)
    ax.set_ylim(-2.6, -1.5)
    if i == 0: ax.set_ylabel(r"$\beta$")
    ax.set_xlabel(r"log$_{10}$(M$_\mathrm{dust}$/M$_\odot$)")

    ticks = ax.get_xticklabels()
    if ticks: ticks[-1].set_visible(False)

    ax_last = axes[3, 6]
    plot_binned_line(bin_dust, beta_dust_med, low_dust, ax_last, color=colors[i])

# Colorbar for all subplots
fig.colorbar(sc, ax=axes.ravel().tolist(), label='i1500 magnitude', shrink=0.6)

plt.savefig(f"Beta_vs_StellarMass_{which}_MEDIAN_4x6.png") 
