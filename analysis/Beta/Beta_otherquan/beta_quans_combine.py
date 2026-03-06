import numpy as np
import caesar
import unyt
import matplotlib.pyplot as plt
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

which = "m25"

files = files_25 if which == 'm25' else files_50

# --- Bands ---
bands = ["i1500", "i2300", "i2800"]
wavelengths = np.array([1500, 2300, 2800])

# --- Figure with 1x6 subplots ---
fig, axes = plt.subplots(1, 6, figsize=(18, 3), sharey=True, constrained_layout=False)
# Adjust spacing
plt.subplots_adjust(left=0.05, right=0.95,bottom=0.15,wspace=0, hspace=0) 

for ax, infile in zip(axes, files):
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

    # Log stellar mass
    log_stellar_mass = np.log10(stellar_mass_cut)

    # Bin beta (median)
    bin_centers, beta_med, beta_iqr, bin_count, low = bin_xy_median(
        x_values=log_stellar_mass,
        y_values=beta,
        N_bins=10,
        min_count=10
    )

    # Scatter all galaxies, colored by i1500 magnitude
    sc = ax.scatter(log_stellar_mass, beta, c=mags_cut[0], cmap='viridis', s=10, alpha=0.5)

    # Median binned line
    plot_binned_line(bin_centers, beta_med, low, ax, color='green')

    # Labels and limits
    ax.set_title(f"z = {round(Z)}", fontsize=10)
    ax.set_xlim(7, 11)
    ax.set_ylim(-2.6, -0.8)
    if ax == axes[0]:
        ax.set_ylabel(r"$\beta$")
    ax.set_xlabel(r"log$_{10}$(M$_\star$/M$_\odot$)")
    xticks = ax.get_xticklabels()
    if len(xticks) > 0:
        xticks[-1].set_visible(False)  # hide the rightmost tick label


# Colorbar for all subplots
fig.colorbar(sc, ax=axes.ravel().tolist(), label='i1500 magnitude', shrink=0.6)

plt.savefig(f"Beta_vs_StellarMass_{which}_MEDIAN_1x6.png") 
