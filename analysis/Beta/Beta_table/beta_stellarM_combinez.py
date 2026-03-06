import numpy as np
import caesar
import pandas as pd
import matplotlib.pyplot as plt
from utils.beta_utils import *

# -------------------------------
# File lists
# -------------------------------
files_25 = sorted([
    '/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/CaesarFile/m25n1024/' + f
    for f in ['caesar_m25n1024_016.hdf5', 'caesar_m25n1024_019.hdf5',
              'caesar_m25n1024_022.hdf5', 'caesar_m25n1024_026.hdf5',
              'caesar_m25n1024_030.hdf5', 'caesar_m25n1024_036.hdf5']
])

files_50 = sorted([
    '/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/CaesarFile/m50n1024/' + f
    for f in ['caesar_m50n1024_016.hdf5', 'caesar_m50n1024_019.hdf5',
              'caesar_m50n1024_022.hdf5', 'caesar_m50n1024_026.hdf5',
              'caesar_m50n1024_030.hdf5', 'caesar_m50n1024_036.hdf5']
])

bands = ["i1500", "i2300", "i2800"]
wavelengths = np.array([1500, 2300, 2800])

# -------------------------------
# Create subplot grid
# -------------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

# -------------------------------
# Loop over snapshots
# -------------------------------
for i, (f25, f50) in enumerate(zip(files_25, files_50)):

    ax = axes[i]

    obj = caesar.load(f25)
    obj_m50 = caesar.load(f50)

    Z = obj.simulation.redshift

    # --- Stellar masses ---
    stellar_mass_m25 = np.array([g.masses['stellar'] for g in obj.galaxies])
    stellar_mass_m50 = np.array([g.masses['stellar'] for g in obj_m50.galaxies])

    # --- Magnitude cuts ---
    mask_m25 = np.array([g.absmag["i1500"] for g in obj.galaxies]) < -16
    mask_m50 = np.array([g.absmag["i1500"] for g in obj_m50.galaxies]) < -17.5

    stellar_mass_m25_cut = stellar_mass_m25[mask_m25]
    stellar_mass_m50_cut = stellar_mass_m50[mask_m50]

    # --- Combine stellar mass ---
    stellar_mass_combined = np.concatenate([stellar_mass_m25_cut, stellar_mass_m50_cut])

    # --- Magnitudes ---
    mags_m25 = np.array([[g.absmag[band] for g in obj.galaxies] for band in bands])
    mags_m50 = np.array([[g.absmag[band] for g in obj_m50.galaxies] for band in bands])

    mags_combined = np.concatenate([
        mags_m25[:, mask_m25],
        mags_m50[:, mask_m50]
    ], axis=1)

    # --- Compute beta ---
    beta_combined = Calbeta(mags_combined, wavelengths)

    # --- log stellar mass ---
    log_stellar_mass = np.log10(stellar_mass_combined)

    # --- Bin ---
    bin_centers, beta_mean, beta_std, bin_count = bin_xy(
        x_values=log_stellar_mass,
        y_values=beta_combined,
        mask_values=None,
        mask_cut=None,
        N_bins=10
    )

    df = pd.DataFrame({
        "z": [round(Z)] * len(bin_centers),
        "log_stellar_mass": bin_centers,
        "beta": beta_mean,
        "beta_err": beta_std
    })

    df.to_csv(f"beta_stellar_mass_z{round(Z)}.csv", index=False)

    # --- Plot ---
    ax.errorbar(bin_centers, beta_mean, yerr=beta_std,
                color='green', linestyle='-', marker='o')

    ax.set_xlim(7, 11)
    ax.set_ylim(-2.6, -0.8)

    ax.text(7.2, -1, f"z = {round(Z)}", fontsize=11)

    if i >= 3:
        ax.set_xlabel(r"log$_{10}$(Stellar Mass / M$_\odot$)")

    if i % 3 == 0:
        ax.set_ylabel(r"$\beta$")

# -------------------------------
# Layout
# -------------------------------
plt.tight_layout()
plt.savefig("Beta_vs_StellarMass_all_redshifts.png", dpi=300)
