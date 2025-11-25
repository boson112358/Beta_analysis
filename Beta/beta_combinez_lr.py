import matplotlib.pyplot as plt
import numpy as np
import caesar
from beta_utils import *

# -------------------------------
# File lists for each box (sorted by redshift)
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

# -------------------------------
# Bands and wavelengths
# -------------------------------
bands = ["i1500", "i2300", "i2800"]
wavelengths = np.array([1500, 2300, 2800])

# -------------------------------
# Plot setup: 3x2 subplots
# -------------------------------
fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True, sharey=True)
axes = axes.flatten()

# -------------------------------
# Loop over redshifts
# -------------------------------
for i, (f25, f50) in enumerate(zip(files_25, files_50)):
    ax = axes[i]

    # Load Caesar files
    obj_25 = caesar.load(f25)
    obj_50 = caesar.load(f50)

    # Dust magnitudes
    mags_25 = np.array([[g.absmag[band] for g in obj_25.galaxies] for band in bands])
    mags_50 = np.array([[g.absmag[band] for g in obj_50.galaxies] for band in bands])

    # --- Apply magnitude cuts ---
    mask_m25 = mags_25[0] < -16       # only keep galaxies with M1500 < -16
    mask_m50 = mags_50[0] < -17.5     # only keep galaxies with M1500 < -17.5

    mags_m25_cut = mags_25[:, mask_m25]
    mags_m50_cut = mags_50[:, mask_m50]

    # --- Combine the two boxes ---
    mags_combined = np.concatenate([mags_m25_cut, mags_m50_cut], axis=1)

    # Compute beta
    beta_combined = Calbeta(mags_combined, wavelengths)
    M1500_combined = mags_combined[0]

    # Bin beta
    bin_centers, beta_mean, beta_std, bin_count = bin_beta(M1500_combined, beta_combined, N_bins=6, mag_cut=-16, min_count=5)

    # --- Linear Regression ---
    slope, intercept, x_fit, y_fit = linear_regression_fit(bin_centers, beta_mean, beta_std)

    # Optional: round for nicer display
    slope_rounded = round(slope, 3)
    intercept_rounded = round(intercept, 3)

    # Plot all curves in green
    ax.errorbar(bin_centers, beta_mean, yerr=beta_std,
                color='green', marker='o', linestyle='-')
    ax.plot(x_fit, y_fit, label=f"y = {slope_rounded}x + {intercept_rounded}")
    
    # Annotate number of galaxies in each bin
    for x, y, count in zip(bin_centers, beta_mean, bin_count):
        ax.text(x, y + 0.05, str(count), fontsize=8, ha='center', color='blue')
    
    # Titles and axis limits
    z = obj_25.simulation.redshift
    ax.set_title(f"z = {round(z)}", fontsize=12)
    ax.set_xlim(-23, -15)
    ax.set_ylim(-2.6, -0.8)
    ax.legend(fontsize=10)
    if i // 2 == 2:  # bottom row
        ax.set_xlabel("M1500")
    if i % 2 == 0:   # left column
        ax.set_ylabel("Beta")

# -------------------------------
# Final layout
# -------------------------------
plt.tight_layout()
plt.savefig("Beta_combined_subplots_lr.png")

