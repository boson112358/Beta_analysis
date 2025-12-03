import matplotlib.pyplot as plt
import numpy as np
import caesar
import pandas as pd
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

# --- List of datasets ---
datasets = [
    {"name": "Cullen2023", "file": "/cosma8/data/dp376/dc-xian3/simba-eor/Beta_analysis/Beta/Beta_obs/cullen_2023.csv"},
    {"name": "Topping2024", "file": "/cosma8/data/dp376/dc-xian3/simba-eor/Beta_analysis/Beta/Beta_obs/topping_2024.csv"}
    # Add more datasets here easily
]

# --- Load datasets into a list of dictionaries ---
obs_data = []
for ds in datasets:
    df = pd.read_csv(ds["file"])
    obs_data.append({
        "name": ds["name"],
        "MUV": df['MUV'],
        "Beta": df['Beta'],
        "Beta_err_plus": df['Beta_err_plus'],
        "Beta_err_minus": df['Beta_err_minus'],
        "zphot": df['zphot'],
        "MUV_err_plus": df.get('MUV_err_plus', pd.Series([0]*len(df))),
        "MUV_err_minus": df.get('MUV_err_minus', pd.Series([0]*len(df)))
        })

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
    
    # Titles and axis limits
    # Simulation redshift
    z = obj_25.simulation.redshift
    round_z = round(z)

    # Loop over all observed datasets
    for data in obs_data:
        # Round observed redshift and select points matching this subplot
        z_mask = data['zphot'].round() == round_z
    
        # Plot with error bars (including optional MUV errors)
        ax.errorbar(
            data['MUV'][z_mask],
            data['Beta'][z_mask],
            xerr=[data['MUV_err_minus'][z_mask], data['MUV_err_plus'][z_mask]],
            yerr=[data['Beta_err_minus'][z_mask], data['Beta_err_plus'][z_mask]],
            fmt='o',  # You can choose different markers per dataset if you want
            label=data['name'],
            alpha=0.7
        )
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
plt.savefig("Beta_combined_obs.png")

