import matplotlib.pyplot as plt
import numpy as np
import caesar
import pandas as pd
from utils.beta_utils import *

#plt.style.use("beta_pub.mplstyle")

# -------------------------------
# Redshifts and dust laws
# -------------------------------
redshifts = ['016', '019', '022', '026', '030', '036']
dust_laws = ['calzetti', 'salmon', 'smc']  # you can add any number of laws here
bands = ["i1500", "i2300", "i2800"]
wavelengths = np.array([1500, 2300, 2800])

# Assign colors for each dust law automatically
color = plt.cm.Greens(0.7)  # 0.0–1.0 scales the shade
linestyles = ['-', '-.', ':', '--']  # solid, dashed, dash-dot, dotted

# -------------------------------
# Plot setup: 3x2 subplots
# -------------------------------
fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True, sharey=True)
axes = axes.flatten()

# -------------------------------
# Observed datasets
# -------------------------------
datasets = [
    {"name": "Cullen2023", "file": "/cosma8/data/dp376/dc-xian3/simba-eor/Beta_analysis/analysis/Beta/Beta_obs/Cullen2023.csv"},
    {"name": "Topping2024", "file": "/cosma8/data/dp376/dc-xian3/simba-eor/Beta_analysis/analysis/Beta/Beta_obs/Topping2024.csv"},
    {"name": "Nanayakkara2023", "file": "/cosma8/data/dp376/dc-xian3/simba-eor/Beta_analysis/analysis/Beta/Beta_obs/Nanayakkara2023.csv"},
    {"name": "Mitsuhashi2025", "file": "/cosma8/data/dp376/dc-xian3/simba-eor/Beta_analysis/analysis/Beta/Beta_obs/Mitsuhashi2025.csv"},
    {"name": "Bouwens2014", "file": "/cosma8/data/dp376/dc-xian3/simba-eor/Beta_analysis/analysis/Beta/Beta_obs/Bouwens2014.csv"},
    {"name": "Cullen2024", "file": "/cosma8/data/dp376/dc-xian3/simba-eor/Beta_analysis/analysis/Beta/Beta_obs/Cullen2024.csv"},
    {"name": "Bhatawdekar2021", "file": "/cosma8/data/dp376/dc-xian3/simba-eor/Beta_analysis/analysis/Beta/Beta_obs/Bhatawdekar2021.csv"},
    ]

obs_data = []
obs_markers = [
    'o',   # circle
    's',   # square
    '^',   # triangle up
    'v',   # triangle down
    'D',   # diamond
    '>',   # triangle right
    '<',   # triangle left
    'p',   # pentagon
    '*',   # star
    'h'    # hexagon
]
obs_colors = ['#1f77b4','#d62728','#2ca02c','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']

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
for i, z in enumerate(redshifts):
    ax = axes[i]

    # Loop over dust laws
    for j, law in enumerate(dust_laws):
        # Load Caesar files for m25 and m50
        f25 = f"/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/Dust_extin/m25n1024/caesar_m25n1024_{z}_{law}.hdf5"
        f50 = f"/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/Dust_extin/m50n1024/caesar_m50n1024_{z}_{law}.hdf5"

        obj25 = caesar.load(f25)
        obj50 = caesar.load(f50)

        # Dust magnitudes
        mags_25 = np.array([[g.absmag[band] for g in obj25.galaxies] for band in bands])
        mags_50 = np.array([[g.absmag[band] for g in obj50.galaxies] for band in bands])

        # Apply magnitude cuts
        mask_25 = mags_25[0] < -16
        mask_50 = mags_50[0] < -17.5
        mags_combined = np.concatenate([mags_25[:, mask_25], mags_50[:, mask_50]], axis=1)

        # Compute beta
        beta_combined = Calbeta(mags_combined, wavelengths)
        M1500_combined = mags_combined[0]

        # Bin beta
        bin_centers, beta_mean, beta_std, _ = bin_beta(M1500_combined, beta_combined, N_bins=6, mag_cut=-16, min_count=5)

        # Plot
        #ax.errorbar(bin_centers, beta_mean, yerr=beta_std, color=color, marker='o', linestyle=linestyles[j], label=law)
        ax.plot(bin_centers, beta_mean,
        color=color,
        linestyle=linestyles[j],
        label=law)

    # Plot nodust line
    # Extract no-dust magnitudes
    mags25_nd = np.array([[g.absmag_nodust[band] for g in obj25.galaxies] for band in bands])
    mags50_nd = np.array([[g.absmag_nodust[band] for g in obj50.galaxies] for band in bands])

    # Apply same magnitude cuts
    mask25_nd = mags25_nd[0] < -16
    mask50_nd = mags50_nd[0] < -17.5

    mags_nd_combined = np.concatenate([mags25_nd[:, mask25_nd],
                                       mags50_nd[:, mask50_nd]], axis=1)

    # Compute beta (no dust)
    beta_nd = Calbeta(mags_nd_combined, wavelengths)
    M1500_nd = mags_nd_combined[0]

    # Bin beta vs M1500 (no dust)
    bin_centers_nd, beta_mean_nd, beta_std_nd, _ = bin_beta(
        M1500_nd, beta_nd, N_bins=6, mag_cut=-16, min_count=5
    )

    ax.plot(bin_centers_nd, beta_mean_nd,
        color=color,
        linestyle='--',
        label='nodust')

    # Plot observed datasets
    round_z = round(obj25.simulation.redshift)
    for k, data in enumerate(obs_data):
        z_mask = data['zphot'].round() == round_z
        ax.errorbar(
            data['MUV'][z_mask],
            data['Beta'][z_mask],
            xerr=[data['MUV_err_minus'][z_mask], data['MUV_err_plus'][z_mask]],
            yerr=[data['Beta_err_minus'][z_mask], data['Beta_err_plus'][z_mask]],
            fmt=obs_markers[k % len(obs_markers)],
            color=obs_colors[k % len(obs_colors)],
            alpha=0.7,
            label=data['name']
        )

    # Titles and labels
    ax.set_title(f"z = {round_z}", fontsize=12)
    ax.set_xlim(-23, -15)
    ax.set_ylim(-2.6, -0.8)
    if i // 2 == 2:  # bottom row
        ax.set_xlabel("M1500")
    if i % 2 == 0:   # left column
        ax.set_ylabel("Beta")
    ax.legend(fontsize=8)

# -------------------------------
# Final layout and save
# -------------------------------
plt.tight_layout()
plt.savefig("Beta_multiple_dust_laws.png", dpi=300)

