import numpy as np
import caesar
import matplotlib.pyplot as plt
from utils.beta_utils import *

# -----------------------------
# Inputs
# -----------------------------
dust_laws = ['calzetti', 'salmon', 'smc']
bands = ["i1500", "i2300", "i2800"]
wavelengths = np.array([1500, 2300, 2800])

# m25/m50 Caesar file template
template_m25 = '/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/Dust_extin/m25n1024/caesar_m25n1024_036_{}.hdf5'
template_m50 = '/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/Dust_extin/m50n1024/caesar_m50n1024_036_{}.hdf5'

# Line styles for dust laws
linestyles = ['-', '-.', ':', '--']  # solid, dash-dot, dotted, dashed
color = 'green'  # same color for all

# -----------------------------
# Plot setup
# -----------------------------
plt.figure(figsize=(6,4))

for j, law in enumerate(dust_laws):
    # Load Caesar files
    obj_m25 = caesar.load(template_m25.format(law))
    obj_m50 = caesar.load(template_m50.format(law))

    # Magnitudes
    mags_m25 = np.array([[g.absmag[band] for g in obj_m25.galaxies] for band in bands])
    mags_m50 = np.array([[g.absmag[band] for g in obj_m50.galaxies] for band in bands])

    # Stellar mass
    stellar_mass_m25 = np.array([g.masses['stellar'] for g in obj_m25.galaxies])
    stellar_mass_m50 = np.array([g.masses['stellar'] for g in obj_m50.galaxies])

    # Apply magnitude cuts
    mask_m25 = mags_m25[0] < -16
    mask_m50 = mags_m50[0] < -17.5

    stellar_mass_combined = np.concatenate([stellar_mass_m25[mask_m25], stellar_mass_m50[mask_m50]])
    mags_combined = np.concatenate([mags_m25[:, mask_m25], mags_m50[:, mask_m50]], axis=1)

    # Compute beta
    beta_combined = Calbeta(mags_combined, wavelengths)

    # Take log10 of stellar mass
    log_stellar_mass = np.log10(stellar_mass_combined)

    # Bin beta vs stellar mass
    bin_centers, beta_mean, beta_std, _ = bin_xy(
        x_values=log_stellar_mass,
        y_values=beta_combined,
        mask_values=None,
        mask_cut=None,
        N_bins=10
    )

    # Plot (lines with error shading)
    plt.plot(bin_centers, beta_mean, color=color, linestyle=linestyles[j % len(linestyles)],
             label=law)
    #plt.fill_between(bin_centers, beta_mean - beta_std, beta_mean + beta_std,
    #                 color=color, alpha=0.2)

# add no dust line
# Magnitudes
mags_m25 = np.array([[g.absmag_nodust[band] for g in obj_m25.galaxies] for band in bands])
mags_m50 = np.array([[g.absmag_nodust[band] for g in obj_m50.galaxies] for band in bands])

# Stellar mass
stellar_mass_m25 = np.array([g.masses['stellar'] for g in obj_m25.galaxies])
stellar_mass_m50 = np.array([g.masses['stellar'] for g in obj_m50.galaxies])

# Apply magnitude cuts
mask_m25 = mags_m25[0] < -16
mask_m50 = mags_m50[0] < -17.5

stellar_mass_combined = np.concatenate([stellar_mass_m25[mask_m25], stellar_mass_m50[mask_m50]])
mags_combined = np.concatenate([mags_m25[:, mask_m25], mags_m50[:, mask_m50]], axis=1)

# Compute beta
beta_combined = Calbeta(mags_combined, wavelengths)

# Take log10 of stellar mass
log_stellar_mass = np.log10(stellar_mass_combined)

# Bin beta vs stellar mass
bin_centers, beta_mean, beta_std, _ = bin_xy(
    x_values=log_stellar_mass,
    y_values=beta_combined,
    mask_values=None,
    mask_cut=None,
    N_bins=10
)

# Plot (lines with error shading)
plt.plot(bin_centers, beta_mean, color=color, linestyle='--',
            label='nodust')

# -----------------------------
# Labels and style
# -----------------------------
plt.xlabel(r"log$_{10}$(Stellar Mass / M$_\odot$)")
plt.ylabel(r"$\beta$")
plt.ylim(-2.6, -0.8)
plt.xlim(7, 11)
plt.text(7.2, -1, f"z = {round(obj_m25.simulation.redshift)}", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("Beta_vs_StellarMass_dust_laws.png", dpi=300)
plt.show()

