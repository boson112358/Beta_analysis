import numpy as np
import caesar
import unyt
import matplotlib.pyplot as plt
from beta_utils import Calbeta, bin_beta, get_binned_beta

# Get the input caesar file
infile = '/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/CaesarFile/m25n1024/caesar_m25n1024_036.hdf5'
infile_m50 = '/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/CaesarFile/m50n1024/caesar_m50n1024_036.hdf5'

# Load caesar file
obj = caesar.load(infile)
obj_m50 = caesar.load(infile_m50)

# Redshift
Z = obj.simulation.redshift

# Get UV magnitude
bands = ["i1500", "i2300", "i2800"]
wavelengths = np.array([1500, 2300, 2800])

# --- Load magnitudes for dust ---
mags_m25 = np.array([[g.absmag[band] for g in obj.galaxies] for band in bands])
mags_m50 = np.array([[g.absmag[band] for g in obj_m50.galaxies] for band in bands])

# --- Combine the two boxes ---
mags_combined = np.concatenate([mags_m25, mags_m50], axis=1)

# --- Compute beta ---
beta_combined = Calbeta(mags_combined, wavelengths)
M1500_combined = mags_combined[0]

# --- Bin beta ---
bin_centers, beta_mean, beta_std = bin_beta(M1500_combined, beta_combined, N_bins=10, mag_cut=-16)

# --- Plot ---
plt.figure(figsize=(6,4))
plt.errorbar(bin_centers, beta_mean, yerr=beta_std,
             color='green', linestyle='-', marker='o', label='m25 + m50 dust')

plt.xlabel("M1500")
plt.ylabel("Beta")
plt.ylim(-2.6, -0.8)
plt.xlim(-23, -15)
plt.text(-22, -1, f"z = {round(Z)}", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("Beta_combined_dust.png")
