import numpy as np
import caesar
import unyt
import matplotlib.pyplot as plt
from utils.beta_utils import *

# Get the input caesar file
infile = '/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/CaesarFile/m25n1024/caesar_m25n1024_036.hdf5'
infile_m50 = '/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/CaesarFile/m50n1024/caesar_m50n1024_036.hdf5'

# Load caesar file
obj = caesar.load(infile)
obj_m50 = caesar.load(infile_m50)

# Number of galaxies
Ngal = obj.ngalaxies

# Redshift
Z = obj.simulation.redshift

# Get UV magnitude
bands = ["i1500", "i2300", "i2800"]
wavelengths = np.array([1500, 2300, 2800])

# --- Extract stellar mass arrays ---
stellar_mass_m25 = np.array([g.masses['stellar'] for g in obj.galaxies])
stellar_mass_m50 = np.array([g.masses['stellar'] for g in obj_m50.galaxies])

# --- Apply magnitude cuts ---
mask_m25 = np.array([g.absmag["i1500"] for g in obj.galaxies]) < -16
mask_m50 = np.array([g.absmag["i1500"] for g in obj_m50.galaxies]) < -17.5

stellar_mass_m25_cut = stellar_mass_m25[mask_m25]
stellar_mass_m50_cut = stellar_mass_m50[mask_m50]

# --- Combine the two boxes ---
stellar_mass_combined = np.concatenate([stellar_mass_m25_cut, stellar_mass_m50_cut])
# Combine magnitudes for beta calculation (same as before)
mags_combined = np.concatenate([
    np.array([[g.absmag[band] for g in obj.galaxies] for band in bands])[:, mask_m25],
    np.array([[g.absmag[band] for g in obj_m50.galaxies] for band in bands])[:, mask_m50]
], axis=1)

# --- Compute beta ---
beta_combined = Calbeta(mags_combined, wavelengths)

# --- Take log10 of stellar mass for binning ---
log_stellar_mass = np.log10(stellar_mass_combined)

# --- Bin beta ---
bin_centers, beta_mean, beta_std, bin_count = bin_xy(
    x_values=log_stellar_mass,   # use log stellar mass
    y_values=beta_combined,
    mask_values=None,            # masking already applied
    mask_cut=None,
    N_bins=10
)

# --- Plot ---
plt.figure(figsize=(6,4))
plt.errorbar(bin_centers, beta_mean, yerr=beta_std,
             color='green', linestyle='-', marker='o', label='m25 + m50 dust')

plt.xlabel(r"log$_{10}$(Stellar Mass / M$_\odot$)")
plt.ylabel(r"$\beta$")
plt.ylim(-2.6, -0.8)
plt.xlim(7, 11)  # log10(M*) range
plt.text(7.2, -1, f"z = {round(Z)}", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("Beta_vs_StellarMass_combined_dust.png")
   
