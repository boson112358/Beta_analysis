import matplotlib.pyplot as plt
import numpy as np
import caesar
from utils.beta_utils import *   # Calbeta, etc.

# -----------------------------------------------------
# File lists for m25 and m50 (sorted by redshift)
# -----------------------------------------------------
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

# -----------------------------------------------------
# Setup: UV magnitudes
# -----------------------------------------------------
bands = ["i1500", "i2300", "i2800"]
wavelengths = np.array([1500, 2300, 2800])

# -----------------------------------------------------
# Storage
# -----------------------------------------------------
redshift_list = []
beta_mean_list = []
beta_std_list = []
Ngal_list      = [] 
beta_mean_lum_list = []
beta_std_lum_list = []
beta_mean_mass_list = []
beta_std_mass_list = []

# -----------------------------------------------------
# Loop over redshifts
# -----------------------------------------------------
for f25, f50 in zip(files_25, files_50):

    # Load Caesar snapshots
    obj_25 = caesar.load(f25)
    obj_50 = caesar.load(f50)

    # Redshift
    z = obj_25.simulation.redshift
    redshift_list.append(z)

    # Magnitudes for the three UV bands
    mags_25 = np.array([[g.absmag[band] for g in obj_25.galaxies] for band in bands])
    mags_50 = np.array([[g.absmag[band] for g in obj_50.galaxies] for band in bands])
    
    # --- Extract stellar mass arrays ---
    stellar_mass_m25 = np.array([g.masses['stellar'] for g in obj_25.galaxies])
    stellar_mass_m50 = np.array([g.masses['stellar'] for g in obj_50.galaxies])
    
    # Apply magnitude cuts
    mask_25 = mags_25[0] < -16
    mask_50 = mags_50[0] < -17.5
    
    stellar_mass_m25_cut = stellar_mass_m25[mask_25]
    stellar_mass_m50_cut = stellar_mass_m50[mask_50]

    stellar_mass_combined = np.concatenate([stellar_mass_m25_cut, stellar_mass_m50_cut])

    mags_combined = np.concatenate([mags_25[:, mask_25],
                                    mags_50[:, mask_50]], axis=1)

    # Compute β slope
    beta_combined = Calbeta(mags_combined, wavelengths)

    # Mean and standard deviation
    beta_mean_list.append(np.mean(beta_combined))
    beta_std_list.append(np.std(beta_combined))

    # Count number of galaxies after cuts
    Ngal_list.append(mags_combined.shape[1])

    # Luminosity-weighted
    luminosity = 10**(-0.4 * mags_combined[0])
    beta_mean_lum = np.average(beta_combined, weights=luminosity)
    beta_std_lum  = np.sqrt(np.average((beta_combined - beta_mean_lum)**2, weights=luminosity))
    beta_mean_lum_list.append(beta_mean_lum)
    beta_std_lum_list.append(beta_std_lum)

    # Stellar-mass-weighted
    beta_mean_mass = np.average(beta_combined, weights=stellar_mass_combined)
    beta_std_mass  = np.sqrt(np.average((beta_combined - beta_mean_mass)**2, weights=stellar_mass_combined))
    
    beta_mean_mass_list.append(beta_mean_mass)
    beta_std_mass_list.append(beta_std_mass)
# -----------------------------------------------------
# Plot β(z)
# -----------------------------------------------------
plt.figure(figsize=(7,5))

plt.errorbar(
    redshift_list, beta_mean_list, yerr=beta_std_list,
    color='green', marker='o', linestyle='-', capsize=4,
    label=r'Mean $\beta$'
)

# Add number of galaxies as text above each point
#for x, y, n in zip(redshift_list, beta_mean_list, Ngal_list):
#    plt.text(x, y + 0.05, str(n), fontsize=9, ha='center', va='bottom')

plt.errorbar(redshift_list, beta_mean_lum_list, yerr=beta_std_lum_list,
             color='blue', marker='o', linestyle='-', label='Lum-weighted')

plt.errorbar(redshift_list, beta_mean_mass_list, yerr=beta_std_mass_list,
             color='red', marker='s', linestyle='--', label='Mass-weighted')


plt.xlabel("Redshift", fontsize=12)
plt.ylabel(r"Mean $\beta$", fontsize=12)
plt.ylim(-2.3,-1.8)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("Beta_vs_redshift.png", dpi=300)
plt.show()

