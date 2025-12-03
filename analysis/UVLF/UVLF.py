import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Function to compute the UV Luminosity Function (UVLF)
# ------------------------------------------------------------
def calc_UVLF(UVmag, boxsize, bin_width=0.5, mag_range=[-22, -15]):
    """
    UVmag: array of UV absolute magnitudes (M1500)
    boxsize: simulation box size in comoving Mpc/h
    bin_width: width of magnitude bins
    mag_range: [min_mag, max_mag] range for binning
    """
    # Define consistent bins
    bins = np.arange(mag_range[0] - 0.5, mag_range[1] + 0.5 + bin_width, bin_width)
    arr, bin_edges = np.histogram(UVmag, bins=bins)
    
    delmag = bin_edges[1:] - bin_edges[:-1]
    mag0 = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Avoid zero counts to prevent log10 errors
    arr = np.where(arr == 0, np.nan, arr)

    # phi in Mpc^-3 mag^-1, multiply by (h/boxsize)^3 if using comoving units
    phi = arr / delmag / (boxsize ** 3)
    logphi = np.log10(phi)
    err_phi = 1.0 / np.sqrt(arr) / np.log(10)

    return mag0, logphi, err_phi


# ------------------------------------------------------------
# Example: mock UV magnitude data for two simulations
# (In reality, load your M1500 arrays here)
# ------------------------------------------------------------
np.random.seed(42)
MUV_25 = np.random.normal(-18, 1.5, 3000)   # SIMBA-25 mock data
MUV_50 = np.random.normal(-19, 1.2, 8000)   # SIMBA-50 mock data

boxsize_25 = 25.0  # comoving Mpc/h
boxsize_50 = 50.0  # comoving Mpc/h

# ------------------------------------------------------------
# Compute UVLFs
# ------------------------------------------------------------
mag25, logphi25, err25 = calc_UVLF(MUV_25, boxsize_25)
mag50, logphi50, err50 = calc_UVLF(MUV_50, boxsize_50)

# Ensure bins match for combining
assert np.allclose(mag25, mag50), "Bins are not aligned — check mag_range and bin_width"

# ------------------------------------------------------------
# Combine UVLFs by taking the higher value per bin
# ------------------------------------------------------------
logphi_comb = np.maximum(logphi25, logphi50)
err_comb = np.where(logphi25 > logphi50, err25, err50)

# ------------------------------------------------------------
# Plot the results
# ------------------------------------------------------------
plt.figure(figsize=(7,6))

plt.errorbar(mag25, logphi25, yerr=err25, fmt='o--', label='SIMBA-25', alpha=0.7)
plt.errorbar(mag50, logphi50, yerr=err50, fmt='s--', label='SIMBA-50', alpha=0.7)
plt.errorbar(mag25, logphi_comb, yerr=err_comb, fmt='^-', color='k', label='Combined', lw=2)

plt.xlabel(r'$M_{1500}$ (mag)', fontsize=14)
plt.ylabel(r'$\log_{10}(\phi)\ [\mathrm{Mpc^{-3}\ mag^{-1}}]$', fontsize=14)
plt.title('UV Luminosity Function', fontsize=15)

plt.gca().invert_xaxis()  # Brighter galaxies on the left
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

