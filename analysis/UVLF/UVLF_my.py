import numpy as np
import caesar
import unyt
import matplotlib.pyplot as plt

def cal_UVLF(magnitude, boxsize, nbins=30, mag_range=[-25, -14]):
    """
    Calculate the UV luminosity function.

    Parameters
    ----------
    magnitude : array_like
        Array of galaxy magnitudes.
    boxsize : float
        Box size in Mpc (comoving).
    nbins : int, optional
        Number of magnitude bins (default 10).
    mag_range : list of float, optional
        [min_mag, max_mag] range for the histogram (default [-22, -14]).

    Returns
    -------
    bin_centers : ndarray
        Centers of magnitude bins (only bins with counts > 0).
    logphi : ndarray
        Log10 of the UVLF in Mpc^-3 mag^-1.
    logphi_err : ndarray
        Poisson error on log10(phi).
    """
    # Histogram
    num, bin_edges = np.histogram(magnitude, bins=nbins, range=mag_range)

    # Bin centers
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Remove empty bins
    mask = num > 0
    num = num[mask]
    bin_centers = bin_centers[mask]

    # Bin width (assume uniform)
    dm = (mag_range[1] - mag_range[0]) / nbins

    # UVLF and Poisson error
    phi = num / (dm * boxsize**3)
    phi_err = np.sqrt(num) / (dm * boxsize**3)

    # Log values and propagated error
    logphi = np.log10(phi)
    logphi_err = phi_err / (phi * np.log(10))

    return bin_centers, logphi, logphi_err

# Get the input caesar file
infile = '/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/CaesarFile/m25n1024/caesar_m25n1024_036.hdf5'
infile_m50 = '/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/CaesarFile/m50n1024/caesar_m50n1024_036.hdf5'

# Load caesar file
obj = caesar.load(infile)
obj_m50 = caesar.load(infile_m50)

# Boxsize in kpccm
Boxsize = obj.simulation.boxsize
Boxsize_m50 = obj_m50.simulation.boxsize 

# Number of galaxies
Ngal = obj.ngalaxies

# Redshift
Z = obj.simulation.redshift

# Print the info
print("Boxsize: ", Boxsize)
print("Number of galaxies: ", Ngal)
print("Redshift: ", Z)

print("magnitude:", obj.galaxies[0].absmag["i1500"])

# Get UV magnitude
magnitude = np.array([g.absmag["i1500"] for g in obj.galaxies])
magnitude_m50 = np.array([g.absmag["i1500"] for g in obj_m50.galaxies])

# Boxsize in Mpccm
Boxsize_Mpc = Boxsize.to("Mpccm").d
Boxsize_Mpc_m50 = Boxsize_m50.to("Mpccm").d

# Get UVLF and error
bin_centers, logphi, logphi_err = cal_UVLF(magnitude, Boxsize_Mpc, nbins=30)
bin_centers_m50, logphi_m50, logphi_err_m50 = cal_UVLF(magnitude_m50, Boxsize_Mpc_m50, nbins=30)

plt.figure()
plt.errorbar(bin_centers, logphi, yerr=logphi_err, capsize=4, capthick=1.2, color="blue")
plt.errorbar(bin_centers_m50, logphi_m50, yerr=logphi_err_m50, capsize=4, capthick=1.2, color="red")
plt.savefig("UVLF.png")
    
