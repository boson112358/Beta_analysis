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

# List of simulation files
infiles = [
    '/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/CaesarFile/m25n1024/caesar_m25n1024_036.hdf5',
    '/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/CaesarFile/m50n1024/caesar_m50n1024_036.hdf5'
]

colors = ["blue", "red"]  # one color per simulation

plt.figure()

for i, infile in enumerate(infiles):
    obj = caesar.load(infile)

    # Boxsize in Mpccm
    boxsize = obj.simulation.boxsize.to("Mpccm").d

    # Get UV magnitudes
    magnitude = np.array([g.absmag["i1500"] for g in obj.galaxies])

    # Compute UVLF
    bin_centers, logphi, logphi_err = cal_UVLF(magnitude, boxsize, nbins=30)

    # Plot
    plt.errorbar(bin_centers, logphi, yerr=logphi_err, capsize=4,
                 capthick=1.2, color=colors[i], label=f"Sim {i+1}")


plt.xlabel(r"$M_{1500}$")
plt.ylabel(r"$\log_{10}(\phi)$")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("UVLF_all.png")
