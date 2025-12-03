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

# File lists for each box (sorted by redshift)
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

boxes = {'25': files_25, '50': files_50}
colors = {'25':'blue', '50':'red'}

# Number of subplots = number of redshifts
nred = len(files_25)
fig, axes = plt.subplots(3, 2, figsize=(12, 15), sharex=True, sharey=True)
axes = axes.flatten()  # flatten to 1D array for easy indexing

for i in range(nred):
    ax = axes[i]
    for box, filelist in boxes.items():
        obj = caesar.load(filelist[i])

        boxsize = obj.simulation.boxsize.to("Mpccm").d
        magnitude = np.array([g.absmag["i1500"] for g in obj.galaxies])

        bin_centers, logphi, logphi_err = cal_UVLF(magnitude, boxsize, nbins=30)

        ax.errorbar(bin_centers, logphi, yerr=logphi_err, capsize=3, capthick=1.2,
                    color=colors[box], label=f"Box {box}" if i==0 else "", fmt='-')

    z = int(round(obj.simulation.redshift))  # convert to int
    ax.text(0.05, 0.9, f"z = {z}", transform=ax.transAxes, 
         fontsize=12, verticalalignment='top', color='black')
    ax.grid(alpha=1)

# x-axis labels for bottom row
for ax in axes[4:]:
    ax.set_xlabel(r"$M_{1500}$", fontsize=12)

# y-axis labels for first column
for ax in axes[::2]:
    ax.set_ylabel(r"$\log_{10}(\phi)$", fontsize=12)

# legend inside first subplot
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, labels, fontsize=12)

# minimal spacing
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1,
                    wspace=0.05, hspace=0.05)

plt.savefig("UVLF_subplots.png")

