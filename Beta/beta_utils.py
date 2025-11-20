import numpy as np

def Calbeta(magnitudes, wavelengths):
    loglambda = np.log10(wavelengths)
    logflux = -0.4 * magnitudes
    slope = np.polyfit(loglambda, logflux, 1)[0]
    return slope - 2

def bin_beta(M1500, beta, N_bins=20, mag_cut=-16):
    """
    Bin galaxies by M1500 and compute mean and std of beta in each bin.

    Parameters
    ----------
    M1500 : array-like
        Absolute UV magnitudes of galaxies (1D array).
    beta : array-like
        Beta slope values for the same galaxies (1D array).
    N_bins : int, optional
        Number of bins. Default is 20.
    mag_cut : float, optional
        Only include galaxies with M1500 < mag_cut. Default is -16.

    Returns
    -------
    bin_centers : ndarray
        Centers of the magnitude bins.
    beta_mean : ndarray
        Mean beta in each bin.
    beta_std : ndarray
        Standard deviation of beta in each bin.
    """
    # Mask galaxies
    mask = M1500 < mag_cut
    M1500_sel = M1500[mask]
    beta_sel  = beta[mask]

    # Define bins
    bins = np.linspace(M1500_sel.min(), M1500_sel.max(), N_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Assign galaxies to bins
    bin_index = np.digitize(M1500_sel, bins)

    # Compute mean and std per bin
    beta_mean = []
    beta_std  = []
    for i in range(1, N_bins + 1):
        in_bin = beta_sel[bin_index == i]
        if len(in_bin) > 0:
            beta_mean.append(in_bin.mean())
            beta_std.append(in_bin.std())
        else:
            beta_mean.append(np.nan)
            beta_std.append(np.nan)

    return bin_centers, np.array(beta_mean), np.array(beta_std)

def get_binned_beta(obj, bands, wavelengths, N_bins=10, mag_cut=-16, nodust=False):
    """Compute binned beta for a Caesar object."""
    if nodust:
        mags = np.array([[g.absmag_nodust[band] for g in obj.galaxies] for band in bands])
    else:
        mags = np.array([[g.absmag[band] for g in obj.galaxies] for band in bands])
    
    beta = Calbeta(mags, wavelengths)
    M1500 = mags[0]
    
    return bin_beta(M1500, beta, N_bins=N_bins, mag_cut=mag_cut)

 
