import numpy as np
import statsmodels.api as sm

# Helper to extract stellar mass array from a Caesar object
def get_stellar_mass(obj):
    return np.array([g.masses['stellar'] for g in obj.galaxies])

def Calbeta(magnitudes, wavelengths):
    loglambda = np.log10(wavelengths)
    logflux = -0.4 * magnitudes
    slope = np.polyfit(loglambda, logflux, 1)[0]
    return slope - 2

def bin_beta(M1500, beta, N_bins=20, mag_cut=-16, min_count=10):
    """
    Bin galaxies by M1500 and compute mean and std of beta in each bin.
    Bins with fewer than min_count galaxies are discarded.

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
    min_count : int, optional
        Minimum number of galaxies in a bin to keep it. Default is 10.

    Returns
    -------
    bin_centers : ndarray
        Centers of the magnitude bins.
    beta_mean : ndarray
        Mean beta in each bin.
    beta_std : ndarray
        Standard deviation of beta in each bin.
    bin_count : ndarray
        Number of galaxies in each bin.
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
    bin_count = []
    for i in range(1, N_bins + 1):
        in_bin = beta_sel[bin_index == i]
        bin_count.append(len(in_bin))
        if len(in_bin) > 0:
            beta_mean.append(in_bin.mean())
            beta_std.append(in_bin.std())
        else:
            beta_mean.append(np.nan)
            beta_std.append(np.nan)

    # Convert to arrays
    bin_centers = np.array(bin_centers)
    beta_mean = np.array(beta_mean)
    beta_std = np.array(beta_std)
    bin_count = np.array(bin_count)

    # Filter bins with fewer than min_count galaxies
    mask_valid = bin_count >= min_count
    bin_centers = bin_centers[mask_valid]
    beta_mean = beta_mean[mask_valid]
    beta_std = beta_std[mask_valid]
    bin_count = bin_count[mask_valid]

    return bin_centers, beta_mean, beta_std, bin_count

def get_binned_beta(obj, bands, wavelengths, N_bins=10, mag_cut=-16, nodust=False):
    """Compute binned beta for a Caesar object."""
    if nodust:
        mags = np.array([[g.absmag_nodust[band] for g in obj.galaxies] for band in bands])
    else:
        mags = np.array([[g.absmag[band] for g in obj.galaxies] for band in bands])
    
    beta = Calbeta(mags, wavelengths)
    M1500 = mags[0]
    
    return bin_beta(M1500, beta, N_bins=N_bins, mag_cut=mag_cut)

def get_binned_xy(obj, bands, wavelengths, 
                    x_values=None, N_bins=10, mag_cut=-16, nodust=False,
                    log_x=False):
    """Compute binned beta for a Caesar object."""
    
    # --- Extract magnitudes ---
    if nodust:
        mags = np.array([[g.absmag_nodust[band] for g in obj.galaxies] for band in bands])
    else:
        mags = np.array([[g.absmag[band] for g in obj.galaxies] for band in bands])
    
    # Compute β
    beta = Calbeta(mags, wavelengths)

    # M1500 always used for masking
    M1500 = mags[0]

    # Default x-axis is still M1500 (old behavior)
    if x_values is None:
        x_values = M1500

    # Take log if requested
    if log_x:
        x_values = np.log10(x_values)

    # Use the generalized binning function with mask
    return bin_xy(
        x_values=x_values,
        y_values=beta,
        mask_values=M1500,
        mask_cut=mag_cut,
        N_bins=N_bins
    )


def bin_xy(x_values, y_values, mask_values=None, mask_cut=None,
           N_bins=20, min_count=10):
    """
    Bin galaxies by x_values and compute mean/std of y_values.
    Optional mask based on mask_values < mask_cut.
    """
    x_values = np.array(x_values)
    y_values = np.array(y_values)

    # ---- Apply mask if provided ----
    if mask_values is not None and mask_cut is not None:
        mask_values = np.array(mask_values)
        keep = mask_values < mask_cut
        x_values = x_values[keep]
        y_values = y_values[keep]

    # ---- Define bins on x ----
    bins = np.linspace(x_values.min(), x_values.max(), N_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_index = np.digitize(x_values, bins)

    y_mean, y_std, bin_count = [], [], []

    for i in range(1, N_bins + 1):
        in_bin = y_values[bin_index == i]
        bin_count.append(len(in_bin))
        if len(in_bin) > 0:
            y_mean.append(in_bin.mean())
            y_std.append(in_bin.std())
        else:
            y_mean.append(np.nan)
            y_std.append(np.nan)

    # Convert to arrays
    bin_centers = np.array(bin_centers)
    y_mean = np.array(y_mean)
    y_std = np.array(y_std)
    bin_count = np.array(bin_count)

    # ---- Min-count filtering ----
    valid = bin_count >= min_count
    return (bin_centers[valid],
            y_mean[valid],
            y_std[valid],
            bin_count[valid])


def linear_regression_fit(bin_centers, beta_mean, beta_std=None):
    """
    Perform weighted linear regression on binned data and return slope/intercept and fitted line.
    
    Parameters:
        bin_centers : array-like
            Independent variable (x)
        beta_mean : array-like
            Dependent variable (y)
        beta_std : array-like, optional
            Standard deviations for weights (weighted regression)
    
    Returns:
        slope, intercept : float
        x_fit, y_fit : arrays for plotting fitted line
    """
    # --- Weights ---
    if beta_std is not None:
        mask = (~np.isnan(beta_std)) & (beta_std > 0)
        bin_centers = bin_centers[mask]
        beta_mean = beta_mean[mask]
        beta_std = beta_std[mask]
        w = 1 / beta_std**2
    else:
        w = None
    
    # --- Fit model ---
    X = sm.add_constant(bin_centers)
    model = sm.WLS(beta_mean, X, weights=w).fit()
    
    print(model.summary())

    # --- Extract parameters ---
    intercept = model.params[0]
    slope = model.params[1]
    
    # --- Prepare smooth line for plotting ---
    x_min, x_max = bin_centers.min(), bin_centers.max()
    x_fit = np.linspace(x_min, x_max, 200)
    X_fit = sm.add_constant(x_fit)
    y_fit = model.predict(X_fit)
    
    return slope, intercept, x_fit, y_fit 
