import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import caesar

from pathlib import Path

from utils.beta_utils import Calbeta


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "axes.grid": False,
    "savefig.dpi": 300,
})


# ============================================================
# Files and parameters
# ============================================================

redshifts = ['016', '019', '022', '026', '030', '036']

dust_law = "calzetti"

bands = ["i1500", "i2300", "i2800"]
wavelengths = np.array([1500, 2300, 2800])

observation_file = Path(__file__).with_name(
    "beta_redshift_binned.csv"
)
redshift_observations = pd.read_csv(observation_file)

observation_markers = {
    "Bouwens 2014": "s",
    "Cullen 2024": "^",
    "Morales 2024": "D",
    "Napolitano 2026": "v",
    "Whitler 2025": "p",
    "Topping 2024": "X",
}

distribution_colours = [
    "#0072B2", "#E69F00", "#009E73",
    "#D55E00", "#CC79A7", "#56B4E9",
]

template_m25 = (
    "/cosma8/data/dp376/dc-xian3/simba-eor/"
    "EoRData/FullSpectra_Fit/m25n1024/"
    "caesar_m25n1024_{}_{}.hdf5"
)

template_m50 = (
    "/cosma8/data/dp376/dc-xian3/simba-eor/"
    "EoRData/FullSpectra_Fit/m50n1024/"
    "caesar_m50n1024_{}_{}.hdf5"
)


# ============================================================
# Storage
# ============================================================

zvals = []

median_beta = []
beta_lower = []
beta_upper = []

# Store beta distribution at each redshift
beta_distributions = []

# ============================================================
# Loop over snapshots
# ============================================================

for z_str in redshifts:

    print(f"Processing z = {z_str}")

    # --------------------------------------------------------
    # Load catalogues
    # --------------------------------------------------------

    obj_m25 = caesar.load(
        template_m25.format(z_str, dust_law)
    )

    obj_m50 = caesar.load(
        template_m50.format(z_str, dust_law)
    )

    z_sim = obj_m25.simulation.redshift


    # --------------------------------------------------------
    # Magnitudes
    # --------------------------------------------------------

    mags_m25 = np.array([
        [g.absmag[band] for g in obj_m25.galaxies]
        for band in bands
    ])

    mags_m50 = np.array([
        [g.absmag[band] for g in obj_m50.galaxies]
        for band in bands
    ])


    # --------------------------------------------------------
    # Magnitude cuts
    # --------------------------------------------------------

    mask_m25 = mags_m25[0] < -16
    mask_m50 = mags_m50[0] < -17.5


    # --------------------------------------------------------
    # Combine M25 + M50
    # --------------------------------------------------------

    mags = np.concatenate(
        [
            mags_m25[:, mask_m25],
            mags_m50[:, mask_m50]
        ],
        axis=1
    )


    if mags.shape[1] == 0:
        continue


    # --------------------------------------------------------
    # Calculate beta
    # --------------------------------------------------------

    beta = Calbeta(
        mags,
        wavelengths
    )


    # --------------------------------------------------------
    # Median + 16-84 percentile
    # --------------------------------------------------------

    median = np.median(beta)

    p16 = np.percentile(beta, 16)
    p84 = np.percentile(beta, 84)

    median_beta.append(median)

    beta_lower.append(median - p16)
    beta_upper.append(p84 - median)

    zvals.append(z_sim)


    # --------------------------------------------------------
    # Store beta distribution
    # --------------------------------------------------------

    beta_distributions.append(beta)


    print(
        f"  z = {z_sim:.2f}, "
        f"N = {len(beta)}, "
        f"median beta = {median:.3f}, "
        f"16-84 = [{p16:.3f}, {p84:.3f}]"
    )


# ============================================================
# Convert to arrays
# ============================================================

zvals = np.array(zvals)

median_beta = np.array(median_beta)
beta_lower = np.array(beta_lower)
beta_upper = np.array(beta_upper)


# ============================================================
# Linear fit to the simulated median-beta evolution
# ============================================================

# Pivoting at z = 6 makes the intercept physically meaningful and reduces
# its covariance with the slope.  The fit is unweighted because the 16th--
# 84th percentile range measures galaxy-to-galaxy scatter, not the error on
# the median.  This fit is printed only; it is not added to the figure.
z_pivot = 6.0
x_fit = zvals - z_pivot

(slope, beta_at_z6), covariance = np.polyfit(
    x_fit,
    median_beta,
    deg=1,
    cov=True,
)

slope_error = np.sqrt(covariance[0, 0])
beta_at_z6_error = np.sqrt(covariance[1, 1])

beta_fitted = slope * x_fit + beta_at_z6
ss_res = np.sum((median_beta - beta_fitted) ** 2)
ss_tot = np.sum((median_beta - np.mean(median_beta)) ** 2)
r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

print("\nLinear fit to the simulated median beta evolution:")
print("  beta(z) = beta_6 + slope * (z - 6)")
print(f"  slope  = {slope:.4f} +/- {slope_error:.4f} per unit redshift")
print(f"  beta_6 = {beta_at_z6:.4f} +/- {beta_at_z6_error:.4f}")
print(f"  R^2    = {r_squared:.4f}")


# ============================================================
# Create figure
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(7.1, 3.15),
    constrained_layout=True,
)


# ============================================================
# LEFT PANEL
# Median beta evolution for the full sample
# ============================================================

ax1.errorbar(
    zvals,
    median_beta,
    yerr=[beta_lower, beta_upper],
    fmt="o-",
    color="black",
    capsize=2.5,
    elinewidth=1.0,
    linewidth=1.8,
    markersize=5.0,
    markerfacecolor="black",
    markeredgecolor="white",
    markeredgewidth=0.6,
    zorder=10,
    label="All galaxies",
)

# Observational measurements binned by redshift.
for dataset, data in redshift_observations.groupby("dataset", sort=False):
    ax1.errorbar(
        data["redshift"],
        data["beta"],
        yerr=data["uncertainty"],
        fmt=observation_markers.get(dataset, "o"),
        linestyle="none",
        color="0.40",
        ecolor="0.72",
        markerfacecolor="white",
        markeredgecolor="0.40",
        markeredgewidth=0.9,
        markersize=4.5,
        capsize=1.5,
        elinewidth=0.7,
        alpha=0.95,
        zorder=5,
        label=dataset,
    )

ax1.set_xlabel("Redshift")
ax1.set_ylabel(r"UV slope, $\beta$")
ax1.text(
    0.03, 0.96, r"(a) Redshift evolution",
    transform=ax1.transAxes, ha="left", va="top"
)

ax1.legend(
    loc="lower left", ncol=2, columnspacing=0.9,
    handletextpad=0.4, borderaxespad=0.4,
)

# Optional: reverse x-axis so cosmic time goes left -> right
# (high z on left, low z on right)
# Do not invert
#ax1.invert_xaxis()


# ============================================================
# RIGHT PANEL
# Beta distributions
# ============================================================

# Use common bins for all redshifts
all_beta = np.concatenate(beta_distributions)

bins = np.linspace(
    np.percentile(all_beta, 0.5),
    np.percentile(all_beta, 99.5),
    30
)


for colour, z, beta in zip(
    distribution_colours, zvals, beta_distributions
):

    ax2.hist(
        beta,
        bins=bins,
        histtype='step',
        color=colour,
        linewidth=1.15,
        density=False,
        label=f"z = {z:.1f}"
    )


ax2.set_xlabel(r"UV slope, $\beta$")
ax2.set_ylabel("Number of galaxies")
ax2.text(
    0.03, 0.96, r"(b) Distributions",
    transform=ax2.transAxes, ha="left", va="top"
)

ax2.legend(
    loc="upper right", ncol=2, columnspacing=0.8,
    handlelength=1.7, borderaxespad=0.4,
)


# ============================================================
# Final formatting
# ============================================================

plt.savefig(
    "Beta_evolution_and_distribution_calzetti.png",
    dpi=300,
    bbox_inches="tight"
)
plt.savefig(
    "Beta_evolution_and_distribution_calzetti.pdf",
    bbox_inches="tight"
)

plt.show()
