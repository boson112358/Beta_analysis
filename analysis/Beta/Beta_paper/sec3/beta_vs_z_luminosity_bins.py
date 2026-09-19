import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import caesar

from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
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

redshifts = ["016", "019", "022", "026", "030", "036"]
dust_law = "calzetti"

bands = ["i1500", "i2300", "i2800"]
wavelengths = np.array([1500, 2300, 2800])

observation_file = Path(__file__).with_name(
    "beta_redshift_luminosity_binned.csv"
)
luminosity_observations = pd.read_csv(observation_file)

observation_markers = {
    "Bouwens 2014": "s",
    "Cullen 2024": "^",
    "Morales 2024": "D",
    "Napolitano 2026": "v",
    "Whitler 2025": "p",
    "Topping 2024": "X",
}

# Both simulation volumes are retained as independent galaxy samples.
# Bright galaxies from M25 and M50 therefore both contribute.
luminosity_bins = [
    (-17.5, -16.0, r"$-17.5 < M_{1500} \leq -16$"),
    (-19.0, -17.5, r"$-19 < M_{1500} \leq -17.5$"),
    (-np.inf, -19.0, r"$M_{1500} \leq -19$"),
]

# Match each simulation-bin label to the corrected observational group.
observation_groups = {
    r"$-17.5 < M_{1500} \leq -16$": "Faint (-17.5 < M_UV <= -16)",
    r"$-19 < M_{1500} \leq -17.5$": "Medium (-19 < M_UV <= -17.5)",
    r"$M_{1500} \leq -19$": "Bright (M_UV <= -19)",
}

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
binned_results = {
    label: {"median": [], "p16": [], "p84": [], "count": []}
    for _, _, label in luminosity_bins
}


# ============================================================
# Loop over snapshots
# ============================================================

for z_str in redshifts:
    print(f"Processing snapshot {z_str}")

    obj_m25 = caesar.load(template_m25.format(z_str, dust_law))
    obj_m50 = caesar.load(template_m50.format(z_str, dust_law))
    z_sim = obj_m25.simulation.redshift

    mags_m25 = np.array([
        [g.absmag[band] for g in obj_m25.galaxies]
        for band in bands
    ])
    mags_m50 = np.array([
        [g.absmag[band] for g in obj_m50.galaxies]
        for band in bands
    ])

    # Retain the original sample selections.
    mask_m25 = mags_m25[0] < -16
    mask_m50 = mags_m50[0] < -17.5

    mags = np.concatenate(
        [mags_m25[:, mask_m25], mags_m50[:, mask_m50]],
        axis=1,
    )

    if mags.shape[1] == 0:
        continue

    zvals.append(z_sim)

    for bright_edge, faint_edge, label in luminosity_bins:
        muv = mags[0]
        bin_mask = (muv > bright_edge) & (muv <= faint_edge)

        if np.any(bin_mask):
            beta_bin = np.asarray(Calbeta(mags[:, bin_mask], wavelengths))
            beta_bin = beta_bin[np.isfinite(beta_bin)]
        else:
            beta_bin = np.array([])

        n_gal = len(beta_bin)
        binned_results[label]["count"].append(n_gal)

        if n_gal > 0:
            binned_results[label]["median"].append(np.median(beta_bin))
            binned_results[label]["p16"].append(np.percentile(beta_bin, 16))
            binned_results[label]["p84"].append(np.percentile(beta_bin, 84))
        else:
            binned_results[label]["median"].append(np.nan)
            binned_results[label]["p16"].append(np.nan)
            binned_results[label]["p84"].append(np.nan)


# ============================================================
# Plot luminosity-binned beta evolution
# ============================================================

zvals = np.asarray(zvals)
fig, ax = plt.subplots(
    figsize=(7.1, 4.0),
    constrained_layout=True,
)

colours = ["#0072B2", "#009E73", "#D55E00"]

for colour, (_, _, label) in zip(colours, luminosity_bins):
    median = np.asarray(binned_results[label]["median"])
    p16 = np.asarray(binned_results[label]["p16"])
    p84 = np.asarray(binned_results[label]["p84"])
    counts = np.asarray(binned_results[label]["count"])
    valid = np.isfinite(median)

    ax.errorbar(
        zvals[valid],
        median[valid],
        yerr=[median[valid] - p16[valid], p84[valid] - median[valid]],
        fmt="o-",
        color=colour,
        alpha=1.0,
        capsize=2.5,
        elinewidth=0.9,
        linewidth=1.6,
        markersize=5.0,
        markeredgecolor="white",
        markeredgewidth=0.5,
        zorder=10,
        label=label,
    )

    print(f"\n{label}")
    for z, n, med in zip(zvals, counts, median):
        print(f"  z = {z:.2f}: N = {n:5d}, median beta = {med:.3f}")

    # --------------------------------------------------------
    # Linear fit to the median-beta evolution in this bin
    # --------------------------------------------------------

    # Pivot at z = 6 so that the intercept is the fitted beta at z = 6.
    # The fit is unweighted because the 16th--84th percentile interval
    # measures galaxy-to-galaxy scatter rather than uncertainty on the median.
    if np.count_nonzero(valid) >= 3:
        z_pivot = 6.0
        x_fit = zvals[valid] - z_pivot

        (slope, beta_at_z6), covariance = np.polyfit(
            x_fit,
            median[valid],
            deg=1,
            cov=True,
        )

        slope_error = np.sqrt(covariance[0, 0])
        beta_at_z6_error = np.sqrt(covariance[1, 1])

        beta_fitted = slope * x_fit + beta_at_z6
        ss_res = np.sum((median[valid] - beta_fitted) ** 2)
        ss_tot = np.sum(
            (median[valid] - np.mean(median[valid])) ** 2
        )
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        print("  Linear fit: beta(z) = beta_6 + slope * (z - 6)")
        print(
            f"    slope  = {slope:.4f} +/- {slope_error:.4f} "
            "per unit redshift"
        )
        print(
            f"    beta_6 = {beta_at_z6:.4f} +/- "
            f"{beta_at_z6_error:.4f}"
        )
        print(f"    R^2    = {r_squared:.4f}")
    else:
        print("  Linear fit not calculated: fewer than three valid redshifts")

    # Use the corrected observational group membership directly. Marker shape
    # identifies the observational study; colour identifies the M_UV bin.
    obs_in_bin = luminosity_observations[
        luminosity_observations["MUV_group"] == observation_groups[label]
    ]

    for dataset, data in obs_in_bin.groupby("dataset", sort=False):
        ax.errorbar(
            data["redshift"],
            data["beta"],
            yerr=data["uncertainty"],
            fmt=observation_markers.get(dataset, "o"),
            linestyle="none",
            color=colour,
            ecolor=to_rgba(colour, 0.30),
            markerfacecolor="white",
            markeredgecolor=colour,
            markeredgewidth=0.9,
            markersize=5.0,
            capsize=1.5,
            elinewidth=0.65,
            alpha=1.0,
            zorder=4,
        )

ax.set_xlabel("Redshift")
ax.set_ylabel(r"UV slope, $\beta$")
ax.text(
    0.02, 0.97, r"UV-luminosity dependence",
    transform=ax.transAxes, ha="left", va="top"
)

# Keep luminosity-bin and observational-study labels in separate legends.
bin_legend = ax.legend(
    loc="upper right",
    title=r"Simulation: $M_{1500}$ bin",
    handlelength=2.2,
    borderaxespad=0.5,
)
ax.add_artist(bin_legend)

observation_handles = [
    Line2D(
        [0],
        [0],
        marker=marker,
        linestyle="none",
        markerfacecolor="white",
        markeredgecolor="0.35",
        markeredgewidth=0.9,
        markersize=4.8,
        label=dataset,
    )
    for dataset, marker in observation_markers.items()
]
ax.legend(
    handles=observation_handles,
    loc="lower left",
    title="Observations",
    borderaxespad=0.5,
)

plt.savefig(
    "Beta_evolution_luminosity_bins_calzetti.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    "Beta_evolution_luminosity_bins_calzetti.pdf",
    bbox_inches="tight",
)
plt.show()
