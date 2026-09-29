"""Plot intrinsic UV-slope diagnostics in three grouped redshift intervals.

Panels
------
(a) beta_nodust versus stellar mass
(b) beta_nodust versus sSFR
(c) beta_nodust versus stellar metallicity
(d) Delta beta = beta_dust - beta_nodust versus A_V

Each curve combines two snapshots and both SIMBA-EoR volumes. Galaxies are
selected using their dust-attenuated Calzetti M1500, matching the fiducial
paper sample. Both beta values are calculated with the same three bands.
"""

from pathlib import Path

import caesar
import matplotlib.pyplot as plt
import numpy as np

from utils.beta_utils import Calbeta, bin_xy_median


# =============================================================================
# User-adjustable settings
# =============================================================================
DUST_LAW = "calzetti"

REDSHIFT_BINS = [
    (r"$z\simeq6$--$7$", ["036", "030"]),
    (r"$z\simeq8$--$9$", ["026", "022"]),
    (r"$z\simeq10$--$11$", ["019", "016"]),
]

TEMPLATE_M25 = (
    "/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/FullSpectra_Fit/"
    "m25n1024/caesar_m25n1024_{}_{}.hdf5"
)
TEMPLATE_M50 = (
    "/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/FullSpectra_Fit/"
    "m50n1024/caesar_m50n1024_{}_{}.hdf5"
)

MAGNITUDE_LIMITS = {"m25n1024": -16.0, "m50n1024": -17.5}
BANDS = ["i1500", "i2300", "i2800"]
WAVELENGTHS = np.array([1500.0, 2300.0, 2800.0])
Z_SUN = 0.0134
NUMBER_OF_BINS = 10

# Exclude the single extreme metallicity value that otherwise stretches the
# equal-width bins in panel (c). This cut is applied only to the z~6--7
# stellar-metallicity relation and does not affect the other panels.
STELLAR_METALLICITY_MAX_Z6_Z7 = 0.7

OUTPUT_STEM = Path("BetaIntrinsicDrivers_Calzetti_RedshiftBins")


# =============================================================================
# MNRAS-style plotting defaults
# =============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.formatter.use_mathtext": True,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "savefig.dpi": 300,
})


REDSHIFT_STYLES = [
    {"color": "#482878", "marker": "o", "linestyle": "-"},
    {"color": "#238A8D", "marker": "s", "linestyle": "--"},
    {"color": "#D8A800", "marker": "^", "linestyle": "-."},
]


PANEL_SETTINGS = [
    {
        "key": "log_stellar_mass",
        "y_key": "beta_nodust",
        "xlabel": r"$\log_{10}(M_\star/{\rm M}_\odot)$",
        "ylabel": r"Intrinsic UV slope, $\beta_{\rm nodust}$",
    },
    {
        "key": "log_ssfr",
        "y_key": "beta_nodust",
        "xlabel": r"$\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})$",
        "ylabel": r"Intrinsic UV slope, $\beta_{\rm nodust}$",
    },
    {
        "key": "stellar_metallicity",
        "y_key": "beta_nodust",
        "xlabel": r"$Z_\star/Z_\odot$",
        "ylabel": r"Intrinsic UV slope, $\beta_{\rm nodust}$",
    },
    {
        "key": "av",
        "y_key": "delta_beta",
        "xlabel": r"$A_V$ (mag)",
        "ylabel": r"$\Delta\beta=\beta_{\rm dust}-\beta_{\rm nodust}$",
    },
]


def scalar_value(value, unit=None):
    """Convert a scalar or unit-aware scalar to float."""
    if unit is not None:
        try:
            return float(value.to(unit).value)
        except (AttributeError, TypeError, ValueError):
            pass
    return float(getattr(value, "value", value))


def stellar_metallicity_in_solar(galaxy):
    """Return Z_star/Z_sun for unit-aware or raw mass-fraction catalogues."""
    value = galaxy.metallicities["stellar"]
    try:
        return float(value.to("Zsun").value)
    except (AttributeError, TypeError, ValueError):
        return scalar_value(value) / Z_SUN


def load_catalogue_sample(filename, magnitude_limit):
    """Load one catalogue and return the selected intrinsic/dust diagnostics."""
    catalogue = caesar.load(filename)
    galaxies = catalogue.galaxies

    magnitudes_dust = np.asarray([
        [galaxy.absmag[band] for galaxy in galaxies]
        for band in BANDS
    ], dtype=float)
    magnitudes_nodust = np.asarray([
        [galaxy.absmag_nodust[band] for galaxy in galaxies]
        for band in BANDS
    ], dtype=float)

    beta_dust = np.asarray(Calbeta(magnitudes_dust, WAVELENGTHS), dtype=float)
    beta_nodust = np.asarray(
        Calbeta(magnitudes_nodust, WAVELENGTHS), dtype=float
    )

    stellar_mass = np.asarray([
        scalar_value(galaxy.masses["stellar"], "Msun")
        for galaxy in galaxies
    ])
    ssfr = np.asarray([
        scalar_value(galaxy.sfr / galaxy.masses["stellar"], "1/yr")
        for galaxy in galaxies
    ])
    stellar_metallicity = np.asarray([
        stellar_metallicity_in_solar(galaxy) for galaxy in galaxies
    ])
    av = np.asarray([
        galaxy.absmag["v"] - galaxy.absmag_nodust["v"]
        for galaxy in galaxies
    ], dtype=float)

    selected = (
        np.isfinite(magnitudes_dust[0])
        & (magnitudes_dust[0] < magnitude_limit)
        & np.isfinite(beta_dust)
        & np.isfinite(beta_nodust)
        & np.isfinite(stellar_mass)
        & (stellar_mass > 0)
    )

    log_ssfr = np.full(len(galaxies), np.nan)
    valid_ssfr = np.isfinite(ssfr) & (ssfr > 0)
    log_ssfr[valid_ssfr] = np.log10(ssfr[valid_ssfr])

    return {
        "log_stellar_mass": np.log10(stellar_mass[selected]),
        "log_ssfr": log_ssfr[selected],
        "stellar_metallicity": stellar_metallicity[selected],
        "av": av[selected],
        "beta_nodust": beta_nodust[selected],
        "delta_beta": beta_dust[selected] - beta_nodust[selected],
    }


def combine_samples(samples):
    return {
        key: np.concatenate([sample[key] for sample in samples])
        for key in samples[0]
    }


def binned_relation(x_values, y_values):
    """Return finite binned medians and 16th--84th percentile limits."""
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = np.asarray(x_values)[valid]
    y_values = np.asarray(y_values)[valid]

    centers, median, p16, p84, counts = bin_xy_median(
        x_values=x_values,
        y_values=y_values,
        mask_values=None,
        mask_cut=None,
        N_bins=NUMBER_OF_BINS,
    )
    centers = np.asarray(centers)
    median = np.asarray(median)
    p16 = np.asarray(p16)
    p84 = np.asarray(p84)
    counts = np.asarray(counts)
    finite = (
        np.isfinite(centers) & np.isfinite(median)
        & np.isfinite(p16) & np.isfinite(p84)
    )
    return centers[finite], median[finite], p16[finite], p84[finite], counts[finite]


# =============================================================================
# Load and group all samples
# =============================================================================
grouped_samples = []

for redshift_label, snapshots in REDSHIFT_BINS:
    samples = []
    for snapshot in snapshots:
        samples.append(load_catalogue_sample(
            TEMPLATE_M25.format(snapshot, DUST_LAW),
            MAGNITUDE_LIMITS["m25n1024"],
        ))
        samples.append(load_catalogue_sample(
            TEMPLATE_M50.format(snapshot, DUST_LAW),
            MAGNITUDE_LIMITS["m50n1024"],
        ))

    combined = combine_samples(samples)
    grouped_samples.append((redshift_label, combined))
    print(
        f"{redshift_label}: {len(combined['beta_nodust'])} selected galaxies"
    )


# =============================================================================
# Plot the four requested diagnostics
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.7))
axes = axes.flatten()

for redshift_index, ((redshift_label, sample), style) in enumerate(
    zip(grouped_samples, REDSHIFT_STYLES)
):
    for ax, settings in zip(axes, PANEL_SETTINGS):
        x_values = sample[settings["key"]]
        y_values = sample[settings["y_key"]]

        if redshift_index == 0 and settings["key"] == "stellar_metallicity":
            metallicity_outlier = (
                np.isfinite(x_values)
                & (x_values >= STELLAR_METALLICITY_MAX_Z6_Z7)
            )
            keep = ~metallicity_outlier
            print(
                f"{redshift_label}, panel (c): excluded "
                f"{np.count_nonzero(metallicity_outlier)} galaxy/galaxies with "
                f"Z_star/Z_sun >= {STELLAR_METALLICITY_MAX_Z6_Z7}"
            )
            x_values = x_values[keep]
            y_values = y_values[keep]

        x, median, p16, p84, counts = binned_relation(
            x_values, y_values
        )

        ax.errorbar(
            x,
            median,
            yerr=[median - p16, p84 - median],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.55,
            markersize=4.0,
            markeredgewidth=0.5,
            capsize=1.8,
            capthick=0.75,
            elinewidth=0.9,
            label=redshift_label,
            zorder=3,
        )


# Panels (a)--(c) share an intrinsic-beta scale; panel (d) has a distinct
# Delta-beta scale and is therefore deliberately not sharey-linked.
for ax in axes[:3]:
    ax.set_ylim(-2.62, -2.08)
    ax.set_yticks(np.arange(-2.6, -2.0, 0.1))

axes[3].axhline(0.0, color="0.55", linewidth=0.7, linestyle=":", zorder=0)
axes[3].set_ylim(bottom=-0.02)

for panel_label, ax, settings in zip("abcd", axes, PANEL_SETTINGS):
    ax.set_xlabel(settings["xlabel"])
    ax.set_ylabel(settings["ylabel"])
    ax.text(
        0.04, 0.95, f"({panel_label})",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
    )
    ax.minorticks_on()
    ax.tick_params(which="major", length=4.0, width=0.8)
    ax.tick_params(which="minor", length=2.2, width=0.6)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    frameon=False, ncol=3, loc="upper center",
    bbox_to_anchor=(0.5, 0.985),
    handlelength=2.6, handletextpad=0.6, columnspacing=1.5,
)

fig.subplots_adjust(
    left=0.105, right=0.985, bottom=0.09, top=0.91,
    wspace=0.27, hspace=0.28,
)

fig.savefig(OUTPUT_STEM.with_suffix(".png"), bbox_inches="tight")
fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")

print(f"Saved {OUTPUT_STEM.with_suffix('.png')}")
print(f"Saved {OUTPUT_STEM.with_suffix('.pdf')}")
plt.show()
