"""Plot intrinsic UV slope against metallicity, sSFR, and redshift.

The script uses the same fiducial SIMBA-EoR selection as the main beta
analysis: dust-attenuated Calzetti M1500 < -16 for m25n1024 and < -17.5 for
m50n1024.  beta_nodust is then calculated from the *intrinsic* 1500, 2300,
and 2800 Angstrom absolute magnitudes.

Outputs
-------
BetaNodust_vs_Zstar_sSFR_Redshift.png
BetaNodust_vs_Zstar_sSFR_Redshift.pdf
BetaNodust_vs_Zstar_sSFR_Redshift_summary.csv
"""

from pathlib import Path

import caesar
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from utils.beta_utils import Calbeta


# -----------------------------------------------------------------------------
# User-adjustable settings
# -----------------------------------------------------------------------------
DUST_LAW = "calzetti"
SNAPSHOTS = ["036", "030", "026", "022", "019", "016"]

DATA_DIR = Path(
    "/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/FullSpectra_Fit"
)
OUTPUT_STEM = Path("BetaNodust_vs_Zstar_sSFR_Redshift")

BOXES = {
    "m25n1024": {"magnitude_limit": -16.0},
    "m50n1024": {"magnitude_limit": -17.5},
}

BANDS = ["i1500", "i2300", "i2800"]
WAVELENGTHS = np.array([1500.0, 2300.0, 2800.0])

# True gives log10(Z_star / Z_sun); False gives the raw log10(Z_star).
METALLICITY_IN_SOLAR_UNITS = True
Z_SUN = 0.0134

# Running-median settings for the two property panels.
NUMBER_OF_BINS = 12
MINIMUM_PER_BIN = 20
PERCENTILES = (16, 50, 84)


# -----------------------------------------------------------------------------
# MNRAS-like plotting style
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "savefig.dpi": 300,
})


def catalogue_path(box, snapshot):
    """Return the path to one post-processed CAESAR catalogue."""
    return DATA_DIR / box / f"caesar_{box}_{snapshot}_{DUST_LAW}.hdf5"


def mass_in_msun(galaxy, component):
    """Read a CAESAR mass component in solar masses."""
    value = galaxy.masses[component]
    try:
        return value.to("Msun").value
    except AttributeError:
        return float(value)


def sfr_in_msun_per_year(galaxy):
    """Read a CAESAR star-formation rate in Msun/yr."""
    value = galaxy.sfr
    try:
        return value.to("Msun/yr").value
    except AttributeError:
        return float(value)


def load_selected_sample(box, snapshot):
    """Load, select, and return the valid galaxies in one catalogue."""
    catalogue = caesar.load(str(catalogue_path(box, snapshot)))
    galaxies = catalogue.galaxies

    observed_m1500 = np.asarray(
        [galaxy.absmag["i1500"] for galaxy in galaxies], dtype=float
    )
    intrinsic_magnitudes = np.asarray([
        [galaxy.absmag_nodust[band] for galaxy in galaxies]
        for band in BANDS
    ], dtype=float)

    stellar_mass = np.asarray(
        [mass_in_msun(galaxy, "stellar") for galaxy in galaxies], dtype=float
    )
    stellar_metallicity = np.asarray(
        [galaxy.metallicities["stellar"] for galaxy in galaxies], dtype=float
    )
    sfr = np.asarray(
        [sfr_in_msun_per_year(galaxy) for galaxy in galaxies], dtype=float
    )

    beta_nodust = np.asarray(
        Calbeta(intrinsic_magnitudes, WAVELENGTHS), dtype=float
    )
    ssfr = sfr / stellar_mass
    metallicity_scale = Z_SUN if METALLICITY_IN_SOLAR_UNITS else 1.0

    selected = (
        np.isfinite(observed_m1500)
        & (observed_m1500 < BOXES[box]["magnitude_limit"])
        & np.isfinite(beta_nodust)
        & np.isfinite(stellar_metallicity)
        & (stellar_metallicity > 0)
        & np.isfinite(ssfr)
        & (ssfr > 0)
    )

    redshift = float(getattr(
        catalogue.simulation.redshift,
        "value",
        catalogue.simulation.redshift,
    ))

    sample = pd.DataFrame({
        "box": box,
        "snapshot": snapshot,
        "redshift": redshift,
        "beta_nodust": beta_nodust[selected],
        "log_stellar_metallicity": np.log10(
            stellar_metallicity[selected] / metallicity_scale
        ),
        "log_ssfr": np.log10(ssfr[selected]),
    })
    return redshift, sample


def equal_number_binned_summary(x, y, number_of_bins=NUMBER_OF_BINS):
    """Return equal-population running medians and 16th--84th percentiles."""
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[valid]
    y = np.asarray(y)[valid]

    if len(x) < MINIMUM_PER_BIN:
        return pd.DataFrame(columns=["x", "p16", "median", "p84", "count"])

    order = np.argsort(x)
    groups = np.array_split(order, min(number_of_bins, len(x) // MINIMUM_PER_BIN))
    rows = []
    for group in groups:
        if len(group) < MINIMUM_PER_BIN:
            continue
        p16, median, p84 = np.percentile(y[group], PERCENTILES)
        rows.append({
            "x": np.median(x[group]),
            "p16": p16,
            "median": median,
            "p84": p84,
            "count": len(group),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Load all snapshots and both boxes
# -----------------------------------------------------------------------------
samples = []
for snapshot in SNAPSHOTS:
    for box in BOXES:
        redshift, sample = load_selected_sample(box, snapshot)
        samples.append(sample)
        print(
            f"Loaded {box}, snapshot {snapshot}, z={redshift:.2f}: "
            f"{len(sample)} valid galaxies"
        )

data = pd.concat(samples, ignore_index=True)
redshifts = np.sort(data["redshift"].unique())


# Save one compact table containing the global redshift evolution and the
# running-median points used in both property panels.
summary_rows = []
for redshift in redshifts:
    at_z = data[np.isclose(data["redshift"], redshift)]
    p16, median, p84 = np.percentile(at_z["beta_nodust"], PERCENTILES)
    summary_rows.append({
        "relation": "redshift",
        "redshift": redshift,
        "x": redshift,
        "count": len(at_z),
        "beta_p16": p16,
        "beta_median": median,
        "beta_p84": p84,
    })

    for relation, column in [
        ("stellar_metallicity", "log_stellar_metallicity"),
        ("ssfr", "log_ssfr"),
    ]:
        binned = equal_number_binned_summary(
            at_z[column].to_numpy(), at_z["beta_nodust"].to_numpy()
        )
        for row in binned.itertuples(index=False):
            summary_rows.append({
                "relation": relation,
                "redshift": redshift,
                "x": row.x,
                "count": row.count,
                "beta_p16": row.p16,
                "beta_median": row.median,
                "beta_p84": row.p84,
            })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_summary.csv"), index=False)


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(
    1, 3, figsize=(7.1, 2.65), sharey=True,
    gridspec_kw={"wspace": 0.08},
)

colour_map = plt.get_cmap("viridis_r")
normalise_redshift = Normalize(vmin=redshifts.min(), vmax=redshifts.max())

property_panels = [
    (
        axes[0],
        "stellar_metallicity",
        r"$\log_{10}(Z_\star/Z_\odot)$" if METALLICITY_IN_SOLAR_UNITS
        else r"$\log_{10}(Z_\star)$",
    ),
    (
        axes[1],
        "ssfr",
        r"$\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})$",
    ),
]

for ax, relation, xlabel in property_panels:
    panel_summary = summary[summary["relation"] == relation]
    for redshift in redshifts:
        curve = panel_summary[
            np.isclose(panel_summary["redshift"], redshift)
        ].sort_values("x")
        colour = colour_map(normalise_redshift(redshift))
        ax.fill_between(
            curve["x"].to_numpy(),
            curve["beta_p16"].to_numpy(),
            curve["beta_p84"].to_numpy(),
            color=colour,
            alpha=0.08,
            linewidth=0,
        )
        ax.plot(
            curve["x"], curve["beta_median"],
            color=colour, linewidth=1.25,
        )

    ax.set_xlabel(xlabel)

axes[0].set_ylabel(r"Intrinsic UV slope, $\beta_{\rm nodust}$")

redshift_summary = summary[summary["relation"] == "redshift"].sort_values("x")
axes[2].fill_between(
    redshift_summary["x"].to_numpy(),
    redshift_summary["beta_p16"].to_numpy(),
    redshift_summary["beta_p84"].to_numpy(),
    color="#4477AA", alpha=0.20, linewidth=0,
    label="16th--84th percentile",
)
axes[2].plot(
    redshift_summary["x"], redshift_summary["beta_median"],
    "o-", color="#004488", linewidth=1.6, markersize=4.2,
    markeredgecolor="white", markeredgewidth=0.45,
    label="Median",
)
axes[2].set_xlabel("Redshift, $z$")
axes[2].legend(frameon=False, loc="best")

for panel_label, ax in zip("abc", axes):
    ax.text(
        0.04, 0.95, f"({panel_label})", transform=ax.transAxes,
        ha="left", va="top",
    )
    ax.minorticks_on()

# A compact horizontal colour bar identifies the redshift of curves in panels
# (a) and (b), while keeping the third panel uncluttered.
colourbar = fig.colorbar(
    ScalarMappable(norm=normalise_redshift, cmap=colour_map),
    ax=axes[:2], orientation="horizontal", fraction=0.09, pad=0.22,
    aspect=35,
)
colourbar.set_label("Redshift, $z$")
colourbar.set_ticks(redshifts)
colourbar.ax.invert_xaxis()

fig.subplots_adjust(left=0.09, right=0.985, top=0.97, bottom=0.30)
fig.savefig(OUTPUT_STEM.with_suffix(".png"), bbox_inches="tight")
fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
plt.show()

print(f"Saved {OUTPUT_STEM.with_suffix('.png')}")
print(f"Saved {OUTPUT_STEM.with_suffix('.pdf')}")
print(f"Saved {OUTPUT_STEM.with_name(OUTPUT_STEM.name + '_summary.csv')}")
