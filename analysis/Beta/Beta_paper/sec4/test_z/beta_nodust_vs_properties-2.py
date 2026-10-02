"""Diagnostic plots separating intrinsic UV-slope and dust effects.

Produces four figures: full-sample beta diagnostics, the same diagnostics in
narrow stellar-mass bins, physical-property evolution at fixed mass, and the
intrinsic UV slope--stellar-mass relation.

The selection matches the main analysis: observed Calzetti M1500 < -16 for
m25n1024 and < -17.5 for m50n1024. Both beta values use the same intrinsic or
attenuated 1500, 2300, and 2800 Angstrom magnitudes.
"""

from pathlib import Path
import warnings

import caesar
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from utils.beta_utils import Calbeta


# =============================================================================
# User-adjustable settings
# =============================================================================
DUST_LAW = "calzetti"
SNAPSHOTS = ["036", "030", "026", "022", "019", "016"]
DATA_DIR = Path(
    "/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/FullSpectra_Fit"
)
BOXES = {
    "m25n1024": {"magnitude_limit": -16.0},
    "m50n1024": {"magnitude_limit": -17.5},
}
BANDS = ["i1500", "i2300", "i2800"]
WAVELENGTHS = np.array([1500.0, 2300.0, 2800.0])

METALLICITY_IN_SOLAR_UNITS = True
Z_SUN = 0.0134

# Six narrow 0.5-dex bins. Adjust these edges if required.
MASS_BIN_EDGES = np.arange(7.5, 10.5 + 0.001, 0.5)
NUMBER_OF_RELATION_BINS = 10
MINIMUM_PER_RELATION_BIN = 20
MINIMUM_PER_MASS_REDSHIFT_BIN = 10
PERCENTILES = (16, 50, 84)

# CAESAR normally uses galaxy.ages["mass_weighted"].
STELLAR_AGE_KEYS = (
    "mass_weighted", "mass_weighted_stellar",
    "stellar_mass_weighted", "stellar",
)


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "savefig.dpi": 300,
})


def catalogue_path(box, snapshot):
    return DATA_DIR / box / f"caesar_{box}_{snapshot}_{DUST_LAW}.hdf5"


def scalar_value(value, unit=None):
    if unit is not None:
        try:
            return float(value.to(unit).value)
        except (AttributeError, TypeError, ValueError):
            pass
    return float(getattr(value, "value", value))


def mass_in_msun(galaxy, component):
    return scalar_value(galaxy.masses[component], "Msun")


def sfr_in_msun_per_year(galaxy):
    return scalar_value(galaxy.sfr, "Msun/yr")


def stellar_age_in_gyr(galaxy):
    """Return mass-weighted stellar age in Gyr, or NaN if unavailable."""
    ages = getattr(galaxy, "ages", None)
    if ages is not None:
        for key in STELLAR_AGE_KEYS:
            try:
                return scalar_value(ages[key], "Gyr")
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
    for attribute in ("stellar_age", "mass_weighted_age"):
        if hasattr(galaxy, attribute):
            try:
                return scalar_value(getattr(galaxy, attribute), "Gyr")
            except (TypeError, ValueError, AttributeError):
                continue
    return np.nan


def safe_log10(values):
    values = np.asarray(values, dtype=float)
    output = np.full(values.shape, np.nan)
    valid = np.isfinite(values) & (values > 0)
    output[valid] = np.log10(values[valid])
    return output


def load_selected_sample(box, snapshot):
    catalogue = caesar.load(str(catalogue_path(box, snapshot)))
    galaxies = catalogue.galaxies

    magnitudes_dust = np.asarray([
        [galaxy.absmag[band] for galaxy in galaxies] for band in BANDS
    ], dtype=float)
    magnitudes_nodust = np.asarray([
        [galaxy.absmag_nodust[band] for galaxy in galaxies] for band in BANDS
    ], dtype=float)
    beta_dust = np.asarray(Calbeta(magnitudes_dust, WAVELENGTHS), dtype=float)
    beta_nodust = np.asarray(Calbeta(magnitudes_nodust, WAVELENGTHS), dtype=float)

    stellar_mass = np.asarray(
        [mass_in_msun(g, "stellar") for g in galaxies], dtype=float
    )
    sfr = np.asarray([sfr_in_msun_per_year(g) for g in galaxies], dtype=float)
    stellar_metallicity = np.asarray(
        [g.metallicities["stellar"] for g in galaxies], dtype=float
    )
    gas_metallicity = np.asarray(
        [g.metallicities["mass_weighted"] for g in galaxies], dtype=float
    )
    stellar_age = np.asarray(
        [stellar_age_in_gyr(g) for g in galaxies], dtype=float
    )
    av = np.asarray([
        g.absmag["v"] - g.absmag_nodust["v"] for g in galaxies
    ], dtype=float)

    selected = (
        np.isfinite(magnitudes_dust[0])
        & (magnitudes_dust[0] < BOXES[box]["magnitude_limit"])
        & np.isfinite(stellar_mass) & (stellar_mass > 0)
        & np.isfinite(beta_dust) & np.isfinite(beta_nodust)
    )
    redshift = scalar_value(catalogue.simulation.redshift)
    z_scale = Z_SUN if METALLICITY_IN_SOLAR_UNITS else 1.0

    sample = pd.DataFrame({
        "box": box,
        "snapshot": snapshot,
        "redshift": redshift,
        "log_stellar_mass": safe_log10(stellar_mass[selected]),
        "beta_dust": beta_dust[selected],
        "beta_nodust": beta_nodust[selected],
        "delta_beta": beta_dust[selected] - beta_nodust[selected],
        "log_stellar_metallicity": safe_log10(
            stellar_metallicity[selected] / z_scale
        ),
        "log_gas_metallicity": safe_log10(gas_metallicity[selected] / z_scale),
        "log_ssfr": safe_log10(sfr[selected] / stellar_mass[selected]),
        "stellar_age_gyr": stellar_age[selected],
        "av": av[selected],
    })
    return redshift, sample


def percentile_record(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0, np.nan, np.nan, np.nan
    p16, median, p84 = np.percentile(values, PERCENTILES)
    return len(values), p16, median, p84


def equal_number_relation(x, y, number_of_bins=NUMBER_OF_RELATION_BINS):
    """Summarise y(x) in approximately equal-population x bins."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    maximum_bins = len(x) // MINIMUM_PER_RELATION_BIN
    if maximum_bins < 1:
        return pd.DataFrame(columns=["x", "p16", "median", "p84", "count"])
    groups = np.array_split(
        np.argsort(x), min(number_of_bins, maximum_bins)
    )
    rows = []
    for group in groups:
        if len(group) < MINIMUM_PER_RELATION_BIN:
            continue
        count, p16, median, p84 = percentile_record(y[group])
        rows.append({"x": np.median(x[group]), "p16": p16,
                     "median": median, "p84": p84, "count": count})
    return pd.DataFrame(rows)


def mass_bin_label(left, right):
    return rf"${left:.1f}\leq\log_{{10}}(M_\star/{{\rm M}}_\odot)<{right:.1f}$"


def in_mass_bin(frame, left, right):
    return frame[
        (frame["log_stellar_mass"] >= left)
        & (frame["log_stellar_mass"] < right)
    ]


def format_axes(axes):
    for label, ax in zip("abcdefghijklmnopqrstuvwxyz", np.ravel(axes)):
        ax.text(0.04, 0.95, f"({label})", transform=ax.transAxes,
                ha="left", va="top")
        ax.minorticks_on()


def save_figure(fig, stem):
    fig.savefig(Path(stem).with_suffix(".png"), bbox_inches="tight")
    fig.savefig(Path(stem).with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {stem}.png and {stem}.pdf")


# Load all data once so every figure uses exactly the same sample.
samples = []
for snapshot in SNAPSHOTS:
    for box in BOXES:
        redshift, sample = load_selected_sample(box, snapshot)
        samples.append(sample)
        print(f"Loaded {box}, snapshot {snapshot}, z={redshift:.2f}: "
              f"{len(sample)} selected galaxies")

data = pd.concat(samples, ignore_index=True)
redshifts = np.sort(data["redshift"].unique())
if not np.any(np.isfinite(data["stellar_age_gyr"])):
    warnings.warn(
        "No stellar ages were found; the age panel will be empty. Inspect "
        "one galaxy's `ages` keys and update STELLAR_AGE_KEYS."
    )

redshift_cmap = plt.get_cmap("viridis_r")
redshift_norm = Normalize(vmin=redshifts.min(), vmax=redshifts.max())
mass_intervals = list(zip(MASS_BIN_EDGES[:-1], MASS_BIN_EDGES[1:]))
mass_colours = plt.get_cmap("plasma")(
    np.linspace(0.08, 0.90, len(mass_intervals))
)
summary_rows = []


# =============================================================================
# Figure 1: full-sample relations
# =============================================================================
fig1, axes1 = plt.subplots(2, 2, figsize=(7.1, 5.35))
full_relations = [
    (axes1[0, 0], "log_stellar_metallicity", "beta_nodust",
     r"$\log_{10}(Z_\star/Z_\odot)$" if METALLICITY_IN_SOLAR_UNITS
     else r"$\log_{10}(Z_\star)$",
     r"Intrinsic UV slope, $\beta_{\rm nodust}$",
     "beta_nodust_vs_stellar_metallicity"),
    (axes1[0, 1], "log_ssfr", "beta_nodust",
     r"$\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})$",
     r"Intrinsic UV slope, $\beta_{\rm nodust}$", "beta_nodust_vs_ssfr"),
    (axes1[1, 1], "av", "delta_beta", r"$A_V$ (mag)",
     r"$\Delta\beta=\beta_{\rm dust}-\beta_{\rm nodust}$",
     "delta_beta_vs_av"),
]

for ax, xcol, ycol, xlabel, ylabel, relation_name in full_relations:
    for redshift in redshifts:
        at_z = data[np.isclose(data["redshift"], redshift)]
        curve = equal_number_relation(at_z[xcol], at_z[ycol])
        if curve.empty:
            continue
        colour = redshift_cmap(redshift_norm(redshift))
        ax.fill_between(curve["x"], curve["p16"], curve["p84"],
                        color=colour, alpha=0.07, linewidth=0)
        ax.plot(curve["x"], curve["median"], color=colour, linewidth=1.3)
        for row in curve.itertuples(index=False):
            summary_rows.append({
                "figure": "full_sample", "relation": relation_name,
                "redshift": redshift, "mass_bin_left": np.nan,
                "mass_bin_right": np.nan, "x": row.x, "count": row.count,
                "p16": row.p16, "median": row.median, "p84": row.p84,
            })
    ax.set(xlabel=xlabel, ylabel=ylabel)

z_rows = []
for redshift in redshifts:
    values = data.loc[np.isclose(data["redshift"], redshift), "beta_nodust"]
    count, p16, median, p84 = percentile_record(values)
    z_rows.append((redshift, count, p16, median, p84))
    summary_rows.append({
        "figure": "full_sample", "relation": "beta_nodust_vs_redshift",
        "redshift": redshift, "mass_bin_left": np.nan,
        "mass_bin_right": np.nan, "x": redshift, "count": count,
        "p16": p16, "median": median, "p84": p84,
    })
z_table = pd.DataFrame(z_rows, columns=["z", "count", "p16", "median", "p84"])
ax = axes1[1, 0]
ax.fill_between(z_table["z"], z_table["p16"], z_table["p84"],
                color="#4477AA", alpha=0.20, linewidth=0)
ax.plot(z_table["z"], z_table["median"], "o-", color="#004488",
        linewidth=1.6, markersize=4.2, markeredgecolor="white",
        markeredgewidth=0.45)
ax.set(xlabel="Redshift, $z$", ylabel=r"Intrinsic UV slope, $\beta_{\rm nodust}$")

format_axes(axes1)
colourbar = fig1.colorbar(
    ScalarMappable(norm=redshift_norm, cmap=redshift_cmap),
    ax=axes1, orientation="horizontal", fraction=0.055, pad=0.10, aspect=45,
)
colourbar.set_label("Redshift, $z$")
colourbar.set_ticks(redshifts)
colourbar.ax.invert_xaxis()
fig1.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=0.18,
                     wspace=0.28, hspace=0.28)
save_figure(fig1, "BetaDiagnostics_FullSample")


# =============================================================================
# Figure 2: the same relations in narrow stellar-mass bins
# =============================================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(7.1, 5.35))
mass_relations = [
    (axes2[0, 0], "log_stellar_metallicity", "beta_nodust",
     r"$\log_{10}(Z_\star/Z_\odot)$" if METALLICITY_IN_SOLAR_UNITS
     else r"$\log_{10}(Z_\star)$", r"$\beta_{\rm nodust}$",
     "beta_nodust_vs_stellar_metallicity"),
    (axes2[0, 1], "log_ssfr", "beta_nodust",
     r"$\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})$",
     r"$\beta_{\rm nodust}$", "beta_nodust_vs_ssfr"),
    (axes2[1, 1], "av", "delta_beta", r"$A_V$ (mag)",
     r"$\Delta\beta$", "delta_beta_vs_av"),
]

for ax, xcol, ycol, xlabel, ylabel, relation_name in mass_relations:
    for (left, right), colour in zip(mass_intervals, mass_colours):
        mass_sample = in_mass_bin(data, left, right)
        curve = equal_number_relation(mass_sample[xcol], mass_sample[ycol])
        if curve.empty:
            continue
        ax.plot(curve["x"], curve["median"], color=colour, linewidth=1.35,
                label=mass_bin_label(left, right))
        for row in curve.itertuples(index=False):
            summary_rows.append({
                "figure": "fixed_mass", "relation": relation_name,
                "redshift": np.nan, "mass_bin_left": left,
                "mass_bin_right": right, "x": row.x, "count": row.count,
                "p16": row.p16, "median": row.median, "p84": row.p84,
            })
    ax.set(xlabel=xlabel, ylabel=ylabel)

ax = axes2[1, 0]
for (left, right), colour in zip(mass_intervals, mass_colours):
    mass_sample = in_mass_bin(data, left, right)
    rows = []
    for redshift in redshifts:
        values = mass_sample.loc[
            np.isclose(mass_sample["redshift"], redshift), "beta_nodust"
        ]
        count, p16, median, p84 = percentile_record(values)
        if count < MINIMUM_PER_MASS_REDSHIFT_BIN:
            continue
        rows.append((redshift, count, p16, median, p84))
        summary_rows.append({
            "figure": "fixed_mass", "relation": "beta_nodust_vs_redshift",
            "redshift": redshift, "mass_bin_left": left,
            "mass_bin_right": right, "x": redshift, "count": count,
            "p16": p16, "median": median, "p84": p84,
        })
    if rows:
        curve = pd.DataFrame(rows, columns=["z", "count", "p16", "median", "p84"])
        ax.plot(curve["z"], curve["median"], "o-", color=colour,
                linewidth=1.3, markersize=3.3, label=mass_bin_label(left, right))
ax.set(xlabel="Redshift, $z$", ylabel=r"$\beta_{\rm nodust}$")

format_axes(axes2)
handles, labels = axes2[1, 0].get_legend_handles_labels()
fig2.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
            bbox_to_anchor=(0.5, -0.005), columnspacing=1.2, handlelength=2.0)
fig2.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=0.20,
                     wspace=0.25, hspace=0.27)
save_figure(fig2, "BetaDiagnostics_FixedMass")


# =============================================================================
# Figure 3: age, metallicity and attenuation evolution at fixed mass
# =============================================================================
fig3, axes3 = plt.subplots(2, 2, figsize=(7.1, 5.35), sharex=True)
history_properties = [
    (axes3[0, 0], "stellar_age_gyr", "Stellar age (Gyr)", "stellar_age"),
    (axes3[0, 1], "log_stellar_metallicity",
     r"$\log_{10}(Z_\star/Z_\odot)$" if METALLICITY_IN_SOLAR_UNITS
     else r"$\log_{10}(Z_\star)$", "stellar_metallicity"),
    (axes3[1, 0], "log_gas_metallicity",
     r"$\log_{10}(Z_{\rm gas}/Z_\odot)$" if METALLICITY_IN_SOLAR_UNITS
     else r"$\log_{10}(Z_{\rm gas})$", "gas_metallicity"),
    (axes3[1, 1], "av", r"$A_V$ (mag)", "av"),
]

for ax, column, ylabel, relation_name in history_properties:
    for (left, right), colour in zip(mass_intervals, mass_colours):
        mass_sample = in_mass_bin(data, left, right)
        rows = []
        for redshift in redshifts:
            values = mass_sample.loc[
                np.isclose(mass_sample["redshift"], redshift), column
            ]
            count, p16, median, p84 = percentile_record(values)
            if count < MINIMUM_PER_MASS_REDSHIFT_BIN:
                continue
            rows.append((redshift, count, p16, median, p84))
            summary_rows.append({
                "figure": "physical_history_fixed_mass",
                "relation": relation_name, "redshift": redshift,
                "mass_bin_left": left, "mass_bin_right": right,
                "x": redshift, "count": count, "p16": p16,
                "median": median, "p84": p84,
            })
        if rows:
            curve = pd.DataFrame(
                rows, columns=["z", "count", "p16", "median", "p84"]
            )
            ax.plot(curve["z"], curve["median"], "o-", color=colour,
                    linewidth=1.3, markersize=3.3,
                    label=mass_bin_label(left, right))
    ax.set_ylabel(ylabel)

axes3[1, 0].set_xlabel("Redshift, $z$")
axes3[1, 1].set_xlabel("Redshift, $z$")
format_axes(axes3)
handles, labels = axes3[1, 1].get_legend_handles_labels()
fig3.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
            bbox_to_anchor=(0.5, -0.005), columnspacing=1.2, handlelength=2.0)
fig3.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=0.20,
                     wspace=0.25, hspace=0.14)
save_figure(fig3, "PhysicalProperties_FixedMass_vs_Redshift")


# =============================================================================
# Figure 4: intrinsic UV slope versus stellar mass
# =============================================================================
fig4, ax4 = plt.subplots(figsize=(3.5, 3.15))

for redshift in redshifts:
    at_z = data[np.isclose(data["redshift"], redshift)]
    curve = equal_number_relation(
        at_z["log_stellar_mass"], at_z["beta_nodust"]
    )
    if curve.empty:
        continue

    colour = redshift_cmap(redshift_norm(redshift))
    ax4.fill_between(
        curve["x"], curve["p16"], curve["p84"],
        color=colour, alpha=0.08, linewidth=0,
    )
    ax4.plot(
        curve["x"], curve["median"], color=colour,
        linewidth=1.45, label=rf"$z={redshift:.1f}$",
    )

    for row in curve.itertuples(index=False):
        summary_rows.append({
            "figure": "beta_nodust_stellar_mass",
            "relation": "beta_nodust_vs_stellar_mass",
            "redshift": redshift,
            "mass_bin_left": np.nan,
            "mass_bin_right": np.nan,
            "x": row.x,
            "count": row.count,
            "p16": row.p16,
            "median": row.median,
            "p84": row.p84,
        })

ax4.set_xlabel(r"$\log_{10}(M_\star/{\rm M}_\odot)$")
ax4.set_ylabel(r"Intrinsic UV slope, $\beta_{\rm nodust}$")
ax4.text(0.04, 0.95, "(a)", transform=ax4.transAxes,
         ha="left", va="top")
ax4.minorticks_on()

colourbar4 = fig4.colorbar(
    ScalarMappable(norm=redshift_norm, cmap=redshift_cmap),
    ax=ax4, orientation="horizontal", fraction=0.08, pad=0.20, aspect=28,
)
colourbar4.set_label("Redshift, $z$")
colourbar4.set_ticks(redshifts)
colourbar4.ax.invert_xaxis()

fig4.subplots_adjust(left=0.18, right=0.97, top=0.97, bottom=0.28)
save_figure(fig4, "BetaNodust_vs_StellarMass")


pd.DataFrame(summary_rows).to_csv("BetaDiagnostics_AllSummaries.csv", index=False)
print("Saved BetaDiagnostics_AllSummaries.csv")
plt.show()
