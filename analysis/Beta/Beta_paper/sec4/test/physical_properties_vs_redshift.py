"""Plot the redshift evolution of physical properties in SIMBA-EoR.

The fiducial sample matches the UV-beta analysis: galaxies are selected using
their dust-attenuated Calzetti M1500, with M1500 < -16 in m25n1024 and
M1500 < -17.5 in m50n1024.  Each panel shows the median and 16th--84th
percentile interval at the six available snapshots.

Outputs
-------
PhysicalProperties_vs_Redshift_Calzetti.png
PhysicalProperties_vs_Redshift_Calzetti.pdf
PhysicalProperties_vs_Redshift_Calzetti.csv
"""

from pathlib import Path

import caesar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# User-adjustable settings
# -----------------------------------------------------------------------------
DUST_LAW = "calzetti"
SNAPSHOTS = ["036", "030", "026", "022", "019", "016"]

DATA_DIR = Path(
    "/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/FullSpectra_Fit"
)
OUTPUT_STEM = Path("PhysicalProperties_vs_Redshift_Calzetti")

BOXES = {
    "m25n1024": {"magnitude_limit": -16.0, "colour": "#0072B2"},
    "m50n1024": {"magnitude_limit": -17.5, "colour": "#D55E00"},
}

# Plot metallicity relative to solar.  Set this to False to plot log10(Z),
# matching a raw mass-fraction definition instead.
METALLICITY_IN_SOLAR_UNITS = True
Z_SUN = 0.0134

# Show the two boxes separately as faint diagnostic curves in addition to the
# main combined-sample result.
SHOW_INDIVIDUAL_BOXES = True

PERCENTILES = (16, 50, 84)


# -----------------------------------------------------------------------------
# MNRAS-like figure style
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
    "legend.fontsize": 7.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "savefig.dpi": 300,
})


def catalogue_path(box, snapshot):
    """Return the Calzetti catalogue path for one box and snapshot."""
    return DATA_DIR / box / f"caesar_{box}_{snapshot}_{DUST_LAW}.hdf5"


def as_float(value):
    """Convert a scalar number or unit-aware scalar to float."""
    return float(getattr(value, "value", value))


def mass_in_msun(galaxy, component):
    """Read one CAESAR mass component in solar masses."""
    value = galaxy.masses[component]
    try:
        return value.to("Msun").value
    except AttributeError:
        return float(value)


def load_selected_sample(box, snapshot):
    """Load one snapshot and return physical properties after the M1500 cut."""
    filename = catalogue_path(box, snapshot)
    catalogue = caesar.load(str(filename))
    galaxies = catalogue.galaxies

    m1500 = np.asarray(
        [galaxy.absmag["i1500"] for galaxy in galaxies], dtype=float
    )
    selected = np.isfinite(m1500) & (m1500 < BOXES[box]["magnitude_limit"])

    stellar_mass = np.asarray(
        [mass_in_msun(galaxy, "stellar") for galaxy in galaxies], dtype=float
    )[selected]
    dust_mass = np.asarray(
        [mass_in_msun(galaxy, "dust") for galaxy in galaxies], dtype=float
    )[selected]
    stellar_metallicity = np.asarray(
        [galaxy.metallicities["stellar"] for galaxy in galaxies], dtype=float
    )[selected]
    gas_metallicity = np.asarray(
        [galaxy.metallicities["mass_weighted"] for galaxy in galaxies], dtype=float
    )[selected]

    metallicity_normalisation = Z_SUN if METALLICITY_IN_SOLAR_UNITS else 1.0

    # Non-positive values are undefined in logarithmic space and are retained
    # as NaN so that each property uses all of its own valid galaxies.
    properties = {
        "dust_mass": safe_log10(dust_mass),
        "stellar_metallicity": safe_log10(
            stellar_metallicity / metallicity_normalisation
        ),
        "gas_metallicity": safe_log10(
            gas_metallicity / metallicity_normalisation
        ),
        "stellar_mass": safe_log10(stellar_mass),
    }

    redshift = as_float(catalogue.simulation.redshift)
    return redshift, properties, int(np.count_nonzero(selected))


def safe_log10(values):
    """Return log10(values), assigning NaN to non-positive/invalid values."""
    values = np.asarray(values, dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values) & (values > 0)
    output[valid] = np.log10(values[valid])
    return output


def summarise(values):
    """Return N and the requested finite-sample percentiles."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0, np.nan, np.nan, np.nan
    p16, median, p84 = np.percentile(values, PERCENTILES)
    return len(values), p16, median, p84


def append_summary(rows, sample_name, snapshot, redshift, property_name, values,
                   selected_count):
    """Append one summary record for plotting and for the output CSV."""
    count, p16, median, p84 = summarise(values)
    rows.append({
        "sample": sample_name,
        "snapshot": snapshot,
        "redshift": redshift,
        "property": property_name,
        "selected_galaxies": selected_count,
        "valid_galaxies": count,
        "p16": p16,
        "median": median,
        "p84": p84,
    })


# -----------------------------------------------------------------------------
# Load all snapshots and construct box-specific and combined summaries
# -----------------------------------------------------------------------------
rows = []

for snapshot in SNAPSHOTS:
    snapshot_samples = {}
    snapshot_redshifts = {}
    snapshot_counts = {}

    for box in BOXES:
        redshift, properties, selected_count = load_selected_sample(box, snapshot)
        snapshot_samples[box] = properties
        snapshot_redshifts[box] = redshift
        snapshot_counts[box] = selected_count

        for property_name, values in properties.items():
            append_summary(
                rows, box, snapshot, redshift, property_name, values,
                selected_count,
            )

    # The snapshot redshifts should agree between boxes; use their mean to
    # avoid giving either catalogue special status at machine precision.
    combined_redshift = np.mean(list(snapshot_redshifts.values()))
    combined_count = sum(snapshot_counts.values())

    for property_name in next(iter(snapshot_samples.values())):
        combined_values = np.concatenate([
            snapshot_samples[box][property_name] for box in BOXES
        ])
        append_summary(
            rows, "combined", snapshot, combined_redshift, property_name,
            combined_values, combined_count,
        )

summary = pd.DataFrame(rows).sort_values(
    ["property", "sample", "redshift"]
)
summary.to_csv(OUTPUT_STEM.with_suffix(".csv"), index=False)


# -----------------------------------------------------------------------------
# Plot the four physical-property histories
# -----------------------------------------------------------------------------
if METALLICITY_IN_SOLAR_UNITS:
    stellar_z_label = r"Stellar metallicity, $\log_{10}(Z_\star/Z_\odot)$"
    gas_z_label = r"Gas metallicity, $\log_{10}(Z_{\rm gas}/Z_\odot)$"
else:
    stellar_z_label = r"Stellar metallicity, $\log_{10}(Z_\star)$"
    gas_z_label = r"Gas metallicity, $\log_{10}(Z_{\rm gas})$"

panels = [
    ("dust_mass", r"Dust mass, $\log_{10}(M_{\rm dust}/{\rm M}_\odot)$"),
    ("stellar_metallicity", stellar_z_label),
    ("gas_metallicity", gas_z_label),
    ("stellar_mass", r"Stellar mass, $\log_{10}(M_\star/{\rm M}_\odot)$"),
]

fig, axes = plt.subplots(
    2, 2, figsize=(7.1, 5.2), sharex=True, constrained_layout=True
)

for panel_label, (ax, (property_name, ylabel)) in zip(
    "abcd", zip(axes.flat, panels)
):
    panel_data = summary[summary["property"] == property_name]

    if SHOW_INDIVIDUAL_BOXES:
        for box, settings in BOXES.items():
            data = panel_data[panel_data["sample"] == box].sort_values("redshift")
            ax.plot(
                data["redshift"], data["median"], marker="o", markersize=3.3,
                linewidth=1.0, color=settings["colour"], alpha=0.58,
                label=box,
            )

    data = panel_data[panel_data["sample"] == "combined"].sort_values("redshift")
    redshift = data["redshift"].to_numpy()
    median = data["median"].to_numpy()
    p16 = data["p16"].to_numpy()
    p84 = data["p84"].to_numpy()

    ax.fill_between(
        redshift, p16, p84, color="0.55", alpha=0.22, linewidth=0,
        label="Combined 16th--84th percentile",
    )
    ax.plot(
        redshift, median, "o-", color="black", linewidth=1.7,
        markersize=4.5, markerfacecolor="black", markeredgecolor="white",
        markeredgewidth=0.55, label="Combined median", zorder=5,
    )

    ax.set_ylabel(ylabel)
    ax.text(
        0.04, 0.94, f"({panel_label})", transform=ax.transAxes,
        ha="left", va="top",
    )
    ax.minorticks_on()
    ax.tick_params(which="major", length=4.0, width=0.8)
    ax.tick_params(which="minor", length=2.2, width=0.6)
    ax.grid(False)

for ax in axes[-1, :]:
    ax.set_xlabel("Redshift")

# One compact shared legend keeps the data panels clear.
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.025),
    ncol=4 if SHOW_INDIVIDUAL_BOXES else 2, frameon=False,
    handlelength=2.2, columnspacing=1.2,
)

fig.savefig(OUTPUT_STEM.with_suffix(".png"), bbox_inches="tight")
fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)


# Print the combined medians as a quick terminal check.
print("\nCombined-sample medians:")
print(
    summary[summary["sample"] == "combined"][
        ["snapshot", "redshift", "property", "valid_galaxies", "median"]
    ].sort_values(["redshift", "property"]).to_string(index=False)
)
print(f"\nSaved {OUTPUT_STEM.with_suffix('.png')}")
print(f"Saved {OUTPUT_STEM.with_suffix('.pdf')}")
print(f"Saved {OUTPUT_STEM.with_suffix('.csv')}")
