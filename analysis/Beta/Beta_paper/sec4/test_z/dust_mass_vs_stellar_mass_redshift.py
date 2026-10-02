"""Plot dust mass versus stellar mass across the six SIMBA-EoR snapshots.

The galaxy selection matches the UV-beta analysis: dust-attenuated Calzetti
M1500 < -16 for m25n1024 and M1500 < -17.5 for m50n1024.  Each redshift panel
shows individual galaxies and the combined-sample median relation with its
16th--84th percentile interval.

Outputs
-------
DustMass_vs_StellarMass_RedshiftPanels_Calzetti.png
DustMass_vs_StellarMass_RedshiftPanels_Calzetti.pdf
DustMass_vs_StellarMass_RedshiftPanels_Calzetti.csv
"""

from pathlib import Path

import caesar
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
OUTPUT_STEM = Path("DustMass_vs_StellarMass_RedshiftPanels_Calzetti")

BOXES = {
    "m25n1024": {"magnitude_limit": -16.0, "colour": "#0072B2"},
    "m50n1024": {"magnitude_limit": -17.5, "colour": "#D55E00"},
}

NUMBER_OF_MASS_BINS = 8
MINIMUM_GALAXIES_PER_BIN = 15
MAXIMUM_BACKGROUND_POINTS_PER_BOX = 10000
LOWER_PERCENTILE = 16
UPPER_PERCENTILE = 84
RANDOM_SEED = 21


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
    """Return the catalogue path for one box and snapshot."""
    return DATA_DIR / box / f"caesar_{box}_{snapshot}_{DUST_LAW}.hdf5"


def mass_in_msun(galaxy, component):
    """Read one CAESAR mass component in solar masses."""
    value = galaxy.masses[component]
    try:
        return value.to("Msun").value
    except AttributeError:
        return float(value)


def scalar_value(value):
    """Convert a number or unit-aware scalar to a plain float."""
    return float(getattr(value, "value", value))


def load_selected_sample(box, snapshot):
    """Return selected stellar masses, dust masses, and catalogue redshift."""
    catalogue = caesar.load(str(catalogue_path(box, snapshot)))
    galaxies = catalogue.galaxies

    m1500 = np.asarray(
        [galaxy.absmag["i1500"] for galaxy in galaxies], dtype=float
    )
    stellar_mass = np.asarray(
        [mass_in_msun(galaxy, "stellar") for galaxy in galaxies], dtype=float
    )
    dust_mass = np.asarray(
        [mass_in_msun(galaxy, "dust") for galaxy in galaxies], dtype=float
    )

    selected = (
        np.isfinite(m1500)
        & (m1500 < BOXES[box]["magnitude_limit"])
        & np.isfinite(stellar_mass)
        & (stellar_mass > 0)
    )
    dust_positive = selected & np.isfinite(dust_mass) & (dust_mass > 0)

    return {
        "box": box,
        "snapshot": snapshot,
        "redshift": scalar_value(catalogue.simulation.redshift),
        "log_stellar_mass": np.log10(stellar_mass[dust_positive]),
        "log_dust_mass": np.log10(dust_mass[dust_positive]),
        "selected_count": int(np.count_nonzero(selected)),
        "dust_positive_count": int(np.count_nonzero(dust_positive)),
    }


def binned_relation(log_stellar_mass, log_dust_mass, bin_edges):
    """Calculate the median and 16th--84th percentile relation."""
    rows = []

    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (
            (log_stellar_mass >= left)
            & (log_stellar_mass < right)
            & np.isfinite(log_dust_mass)
        )
        count = np.count_nonzero(in_bin)
        if count < MINIMUM_GALAXIES_PER_BIN:
            continue

        mass_values = log_stellar_mass[in_bin]
        dust_values = log_dust_mass[in_bin]
        p16, median, p84 = np.percentile(
            dust_values, [LOWER_PERCENTILE, 50, UPPER_PERCENTILE]
        )
        rows.append({
            "mass_bin_left": left,
            "mass_bin_right": right,
            "mass_median": np.median(mass_values),
            "dust_p16": p16,
            "dust_median": median,
            "dust_p84": p84,
            "count": count,
        })

    return pd.DataFrame(rows)


def subsample_indices(number, maximum, rng):
    """Return indices for an unbiased plotting-only subsample."""
    if number <= maximum:
        return np.arange(number)
    return rng.choice(number, size=maximum, replace=False)


# -----------------------------------------------------------------------------
# Load the selected samples.  A common mass-bin definition is used in every
# panel so that the relation can be compared directly across redshift.
# -----------------------------------------------------------------------------
samples = []
for snapshot in SNAPSHOTS:
    for box in BOXES:
        sample = load_selected_sample(box, snapshot)
        samples.append(sample)
        positive_fraction = (
            sample["dust_positive_count"] / sample["selected_count"]
            if sample["selected_count"] > 0 else np.nan
        )
        print(
            f"{box}, snapshot {snapshot}, z={sample['redshift']:.3f}: "
            f"selected={sample['selected_count']}, "
            f"M_dust>0={sample['dust_positive_count']} "
            f"({positive_fraction:.3f})"
        )

all_log_stellar_mass = np.concatenate([
    sample["log_stellar_mass"] for sample in samples
])
all_log_dust_mass = np.concatenate([
    sample["log_dust_mass"] for sample in samples
])
mass_min, mass_max = np.percentile(all_log_stellar_mass, [0.5, 99.5])
dust_min, dust_max = np.percentile(all_log_dust_mass, [0.5, 99.5])
mass_bin_edges = np.linspace(mass_min, mass_max, NUMBER_OF_MASS_BINS + 1)


# -----------------------------------------------------------------------------
# Plot one panel per snapshot
# -----------------------------------------------------------------------------
rng = np.random.default_rng(RANDOM_SEED)
fig, axes = plt.subplots(
    2, 3, figsize=(7.1, 4.8), sharex=True, sharey=True,
    constrained_layout=True,
)

summary_tables = []
snapshot_groups = []

for snapshot in SNAPSHOTS:
    snapshot_samples = [
        sample for sample in samples if sample["snapshot"] == snapshot
    ]
    redshift = np.mean([sample["redshift"] for sample in snapshot_samples])
    snapshot_groups.append((redshift, snapshot, snapshot_samples))

# Arrange panels from low to high redshift, matching the usual beta-evolution
# figures used elsewhere in the paper.
snapshot_groups.sort(key=lambda item: item[0])

for panel_label, ax, (redshift, snapshot, snapshot_samples) in zip(
    "abcdef", axes.flat, snapshot_groups
):
    combined_mass = []
    combined_dust = []
    selected_total = 0
    dust_positive_total = 0

    for sample in snapshot_samples:
        colour = BOXES[sample["box"]]["colour"]
        indices = subsample_indices(
            len(sample["log_stellar_mass"]),
            MAXIMUM_BACKGROUND_POINTS_PER_BOX,
            rng,
        )
        ax.scatter(
            sample["log_stellar_mass"][indices],
            sample["log_dust_mass"][indices],
            s=3.0, color=colour, alpha=0.13, linewidths=0,
            rasterized=True, zorder=1,
        )
        combined_mass.append(sample["log_stellar_mass"])
        combined_dust.append(sample["log_dust_mass"])
        selected_total += sample["selected_count"]
        dust_positive_total += sample["dust_positive_count"]

    combined_mass = np.concatenate(combined_mass)
    combined_dust = np.concatenate(combined_dust)
    relation = binned_relation(combined_mass, combined_dust, mass_bin_edges)
    relation.insert(0, "snapshot", snapshot)
    relation.insert(1, "redshift", redshift)
    relation["selected_galaxies"] = selected_total
    relation["dust_positive_galaxies"] = dust_positive_total
    relation["dust_positive_fraction"] = (
        dust_positive_total / selected_total if selected_total else np.nan
    )
    summary_tables.append(relation)

    if not relation.empty:
        x = relation["mass_median"].to_numpy()
        median = relation["dust_median"].to_numpy()
        p16 = relation["dust_p16"].to_numpy()
        p84 = relation["dust_p84"].to_numpy()
        ax.fill_between(
            x, p16, p84, color="0.25", alpha=0.18, linewidth=0,
            zorder=2,
        )
        ax.plot(
            x, median, "o-", color="black", linewidth=1.6,
            markersize=4.0, markerfacecolor="black", markeredgecolor="white",
            markeredgewidth=0.55, zorder=3,
        )

    positive_fraction = (
        dust_positive_total / selected_total if selected_total else np.nan
    )
    ax.text(
        0.05, 0.95, rf"({panel_label}) $z={redshift:.2f}$",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
    )
    ax.text(
        0.95, 0.06, rf"$f(M_{{\rm dust}}>0)={positive_fraction:.2f}$",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
        color="0.30",
    )
    ax.minorticks_on()
    ax.tick_params(which="major", length=4.0, width=0.8)
    ax.tick_params(which="minor", length=2.2, width=0.6)

for ax in axes[-1, :]:
    ax.set_xlabel(r"Stellar mass, $\log_{10}(M_\star/{\rm M}_\odot)$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"Dust mass, $\log_{10}(M_{\rm dust}/{\rm M}_\odot)$")

# Common robust limits prevent a few numerical outliers from compressing all
# six relations.  A small padding keeps points away from the panel borders.
mass_padding = 0.04 * (mass_max - mass_min)
dust_padding = 0.06 * (dust_max - dust_min)
for ax in axes.flat:
    ax.set_xlim(mass_min - mass_padding, mass_max + mass_padding)
    ax.set_ylim(dust_min - dust_padding, dust_max + dust_padding)

legend_handles = [
    Line2D(
        [0], [0], marker="o", linestyle="none", markersize=4,
        color=BOXES[box]["colour"], alpha=0.65, label=box,
    )
    for box in BOXES
]
legend_handles.extend([
    Line2D(
        [0], [0], marker="o", color="black", linewidth=1.6,
        markersize=4, label="Combined median",
    ),
    Line2D(
        [0], [0], color="0.45", linewidth=5, alpha=0.25,
        label="16th--84th percentile",
    ),
])
fig.legend(
    handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.03),
    ncol=4, frameon=False, handlelength=2.1, columnspacing=1.2,
)

summary = pd.concat(summary_tables, ignore_index=True)
summary.to_csv(OUTPUT_STEM.with_suffix(".csv"), index=False)
fig.savefig(OUTPUT_STEM.with_suffix(".png"), bbox_inches="tight")
fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)

print(f"\nSaved {OUTPUT_STEM.with_suffix('.png')}")
print(f"Saved {OUTPUT_STEM.with_suffix('.pdf')}")
print(f"Saved {OUTPUT_STEM.with_suffix('.csv')}")
