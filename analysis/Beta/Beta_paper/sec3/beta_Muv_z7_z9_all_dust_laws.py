import matplotlib.pyplot as plt
import numpy as np
import caesar
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path

from utils.beta_utils import Calbeta, bin_xy_median


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = "/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/FullSpectra_Fit"

BOXES = {
    "m25n1024": {"magnitude_cut": -16.0},
    "m50n1024": {"magnitude_cut": -17.5},
}

# Representative later and earlier EoR snapshots, arranged vertically.
PANELS = [
    {"label": r"$z=7$", "z_group": "z7", "snapshots": ["030"]},
    {"label": r"$z=9$", "z_group": "z9", "snapshots": ["022"]},
]

# The catalogue filenames are assumed to end in
# ``_calzetti.hdf5``, ``_lmc.hdf5``, ``_smc.hdf5``, and ``_mw.hdf5``.
# Calzetti remains the fiducial model and is therefore drawn slightly more
# prominently than the other attenuation prescriptions.
DUST_LAWS = {
    "calzetti": {
        "label": "Calzetti",
        "colour": "#1B7837",
        "linestyle": "-",
        "linewidth": 1.7,
        "marker": "o",
    },
    "lmc": {
        "label": "LMC",
        "colour": "#0072B2",
        "linestyle": "--",
        "linewidth": 1.45,
        "marker": None,
    },
    "smc": {
        "label": "SMC",
        "colour": "#D55E00",
        "linestyle": "-.",
        "linewidth": 1.45,
        "marker": None,
    },
    "mw": {
        "label": "MW",
        "colour": "#7B3294",
        "linestyle": ":",
        "linewidth": 1.6,
        "marker": None,
    },
}

FIDUCIAL_DUST_LAW = "calzetti"
BANDS = ["i1500", "i2300", "i2800"]
WAVELENGTHS = np.array([1500.0, 2300.0, 2800.0])

# Keep the CSV beside this script.  Using __file__ makes the path independent
# of the directory from which the script is submitted on COSMA.
OBSERVATION_FILE = Path(__file__).resolve().parent / "beta_MUV_binned_by_redshift.csv"

# MNRAS-like plotting defaults.  STIX gives a clean journal-style serif font
# without requiring a local LaTeX installation.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

NODUST_COLOUR = "#4D4D4D"

# Dataset-specific styling is held fixed between redshift panels.  The colours
# and marker shapes remain distinguishable in print and for common forms of
# colour-vision deficiency.
OBSERVATION_STYLES = {
    "Bouwens 2014": {"marker": "o", "colour": "#D55E00"},
    "Cullen 2024": {"marker": "s", "colour": "#7B3294"},
    "Morales 2024": {"marker": "D", "colour": "#0072B2"},
    "Napolitano 2026": {"marker": "^", "colour": "#CC79A7"},
    "Whitler 2025": {"marker": "p", "colour": "#E69F00"},
    "Topping 2024": {"marker": "X", "colour": "#7A8A99", "alpha": 0.65, "markersize": 3.2},
}


def catalogue_path(box, snapshot, dust_law):
    """Return the catalogue path for one box, snapshot, and dust law."""
    return (
        f"{DATA_DIR}/{box}/"
        f"caesar_{box}_{snapshot}_{dust_law}.hdf5"
    )


def get_magnitudes(catalogue, magnitude_field):
    """Extract the three UV absolute magnitudes from a CAESAR catalogue."""
    return np.array(
        [
            [getattr(galaxy, magnitude_field)[band] for galaxy in catalogue.galaxies]
            for band in BANDS
        ]
    )


def load_panel_samples(snapshots):
    """Load all dust laws and combine both boxes for one redshift panel."""
    dust_samples = {dust_law: [] for dust_law in DUST_LAWS}
    nodust_samples = []

    for snapshot in snapshots:
        for box, settings in BOXES.items():
            cut = settings["magnitude_cut"]

            for dust_law in DUST_LAWS:
                catalogue = caesar.load(
                    catalogue_path(box, snapshot, dust_law)
                )
                magnitudes = get_magnitudes(catalogue, "absmag")
                dust_samples[dust_law].append(
                    magnitudes[:, magnitudes[0] < cut]
                )

                # The intrinsic magnitudes should be identical in every dust
                # catalogue, so load them once from the fiducial catalogue.
                if dust_law == FIDUCIAL_DUST_LAW:
                    nodust_magnitudes = get_magnitudes(
                        catalogue, "absmag_nodust"
                    )
                    nodust_samples.append(
                        nodust_magnitudes[:, nodust_magnitudes[0] < cut]
                    )

    return (
        {
            dust_law: np.concatenate(samples, axis=1)
            for dust_law, samples in dust_samples.items()
        },
        np.concatenate(nodust_samples, axis=1),
    )


def calculate_binned_beta(magnitudes):
    """Calculate beta and its median in M_UV bins."""
    beta = Calbeta(magnitudes, WAVELENGTHS)
    return bin_xy_median(
        x_values=magnitudes[0],
        y_values=beta,
        mask_values=None,
        mask_cut=None,
        N_bins=6,
        min_count=5,
    )


def load_observations(filename):
    """Read and validate the binned observational measurements."""
    observations = pd.read_csv(filename)
    required_columns = {
        "dataset", "z_group", "M_UV", "beta", "uncertainty"
    }
    missing = required_columns.difference(observations.columns)
    if missing:
        raise ValueError(
            "The observational CSV is missing columns: "
            + ", ".join(sorted(missing))
        )
    return observations


# -----------------------------------------------------------------------------
# Plot the two representative redshift panels in a vertical, single-column
# layout suitable for an MNRAS figure.  The axes share both scales so that the
# separation among the attenuation laws can be compared directly.
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(3.5, 6.1),
    sharex=True,
    sharey=True,
    gridspec_kw={"hspace": 0.06},
)
observations = load_observations(OBSERVATION_FILE)

for index, panel in enumerate(PANELS):
    ax = axes[index]
    dust_magnitudes, nodust_magnitudes = load_panel_samples(panel["snapshots"])

    nodust_centres, nodust_median, nodust_p16, nodust_p84, _ = calculate_binned_beta(
        nodust_magnitudes
    )

    # Draw all attenuation laws.  To keep the figure readable, only the
    # fiducial Calzetti relation carries a 16th--84th percentile band.
    for dust_law, style in DUST_LAWS.items():
        centres, median, p16, p84, _ = calculate_binned_beta(
            dust_magnitudes[dust_law]
        )

        if dust_law == FIDUCIAL_DUST_LAW:
            ax.fill_between(
                centres,
                p16,
                p84,
                color=style["colour"],
                alpha=0.12,
                linewidth=0,
                zorder=1,
            )

        ax.plot(
            centres,
            median,
            color=style["colour"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markersize=2.6 if style["marker"] else 0,
            markeredgewidth=0,
            zorder=3,
        )

    ax.fill_between(
        nodust_centres,
        nodust_p16,
        nodust_p84,
        color=NODUST_COLOUR,
        alpha=0.10,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        nodust_centres,
        nodust_median,
        color=NODUST_COLOUR,
        linestyle="--",
        linewidth=1.5,
        zorder=2,
    )

    # Add the observational measurements assigned to this redshift group.
    panel_observations = observations[
        observations["z_group"] == panel["z_group"]
    ]
    for dataset, style in OBSERVATION_STYLES.items():
        sample = panel_observations[
            panel_observations["dataset"] == dataset
        ]
        if sample.empty:
            continue

        # NaN uncertainties occur for bins containing a single galaxy.  A zero
        # plotting error keeps the marker visible without inventing an error.
        y_error = sample["uncertainty"].fillna(0.0).to_numpy()
        ax.errorbar(
            sample["M_UV"],
            sample["beta"],
            yerr=y_error,
            fmt=style["marker"],
            markersize=style.get("markersize", 4.0),
            markerfacecolor="white",
            markeredgecolor=style["colour"],
            markeredgewidth=0.9,
            ecolor=style["colour"],
            elinewidth=0.65,
            capsize=1.3,
            capthick=0.65,
            linestyle="none",
            alpha=style.get("alpha", 0.95),
            zorder=4,
        )

    # Panel annotation is quieter and uses less space than a title.
    ax.text(
        0.06,
        0.91,
        panel["label"],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )

    ax.set_xlim(-23, -15)
    ax.set_ylim(-2.90, -1.20)
    ax.set_xticks(np.arange(-23, -14, 2))
    ax.set_yticks(np.arange(-2.8, -1.1, 0.4))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", direction="in", top=True, right=True)

# The shared x-axis is labelled only on the lower panel.
axes[0].tick_params(labelbottom=False)

fig.supxlabel(r"Rest-frame UV absolute magnitude, $M_{1500}$", y=0.035)
fig.supylabel(r"UV continuum slope, $\beta$", x=0.015)

# One shared legend is cleaner and leaves the panels free for observational
# points that may be added later.
legend_handles = [
    Line2D(
        [0],
        [0],
        color=style["colour"],
        lw=style["linewidth"],
        linestyle=style["linestyle"],
        marker=style["marker"],
        markersize=3 if style["marker"] else 0,
        label=style["label"],
    )
    for style in DUST_LAWS.values()
]

legend_handles.append(
    Line2D(
        [0], [0], color=NODUST_COLOUR, lw=1.5,
        linestyle="--", label="No dust"
    )
)

for dataset, style in OBSERVATION_STYLES.items():
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=style["colour"],
            marker=style["marker"],
            markerfacecolor="white",
            markeredgewidth=0.9,
            linestyle="none",
            markersize=style.get("markersize", 4.5),
            label=dataset,
        )
    )

fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.965),
    ncol=2,
    frameon=False,
    fontsize=7.5,
    handlelength=2.2,
    handletextpad=0.45,
    columnspacing=1.0,
    labelspacing=0.22,
)

# Raise the panels towards the compact legend while retaining a small gap
# above the z=7 axes.
fig.subplots_adjust(left=0.18, right=0.98, bottom=0.11, top=0.79)

fig.savefig("Beta_vs_Muv_z7_z9_all_dust_laws_MNRAS.pdf")
fig.savefig("Beta_vs_Muv_z7_z9_all_dust_laws_MNRAS.png")
plt.close(fig)
