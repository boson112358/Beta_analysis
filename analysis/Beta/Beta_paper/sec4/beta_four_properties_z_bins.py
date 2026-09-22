import numpy as np
import caesar
import matplotlib.pyplot as plt
from pathlib import Path
from utils.beta_utils import Calbeta, bin_xy_median

# ------------------------------------------------
# MNRAS-style plotting defaults
# ------------------------------------------------
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
    "legend.fontsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "savefig.dpi": 300,
})

# ------------------------------------------------
# Dust law and redshift bins
# ------------------------------------------------
dust_law = "calzetti"

redshift_bins = [
    (r"$6 \leq z \leq 7$", ["036", "030"]),
    (r"$8 \leq z \leq 9$", ["026", "022"]),
    (r"$10 \leq z \leq 11$", ["019", "016"]),
]

# ------------------------------------------------
# File templates
# ------------------------------------------------
template_m25 = (
    "/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/FullSpectra_Fit/"
    "m25n1024/caesar_m25n1024_{}_{}.hdf5"
)

template_m50 = (
    "/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/FullSpectra_Fit/"
    "m50n1024/caesar_m50n1024_{}_{}.hdf5"
)

bands = ["i1500", "i2300", "i2800"]
wavelengths = np.array([1500, 2300, 2800])

# Observational tables are expected in the same directory as this script.
data_directory = Path(__file__).resolve().parent
stellar_mass_observation_file = data_directory / "beta_stellar_mass_binned.csv"
ssfr_observation_file = data_directory / "beta_sSFR_binned.csv"

observation_styles = {
    "Morales 2024": {
        "label": r"Morales+24 ($z\simeq9$)",
        "color": "#666666",
        "marker": "D",
    },
    "Napolitano 2026": {
        "label": r"Napolitano+26 ($z\simeq6$)",
        "color": "#A0A0A0",
        "marker": "P",
    },
}

# ------------------------------------------------
# Four physical quantities to plot
# ------------------------------------------------
property_settings = {
    "stellar_mass": {
        "label": r"Stellar mass $\log_{10}(M_\star/M_\odot)$",
        "take_log": True,
    },
    "ssfr": {
        "label": r"Specific SFR $\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})$",
        "take_log": True,
    },
    "stellar_metallicity": {
        "label": r"Stellar metallicity $Z_\star$",
        "take_log": False,
    },
    "Av": {
        "label": r"$V$-band attenuation $A_V$ (mag)",
        "take_log": False,
    },
}


def extract_properties(galaxies):
    """Extract the four requested physical quantities."""
    stellar_mass = np.array([
        g.masses["stellar"].to("Msun").value for g in galaxies
    ])

    ssfr = np.array([
        (g.sfr / g.masses["stellar"]).to("1/yr").value
        for g in galaxies
    ])

    stellar_metallicity = np.array([
        g.metallicities["stellar"] for g in galaxies
    ], dtype=float)

    Av = np.array([
        g.absmag["v"] - g.absmag_nodust["v"]
        for g in galaxies
    ])

    return {
        "stellar_mass": stellar_mass,
        "ssfr": ssfr,
        "stellar_metallicity": stellar_metallicity,
        "Av": Av,
    }


def load_observations(file_path):
    """Load a binned observational CSV table."""
    return np.genfromtxt(
        file_path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )


def plot_observations(ax, observations, x_column):
    """Plot observational medians and beta uncertainties on one panel."""
    handles = []
    labels = []

    for dataset, style in observation_styles.items():
        mask = observations["dataset"] == dataset
        if not np.any(mask):
            continue

        handle = ax.errorbar(
            observations[x_column][mask],
            observations["beta"][mask],
            yerr=observations["beta_std"][mask],
            fmt=style["marker"],
            linestyle="none",
            color=style["color"],
            ecolor=style["color"],
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=0.7,
            markersize=4.0,
            elinewidth=0.6,
            capsize=1.2,
            capthick=0.6,
            alpha=0.55,
            zorder=1,
        )
        handles.append(handle)
        labels.append(style["label"])

    return handles, labels


# ------------------------------------------------
# Figure
# ------------------------------------------------
fig, axes = plt.subplots(
    2,
    2,
    figsize=(7.1, 6.0),
    sharey=True,
)
axes = axes.flatten()

# Muted colours plus different markers and line styles remain readable
# when printed in greyscale.
redshift_styles = [
    {"color": "#482878", "marker": "o", "linestyle": "-"},
    {"color": "#238A8D", "marker": "s", "linestyle": "--"},
    {"color": "#D8A800", "marker": "^", "linestyle": "-."},
]

# ------------------------------------------------
# Loop over redshift bins
# ------------------------------------------------
for (redshift_label, snapshots), style in zip(
    redshift_bins, redshift_styles
):
    beta_samples = []
    property_samples = {
        property_name: [] for property_name in property_settings
    }

    # Combine two snapshots and both simulation boxes in each redshift bin.
    for snap in snapshots:
        f25 = template_m25.format(snap, dust_law)
        f50 = template_m50.format(snap, dust_law)

        obj_m25 = caesar.load(f25)
        obj_m50 = caesar.load(f50)

        mags_m25 = np.array([
            [g.absmag[band] for g in obj_m25.galaxies]
            for band in bands
        ])

        mags_m50 = np.array([
            [g.absmag[band] for g in obj_m50.galaxies]
            for band in bands
        ])

        # Apply the observed M1500 cuts for the Calzetti law.
        mask_m25 = mags_m25[0] < -16
        mask_m50 = mags_m50[0] < -17.5

        beta_m25 = Calbeta(mags_m25, wavelengths)
        beta_m50 = Calbeta(mags_m50, wavelengths)

        beta_samples.extend([
            beta_m25[mask_m25],
            beta_m50[mask_m50],
        ])

        properties_m25 = extract_properties(obj_m25.galaxies)
        properties_m50 = extract_properties(obj_m50.galaxies)

        for property_name in property_settings:
            property_samples[property_name].extend([
                properties_m25[property_name][mask_m25],
                properties_m50[property_name][mask_m50],
            ])

    beta_combined = np.concatenate(beta_samples)

    # Bin and plot beta against each physical quantity.
    for ax, (property_name, settings) in zip(
        axes, property_settings.items()
    ):
        x_values = np.concatenate(property_samples[property_name])

        valid = np.isfinite(x_values) & np.isfinite(beta_combined)
        if settings["take_log"]:
            valid &= x_values > 0

        x_values = x_values[valid]
        beta_values = beta_combined[valid]

        if settings["take_log"]:
            x_values = np.log10(x_values)

        bin_centers, beta_median, beta_p16, beta_p84, _ = bin_xy_median(
            x_values=x_values,
            y_values=beta_values,
            mask_values=None,
            mask_cut=None,
            N_bins=10,
        )

        ax.errorbar(
            bin_centers,
            beta_median,
            yerr=[
                beta_median - beta_p16,
                beta_p84 - beta_median,
            ],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            capsize=1.8,
            capthick=0.8,
            elinewidth=1.0,
            linewidth=1.6,
            markersize=4.2,
            markeredgewidth=0.6,
            alpha=1.0,
            zorder=3,
            label=redshift_label,
        )

# Preserve a separate figure-level legend for the simulated redshift bins.
simulation_handles, simulation_labels = axes[0].get_legend_handles_labels()

# ------------------------------------------------
# Observational comparisons
# ------------------------------------------------
stellar_mass_observations = load_observations(
    stellar_mass_observation_file
)
ssfr_observations = load_observations(ssfr_observation_file)

mass_obs_handles, mass_obs_labels = plot_observations(
    axes[0], stellar_mass_observations, "mass"
)
ssfr_obs_handles, ssfr_obs_labels = plot_observations(
    axes[1], ssfr_observations, "ssfr"
)

# ------------------------------------------------
# Formatting
# ------------------------------------------------
panel_labels = ["(a)", "(b)", "(c)", "(d)"]

for ax, (_, settings), panel_label in zip(
    axes, property_settings.items(), panel_labels
):
    ax.set_ylim(-2.85, -1.50)
    ax.set_yticks(np.arange(-2.8, -1.5, 0.2))
    ax.set_xlabel(settings["label"])
    ax.minorticks_on()
    ax.tick_params(which="major", length=4.0, width=0.8)
    ax.tick_params(which="minor", length=2.2, width=0.6)
    # Move panel (b) slightly right to avoid its leftmost error bar.
    panel_label_x = 0.10 if panel_label == "(b)" else 0.04
    ax.text(
        panel_label_x,
        0.94,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

# Compact ranges matched to the plotted samples.
axes[0].set_xlim(6.7, 10.15)
axes[1].set_xlim(-9.55, -7.40)
axes[2].set_xlim(0.0007, 0.0081)
axes[3].set_xlim(0.0, 3.2)

# Stellar metallicity is plotted linearly.
axes[2].ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))

fig.text(
    0.018,
    0.50,
    r"UV slope $\beta$",
    ha="center",
    va="center",
    rotation="vertical",
    fontsize=10,
)

fig.legend(
    simulation_handles,
    simulation_labels,
    frameon=False,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.985),
    handlelength=2.6,
    handletextpad=0.6,
    columnspacing=1.4,
)

# A second figure-level legend identifies the observational datasets used
# in both upper panels without covering any plotted data.
fig.legend(
    mass_obs_handles,
    mass_obs_labels,
    frameon=False,
    fontsize=7,
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.945),
    handlelength=1.2,
    handletextpad=0.5,
    columnspacing=1.5,
)

fig.subplots_adjust(
    left=0.10,
    right=0.985,
    bottom=0.09,
    top=0.89,
    wspace=0.10,
    hspace=0.22,
)

fig.savefig(
    "Beta_vs_FourPhysicalProperties_Calzetti_RedshiftBins.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    "Beta_vs_FourPhysicalProperties_Calzetti_RedshiftBins.pdf",
    bbox_inches="tight",
)

plt.show()
