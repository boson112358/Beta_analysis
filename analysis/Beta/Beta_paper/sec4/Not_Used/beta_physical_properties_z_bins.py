import numpy as np
import caesar
import matplotlib.pyplot as plt
from utils.beta_utils import *

# ------------------------------------------------
# Dust law
# ------------------------------------------------
dust_law = "calzetti"

# ------------------------------------------------
# Redshift bins and their snapshots
# ------------------------------------------------
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

# ------------------------------------------------
# Physical quantities to plot
# ------------------------------------------------
property_settings = {
    "stellar_mass": {
        "label": r"$\log_{10}(M_\star/M_\odot)$",
        "take_log": True,
    },
    "sfr": {
        "label": r"$\log_{10}(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})$",
        "take_log": True,
    },
    "ssfr": {
        "label": r"$\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})$",
        "take_log": True,
    },
    "gas_metallicity": {
        "label": r"$Z_{\rm gas}$",
        "take_log": False,
    },
    "stellar_metallicity": {
        "label": r"$Z_\star$",
        "take_log": False,
    },
    "Av": {
        "label": r"$A_V$ (mag)",
        "take_log": False,
    },
    "dust_mass": {
        "label": r"$\log_{10}(M_{\rm dust}/M_\odot)$",
        "take_log": True,
    },
    "stellar_age": {
        "label": r"Mass-weighted stellar age (Gyr)",
        "take_log": False,
    },
}


def extract_properties(galaxies):
    """Return the eight physical quantities for a CAESAR galaxy list."""
    stellar_mass = np.array([
        g.masses["stellar"].to("Msun").value for g in galaxies
    ])

    sfr = np.array([
        g.sfr.to("Msun/yr").value for g in galaxies
    ])

    ssfr = np.array([
        (g.sfr / g.masses["stellar"]).to("1/yr").value
        for g in galaxies
    ])

    gas_metallicity = np.array([
        g.metallicities["mass_weighted"] for g in galaxies
    ], dtype=float)

    stellar_metallicity = np.array([
        g.metallicities["stellar"] for g in galaxies
    ], dtype=float)

    Av = np.array([
        g.absmag["v"] - g.absmag_nodust["v"]
        for g in galaxies
    ])

    dust_mass = np.array([
        g.masses["dust"].to("Msun").value for g in galaxies
    ])

    stellar_age = np.array([
        g.ages["mass_weighted"].to("Gyr").value for g in galaxies
    ])

    return {
        "stellar_mass": stellar_mass,
        "sfr": sfr,
        "ssfr": ssfr,
        "gas_metallicity": gas_metallicity,
        "stellar_metallicity": stellar_metallicity,
        "Av": Av,
        "dust_mass": dust_mass,
        "stellar_age": stellar_age,
    }


# ------------------------------------------------
# Figure
# ------------------------------------------------
fig, axes = plt.subplots(
    4,
    2,
    figsize=(11, 14),
    sharey=True,
)
axes = axes.flatten()

# One colour for each redshift bin, ordered from low to high redshift.
redshift_colors = plt.cm.viridis(np.linspace(0.10, 0.90, len(redshift_bins)))

# ------------------------------------------------
# Loop over redshift bins
# ------------------------------------------------
for (redshift_label, snapshots), color in zip(redshift_bins, redshift_colors):

    beta_samples = []
    property_samples = {
        property_name: [] for property_name in property_settings
    }

    # Combine both snapshots and both simulation boxes within each redshift bin.
    for snap in snapshots:

        # -------------------------
        # Load Calzetti files
        # -------------------------
        f25 = template_m25.format(snap, dust_law)
        f50 = template_m50.format(snap, dust_law)

        obj = caesar.load(f25)
        obj_m50 = caesar.load(f50)

        # -------------------------
        # Calzetti-attenuated magnitudes
        # -------------------------
        mags_m25 = np.array([
            [g.absmag[band] for g in obj.galaxies]
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

        properties_m25 = extract_properties(obj.galaxies)
        properties_m50 = extract_properties(obj_m50.galaxies)

        for property_name in property_settings:
            property_samples[property_name].extend([
                properties_m25[property_name][mask_m25],
                properties_m50[property_name][mask_m50],
            ])

    beta_combined = np.concatenate(beta_samples)

    # Calculate and plot the binned relation for every physical quantity.
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
            color=color,
            marker="o",
            linestyle="-",
            capsize=2,
            linewidth=2,
            markersize=5,
            label=redshift_label,
        )

# ------------------------------------------------
# Formatting
# ------------------------------------------------
for ax, (_, settings) in zip(axes, property_settings.items()):
    ax.set_ylim(-2.5, -1.6)
    ax.set_yticks(np.arange(-2.4, -1.5, 0.2))
    ax.set_xlabel(settings["label"])
    ax.margins(x=0.05)

# Use scientific notation for the two linear metallicity axes.
axes[3].ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
axes[4].ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))

fig.supylabel(r"$\beta$")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    frameon=False,
    fontsize=10,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
)

plt.tight_layout(rect=(0.02, 0.02, 1.0, 0.96))

plt.savefig(
    "Beta_vs_PhysicalProperties_and_Age_Calzetti_RedshiftBins.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
