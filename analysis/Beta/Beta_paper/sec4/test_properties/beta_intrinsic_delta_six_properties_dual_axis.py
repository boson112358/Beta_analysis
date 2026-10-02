"""Six dual-y-axis panels comparing intrinsic beta and dust reddening.

Axis limits vertically separate the curves; no offsets are added to the data.

Delta beta = attenuated beta - intrinsic beta, evaluated per galaxy.
Both slopes use the same three-band Calbeta estimator and selected galaxies.

CAESAR field definitions: https://caesar.readthedocs.io/en/latest/catalog.html
Run alongside utils/beta_utils.py with access to the CAESAR catalogues.
Saves one combined PNG/PDF figure; each row compares the same property.
Observed total-beta tables are intentionally not overlaid.
No CSV output is written.
"""
import warnings
import numpy as np
import caesar
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
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

stellar_age_key = "mass_weighted"

# Fractions of panel height occupied by each quantity's full percentile range.
# Adjust these to change vertical separation without changing any data values.
# Slight overlap is allowed; intrinsic curves sit lower, delta beta higher.
vertical_bands = {"intrinsic": (0.08, 0.54), "delta": (0.46, 0.92)}

redshift_bins = [
    (r"$z \approx 6,\,7$", ["036", "030"]),
    (r"$z \approx 8,\,9$", ["026", "022"]),
    (r"$z \approx 10,\,11$", ["019", "016"]),
]

# ------------------------------------------------
# File templates
# ------------------------------------------------
template_m25 = (
    "/home/zxiang/simba-eor/my_dustext_output_updated/"
    "m25n1024/caesar_m25n1024_{}_{}.hdf5"
)

template_m50 = (
    "/home/zxiang/simba-eor/my_dustext_output_updated/"
    "m50n1024/caesar_m50n1024_{}_{}.hdf5"
)

bands = ["i1500", "i2300", "i2800"]
wavelengths = np.array([1500, 2300, 2800])

# Exclude the single extreme metallicity value that otherwise stretches the
# equal-width bins in panel (c). This cut is applied only to the z~6--7
# stellar-metallicity relation and does not affect the other panels.
stellar_metallicity_max_z6_z7 = 0.7

# ------------------------------------------------
# Six physical quantities to plot
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
        "label": r"Stellar metallicity $Z_\star/Z_\odot$",
        "take_log": False,
    },
    "Av": {
        "label": r"$V$-band attenuation $A_V$ (mag)",
        "take_log": False,
    },
    "stellar_age": {
        "label": r"Mass-weighted stellar age (Myr)",
        "take_log": False,
    },
    "dust_mass": {
        "label": r"Dust mass $\log_{10}(M_{\rm dust}/M_\odot)$",
        "take_log": True,
    },

}


def extract_properties(galaxies):
    """Extract properties; missing new fields affect only their own panels."""
    stellar_mass = np.array([
        g.masses["stellar"].to("Msun").value for g in galaxies
    ])

    ssfr = np.array([
        (g.sfr / g.masses["stellar"]).to("1/yr").value
        for g in galaxies
    ])

    stellar_metallicity = np.array([
        g.metallicities["stellar"].to("Zsun").value
        for g in galaxies
    ], dtype=float)
    
    Av = np.array([
        g.absmag["v"] - g.absmag_nodust["v"]
        for g in galaxies
    ])

    def optional_values(group, key, unit):
        values = []
        missing = 0
        for g in galaxies:
            try:
                value = getattr(g, group)[key]
            except (AttributeError, KeyError):
                values.append(np.nan)
                missing += 1
                continue
            # Require explicit units; do not silently guess ages or Z units.
            values.append(float(value.to(unit).value))
        if missing:
            warnings.warn(f"{group}[{key!r}] missing for {missing} galaxies; "
                          "excluded from affected panels only.")
        return np.asarray(values, dtype=float)

    stellar_age = optional_values("ages", stellar_age_key, "Myr")
    dust_mass = optional_values("masses", "dust", "Msun")
    return {
        "stellar_age": stellar_age,
        "dust_mass": dust_mass,
        "stellar_mass": stellar_mass,
        "ssfr": ssfr,
        "stellar_metallicity": stellar_metallicity,
        "Av": Av,
    }


fig, panel_axes = plt.subplots(3, 2, figsize=(7.5, 9.4))
left_axes = panel_axes.flatten()
right_axes = np.array([ax.twinx() for ax in left_axes])
figures = {
    "intrinsic": (fig, left_axes),
    "delta": (fig, right_axes),
}
y_extents = {"intrinsic": [], "delta": []}

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
for redshift_index, ((redshift_label, snapshots), style) in enumerate(
    zip(redshift_bins, redshift_styles)
):
    beta_samples = {"intrinsic": [], "delta": []}
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

        # Fit intrinsic magnitudes with exactly the same estimator as total beta.
        for galaxies, beta_dust, selection in (
            (obj_m25.galaxies, beta_m25, mask_m25),
            (obj_m50.galaxies, beta_m50, mask_m50),
        ):
            intrinsic_mags = np.array([
                [g.absmag_nodust[band] for g in galaxies] for band in bands
            ])
            beta_intrinsic = np.asarray(Calbeta(intrinsic_mags, wavelengths))
            beta_dust = np.asarray(beta_dust)
            # Matched finite samples allow comparison of the two figures.
            paired = np.isfinite(beta_intrinsic) & np.isfinite(beta_dust)
            beta_intrinsic = np.where(paired, beta_intrinsic, np.nan)
            delta_beta = np.where(paired, beta_dust - beta_intrinsic, np.nan)
            beta_samples["intrinsic"].append(beta_intrinsic[selection])
            beta_samples["delta"].append(delta_beta[selection])

        properties_m25 = extract_properties(obj_m25.galaxies)
        properties_m50 = extract_properties(obj_m50.galaxies)

        for property_name in property_settings:
            property_samples[property_name].extend([
                properties_m25[property_name][mask_m25],
                properties_m50[property_name][mask_m50],
            ])

    for quantity, (fig, axes) in figures.items():
        beta_combined = np.concatenate(beta_samples[quantity])

        # Bin and plot beta against each physical quantity.
        for ax, (property_name, settings) in zip(
            axes, property_settings.items()
        ):
            x_values = np.concatenate(property_samples[property_name])

            valid = np.isfinite(x_values) & np.isfinite(beta_combined)
            if settings["take_log"]:
                valid &= x_values > 0

            if redshift_index == 0 and property_name == "stellar_metallicity":
                metallicity_outlier = (
                    np.isfinite(x_values)
                    & (x_values >= stellar_metallicity_max_z6_z7)
                )
                print(
                    f"{redshift_label}, panel (c): excluded "
                    f"{np.count_nonzero(metallicity_outlier)} galaxy/galaxies "
                    f"with Z_star/Z_sun >= {stellar_metallicity_max_z6_z7}"
                )
                valid &= x_values < stellar_metallicity_max_z6_z7

            if property_name == "stellar_age":
                valid &= x_values >= 0
            print(f"{redshift_label}, {property_name}: {np.count_nonzero(valid)} valid galaxies")
            x_values = x_values[valid]
            beta_values = beta_combined[valid]

            if settings["take_log"]:
                x_values = np.log10(x_values)

            if len(x_values) < 2 or np.ptp(x_values) == 0:
                warnings.warn(f"Skipping {property_name} for {redshift_label}: insufficient range/data.")
                continue

            bin_centers, beta_median, beta_p16, beta_p84, _ = bin_xy_median(
                x_values=x_values,
                y_values=beta_values,
                mask_values=None,
                mask_cut=None,
                N_bins=10,
            )

            y_extents[quantity].extend([
                np.asarray(beta_p16).ravel(), np.asarray(beta_p84).ravel()
            ])
            ax.errorbar(
                bin_centers,
                beta_median,
                yerr=[
                    beta_median - beta_p16,
                    beta_p84 - beta_median,
                ],
                color=style["color"],
                marker=style["marker"],
                linestyle="-" if quantity == "intrinsic" else "--",
                markerfacecolor=style["color"] if quantity == "intrinsic" else "white",
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

# Use one consistent scale per quantity across all six panels.
# Map the plotted percentile envelope to the chosen fraction of panel height.
for quantity, (_, axes) in figures.items():
    values = np.concatenate(y_extents[quantity]) if y_extents[quantity] else np.array([])
    values = values[np.isfinite(values)]
    if values.size:
        lo, hi = values.min(), values.max()
    else:
        lo, hi = (-2.8, -2.0) if quantity == "intrinsic" else (0.0, 0.8)
    if hi <= lo:
        lo, hi = lo - 0.1, hi + 0.1
    bottom, top = vertical_bands[quantity]
    if not 0 <= bottom < top <= 1:
        raise ValueError(f"Invalid vertical band for {quantity}")
    span = (hi - lo) / (top - bottom)
    limits = (lo - bottom * span, lo + (1 - bottom) * span)
    ticks = MaxNLocator(nbins=4).tick_values(lo, hi)
    ticks = ticks[(ticks >= limits[0]) & (ticks <= limits[1])]
    for ax in axes:
        ax.set_ylim(*limits)
        ax.set_yticks(ticks)

fixed_xlim = {
    "stellar_mass": (6.7, 10.15),
    "ssfr": (-9.55, -7.40),
    "stellar_metallicity": (0.05, 0.65),
    "Av": (0.0, 3.2),
}
for index, (property_name, settings) in enumerate(property_settings.items()):
    left, right = left_axes[index], right_axes[index]
    left.set_xlabel(settings["label"])
    left.set_ylabel("Intrinsic β")
    right.set_ylabel("Δβ = β_dust − β_int")
    for ax in (left, right):
        ax.minorticks_on()
        ax.tick_params(which="major", length=4.0, width=0.8)
        ax.tick_params(which="minor", length=2.2, width=0.6)
    left.tick_params(axis="y", right=False)
    right.tick_params(axis="y", left=False, right=True)
    left.text(0.04, 0.96, f"({chr(97 + index)})", transform=left.transAxes,
              ha="left", va="top", fontsize=9)
    if property_name in fixed_xlim:
        left.set_xlim(*fixed_xlim[property_name])

redshift_handles = [
    Line2D([], [], color=style["color"], marker=style["marker"], linestyle="none",
           label=label, markersize=4.5)
    for (label, _), style in zip(redshift_bins, redshift_styles)
]
quantity_handles = [
    Line2D([], [], color="0.25", marker="o", linestyle="-",
           label=r"Intrinsic $β$ (left axis)", markersize=4),
    Line2D([], [], color="0.25", marker="o", markerfacecolor="white", linestyle="--",
           label=r"$\Deltaβ$ (right axis)", markersize=4),
]
fig.legend(handles=redshift_handles, frameon=False, ncol=3,
           loc="upper center", bbox_to_anchor=(0.5, 0.995))
fig.legend(handles=quantity_handles, frameon=False, ncol=2,
           loc="upper center", bbox_to_anchor=(0.5, 0.970))
fig.subplots_adjust(left=0.09, right=0.90, bottom=0.065, top=0.925,
                    wspace=0.58, hspace=0.36)
output_stem = f"BetaIntrinsic_and_DeltaBeta_SixProperties_DualAxis_{dust_law.title()}"
fig.savefig(output_stem + ".png", dpi=300, bbox_inches="tight")
fig.savefig(output_stem + ".pdf", bbox_inches="tight")
print(f"Saved {output_stem}.png and .pdf")
plt.show()
