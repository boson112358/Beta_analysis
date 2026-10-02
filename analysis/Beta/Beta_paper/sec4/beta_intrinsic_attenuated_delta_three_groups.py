"""Three paired-property figures: intrinsic/attenuated beta above, delta below.

Each upper/lower pair shares its x axis. No data offsets are applied.

Delta beta = attenuated beta - intrinsic beta, evaluated per galaxy.
Both slopes use the same three-band Calbeta estimator and selected galaxies.

CAESAR field definitions: https://caesar.readthedocs.io/en/latest/catalog.html
Run alongside utils/beta_utils.py with access to the CAESAR catalogues.
Saves three figures, each as PNG and PDF, loading each catalogue only once.
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

# Keep median curves legible while de-emphasising percentile error bars.
errorbar_alpha = 0.25
errorbar_linewidth = 0.65
intrinsic_errorbar_linestyle = (0, (2.0, 1.5))
main_to_delta_height = (3.0, 1.0)
intrinsic_linestyle = (0, (4, 2.5))
# None uses the full plotted percentile range across ALL three figures.
# Set tuples here if fixed publication limits are preferred.
main_ylim = None
delta_ylim = None

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


plot_groups = [
    ("StellarMass_sSFR", ("stellar_mass", "ssfr")),
    ("StellarMetallicity_StellarAge", ("stellar_metallicity", "stellar_age")),
    ("Av_DustMass", ("Av", "dust_mass")),
]
group_figures = []
axes_by_property = {}
for group_name, properties in plot_groups:
    fig = plt.figure(figsize=(7.5, 4.5))
    outer = fig.add_gridspec(1, 2, left=0.10, right=0.98, bottom=0.14,
                             top=0.80, wspace=0.10)
    for column, property_name in enumerate(properties):
        inner = outer[column].subgridspec(
            2, 1, height_ratios=main_to_delta_height, hspace=0.06)
        main = fig.add_subplot(inner[0])
        delta = fig.add_subplot(inner[1], sharex=main)
        main.tick_params(axis="x", labelbottom=False)
        axes_by_property[property_name] = (main, delta)
    group_figures.append((group_name, properties, fig))
# Preserve property-to-axis mapping even though the output grouping is reordered.
main_axes = [axes_by_property[name][0] for name in property_settings]
delta_axes = [axes_by_property[name][1] for name in property_settings]
quantity_axes = {"intrinsic": main_axes, "attenuated": main_axes, "delta": delta_axes}

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
    beta_samples = {"intrinsic": [], "attenuated": [], "delta": []}
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
            # Matched finite samples allow comparison of all three beta measures.
            paired = np.isfinite(beta_intrinsic) & np.isfinite(beta_dust)
            beta_intrinsic = np.where(paired, beta_intrinsic, np.nan)
            delta_beta = np.where(paired, beta_dust - beta_intrinsic, np.nan)
            beta_samples["attenuated"].append(np.where(paired, beta_dust, np.nan)[selection])
            beta_samples["intrinsic"].append(beta_intrinsic[selection])
            beta_samples["delta"].append(delta_beta[selection])

        properties_m25 = extract_properties(obj_m25.galaxies)
        properties_m50 = extract_properties(obj_m50.galaxies)

        for property_name in property_settings:
            property_samples[property_name].extend([
                properties_m25[property_name][mask_m25],
                properties_m50[property_name][mask_m50],
            ])

    for quantity, axes in quantity_axes.items():
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
                    f"{redshift_label}, stellar metallicity: excluded "
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

            # Apply transparency to error-bar artists only, not median curves.
            _, caps, bars = ax.errorbar(
                bin_centers, beta_median,
                yerr=[beta_median - beta_p16, beta_p84 - beta_median],
                fmt="none", ecolor=style["color"],
                elinewidth=errorbar_linewidth,
                capsize=0 if quantity == "intrinsic" else 1.8, capthick=0.6,
                zorder=1,
            )
            for bar in bars:
                bar.set_linestyle(intrinsic_errorbar_linestyle if quantity == "intrinsic" else "solid")
            for artist in (*caps, *bars):
                artist.set_alpha(errorbar_alpha)
            ax.plot(
                bin_centers, beta_median, color=style["color"],
                marker=style["marker"],
                linestyle=intrinsic_linestyle if quantity == "intrinsic" else "-",
                markerfacecolor="white" if quantity == "intrinsic" else style["color"],
                linewidth=1.25, markersize=3.2, markeredgewidth=0.65,
                alpha=0.95, zorder=3, label=redshift_label,
            )

# Format panels first; common y limits are applied to all figures below.
fixed_xlim = {
    "stellar_mass": (6.7, 10.15),
    "ssfr": (-9.55, -7.40),
    "stellar_metallicity": (0.05, 0.65),
    "Av": (0.0, 3.2),
}
for index, (property_name, settings) in enumerate(property_settings.items()):
    main, delta = main_axes[index], delta_axes[index]
    main.set_ylabel("UV slope β")
    delta.set_ylabel("Δβ")
    delta.set_xlabel(settings["label"])
    delta.axhline(0, color="0.65", linestyle=":", linewidth=0.65, zorder=0)
    delta.yaxis.set_major_locator(MaxNLocator(nbins=3))
    main.yaxis.set_major_locator(MaxNLocator(nbins=5))
    for ax in (main, delta):
        ax.minorticks_on()
        ax.tick_params(which="major", length=3.5, width=0.8)
        ax.tick_params(which="minor", length=1.8, width=0.6)
        ax.margins(y=0.12)
    main.text(0.04, 0.95, f"({chr(97 + next(properties.index(property_name) for _, properties in plot_groups if property_name in properties))})", transform=main.transAxes,
              ha="left", va="top", fontsize=9)
    if property_name in fixed_xlim:
        main.set_xlim(*fixed_xlim[property_name])

def common_y_limits(axes, override=None):
    """Include error bars (dataLim), then pad the global finite range."""
    if override is not None:
        return override
    bounds = np.array([ax.dataLim.intervaly for ax in axes])
    bounds = bounds[np.isfinite(bounds)]
    if bounds.size == 0:
        return (-1.0, 1.0)
    lo, hi = bounds.min(), bounds.max()
    pad = 0.08 * max(hi - lo, 0.1)
    return lo - pad, hi + pad

shared_main_limits = common_y_limits(main_axes, main_ylim)
shared_delta_limits = common_y_limits(delta_axes, delta_ylim)
for ax in main_axes:
    ax.set_ylim(*shared_main_limits)
for ax in delta_axes:
    ax.set_ylim(*shared_delta_limits)
# Identical scales allow the right panels to omit repeated y-axis labels.
for _, properties, _ in group_figures:
    for ax in axes_by_property[properties[1]]:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)

redshift_handles = [
    Line2D([], [], color=style["color"], marker=style["marker"], linestyle="none",
           label=label, markersize=4.5)
    for (label, _), style in zip(redshift_bins, redshift_styles)
]
# NaN-position errorbar artists provide legend keys without visible data.
quantity_handles = []
for quantity, label in (("intrinsic", "Intrinsic β"), ("attenuated", "Attenuated β")):
    handle = main_axes[0].errorbar(
        [np.nan], [np.nan], yerr=[1.0], color="0.20", linewidth=1.6,
        linestyle=intrinsic_linestyle if quantity == "intrinsic" else "-",
        capsize=0 if quantity == "intrinsic" else 1.8,
        elinewidth=0.8, capthick=0.6, label=label,
    )
    for bar in handle.lines[2]:
        bar.set_linestyle(intrinsic_errorbar_linestyle if quantity == "intrinsic" else "solid")
    quantity_handles.append(handle)
for group_name, properties, fig in group_figures:
    fig.legend(handles=redshift_handles, frameon=False, ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, 0.995))
    fig.legend(handles=quantity_handles, frameon=False, ncol=2,
               handlelength=4.5, handletextpad=0.8, columnspacing=2.0,
               loc="upper center", bbox_to_anchor=(0.5, 0.945))
    fig.text(0.5, 0.865, "Lower panels: Δβ = attenuated β − intrinsic β",
             ha="center", fontsize=8)
    output_stem = f"BetaIntrinsic_Attenuated_Delta_{group_name}_{dust_law.title()}"
    fig.savefig(output_stem + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(output_stem + ".pdf", bbox_inches="tight")
    print(f"Saved {output_stem}.png and .pdf")
plt.show()
