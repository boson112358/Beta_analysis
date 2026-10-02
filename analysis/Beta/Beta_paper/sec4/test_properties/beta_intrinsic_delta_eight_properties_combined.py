"""Combined intrinsic-beta and delta-beta figure: eight rows, two columns.

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

# Default follows the requested M_dust / Z_gas expression exactly.
# Z_gas is an absolute metal mass fraction, NOT metallicity in solar units.
# This expression has mass units and is not a dimensionless dust-to-metal ratio.
# Alternatives:
#   "dust_to_gas_metals": M_dust / (Z_gas * M_gas)
#   "dust_to_total_metals": M_dust / (Z_gas * M_gas + M_dust)
# The last option assumes Z_gas excludes metals locked in dust; verify your
# simulation's metallicity convention before interpreting either alternative.
ratio_mode = "dust_to_gas_metals"
stellar_age_key = "mass_weighted"
gas_metallicity_key = "mass_weighted"
ratio_labels = {
    "dust_mass_over_Zgas": r"Dust mass / gas metallicity $\log_{10}[(M_{\rm dust}/Z_{\rm gas})/M_\odot]$",
    "dust_to_gas_metals": r"Dust / gas metal mass $\log_{10}[M_{\rm dust}/(Z_{\rm gas}M_{\rm gas})]$",
    "dust_to_total_metals": r"Dust / total metal mass $\log_{10}[M_{\rm dust}/(Z_{\rm gas}M_{\rm gas}+M_{\rm dust})]$",
}
if ratio_mode not in ratio_labels:
    raise ValueError(f"Unknown ratio_mode: {ratio_mode}")

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
# Eight physical quantities to plot
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
    "dust_metal_ratio": {
        "label": ratio_labels[ratio_mode],
        "take_log": True,
    },
    "gas_metallicity": {
        "label": r"Gas metallicity $Z_{\rm gas}/Z_\odot$",
        "take_log": False,
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
    gas_metallicity = optional_values("metallicities", gas_metallicity_key, "Zsun")
    gas_Z_fraction = optional_values("metallicities", gas_metallicity_key, "dimensionless")
    if ratio_mode == "dust_mass_over_Zgas":
        denominator = gas_Z_fraction.copy()
    else:
        gas_mass = optional_values("masses", "gas", "Msun")
        denominator = gas_Z_fraction * gas_mass
        if ratio_mode == "dust_to_total_metals":
            denominator = denominator + dust_mass
    dust_metal_ratio = np.full(dust_mass.shape, np.nan)
    valid_ratio = (np.isfinite(denominator) & (denominator > 0)
                   & np.isfinite(dust_mass) & (dust_mass >= 0)
                   & np.isfinite(gas_Z_fraction) & (gas_Z_fraction >= 0))
    np.divide(dust_mass, denominator, out=dust_metal_ratio, where=valid_ratio)

    return {
        "stellar_age": stellar_age,
        "dust_mass": dust_mass,
        "dust_metal_ratio": dust_metal_ratio,
        "gas_metallicity": gas_metallicity,
        "stellar_mass": stellar_mass,
        "ssfr": ssfr,
        "stellar_metallicity": stellar_metallicity,
        "Av": Av,
    }


# One property per row, with intrinsic beta and delta beta side by side.
# Share y only within each quantity; the two quantities have different offsets.
fig, panel_axes = plt.subplots(8, 2, figsize=(7.1, 18.0), sharex="row", sharey="col")
figures = {
    "intrinsic": (fig, panel_axes[:, 0]),
    "delta": (fig, panel_axes[:, 1]),
}

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

            if property_name in ("stellar_age", "gas_metallicity"):
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

# Format paired panels with identical x ranges and independent y scales.
fixed_xlim = {
    "stellar_mass": (6.7, 10.15),
    "ssfr": (-9.55, -7.40),
    "stellar_metallicity": (0.05, 0.65),
    "Av": (0.0, 3.2),
}
for row, (property_name, settings) in enumerate(property_settings.items()):
    for col, quantity in enumerate(("intrinsic", "delta")):
        ax = panel_axes[row, col]
        ax.set_xlabel(settings["label"])
        ax.set_ylabel(r"$\beta_{\rm int}$" if col == 0 else r"$\Delta\beta$")
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=True)
        ax.minorticks_on()
        ax.tick_params(which="major", length=4.0, width=0.8)
        ax.tick_params(which="minor", length=2.2, width=0.6)
        ax.text(0.04, 0.94, f"({chr(97 + 2 * row + col)})",
                transform=ax.transAxes, ha="left", va="top", fontsize=9)
        if quantity == "delta":
            ax.axhline(0.0, color="0.65", linewidth=0.7, linestyle=":", zorder=0)
        if property_name in fixed_xlim:
            ax.set_xlim(*fixed_xlim[property_name])

panel_axes[0, 0].set_title(r"Intrinsic UV slope $\beta_{\rm int}$", pad=12)
panel_axes[0, 1].set_title(r"Dust reddening $\Delta\beta=\beta_{\rm dust}-\beta_{\rm int}$", pad=12)
# Gather the redshift legend from all panels in case the first has missing data.
legend_entries = {}
for ax in panel_axes.flat:
    handles, labels = ax.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        legend_entries.setdefault(label, handle)
fig.legend(list(legend_entries.values()), list(legend_entries),
           frameon=False, ncol=3, loc="upper center",
           bbox_to_anchor=(0.5, 0.995), handlelength=2.6,
           handletextpad=0.6, columnspacing=1.4)
fig.subplots_adjust(left=0.10, right=0.985, bottom=0.035, top=0.96,
                    wspace=0.28, hspace=0.52)
output_stem = f"BetaIntrinsic_and_DeltaBeta_EightProperties_{dust_law.title()}_RedshiftBins"
fig.savefig(output_stem + ".png", dpi=300, bbox_inches="tight")
fig.savefig(output_stem + ".pdf", bbox_inches="tight")
print(f"Saved {output_stem}.png and .pdf")
plt.show()
