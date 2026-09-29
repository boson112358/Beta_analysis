import numpy as np
import caesar
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from utils.beta_utils import Calbeta


# ================================================================
# MNRAS-style plotting defaults
# ================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.linewidth": 1.0,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "savefig.dpi": 300,
})


# ================================================================
# User-adjustable choices
# ================================================================
dust_law = "calzetti"

# Quantity represented by the colour and numerical label of each large point.
# Choose: "stellar_metallicity", "av", or "ssfr".
secondary_property = "stellar_metallicity"

property_settings = {
    "stellar_metallicity": {
        "data_key": "log_stellar_metallicity",
        "symbol": r"\log_{10}(Z_\star/Z_\odot)",
        "description": "log10 stellar metallicity [Zsun]",
        "colourbar_label": r"$\log_{10}(Z_\star/Z_\odot)$",
        "file_tag": "StellarMetallicity",
    },
    "av": {
        "data_key": "av",
        "symbol": r"A_V",
        "description": "V-band attenuation A_V [mag]",
        "colourbar_label": r"$A_V$ (mag)",
        "file_tag": "Av",
    },
    "ssfr": {
        "data_key": "log_ssfr",
        "symbol": r"\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})",
        "description": "log10 sSFR [yr^-1]",
        "colourbar_label": r"$\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})$",
        "file_tag": "sSFR",
    },
}

# All six snapshots are combined into one sample.
snapshots = ["036", "030", "026", "022", "019", "016"]

# Observed M1500 completeness limits used for the two boxes.
magnitude_limit_m25 = -16.0
magnitude_limit_m50 = -17.5

# Display at most this many individual galaxies in the grey background.
# All selected galaxies are still used to calculate the four regions.
max_background_points = 30000
random_seed = 13

# Percentiles used for the plotted population spread.
lower_percentile = 16
upper_percentile = 84

# The beta split is a sloping line in each mass half. The slope is selected
# numerically so the samples above and below the line contain 50% each and
# have nearly identical median stellar masses.
balance_slope_half_width = 1.5
balance_grid_size = 101
balance_refinements = 3


# ================================================================
# Input catalogues
# ================================================================
template_m25 = (
    "/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/FullSpectra_Fit/"
    "m25n1024/caesar_m25n1024_{}_{}.hdf5"
)

template_m50 = (
    "/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/FullSpectra_Fit/"
    "m50n1024/caesar_m50n1024_{}_{}.hdf5"
)

bands = ["i1500", "i2300", "i2800"]
wavelengths = np.array([1500.0, 2300.0, 2800.0])


def load_selected_galaxies(file_name, magnitude_limit):
    """Load one catalogue and apply its observed Calzetti M1500 cut."""
    obj = caesar.load(file_name)

    magnitudes = np.array([
        [galaxy.absmag[band] for galaxy in obj.galaxies]
        for band in bands
    ])

    selected = magnitudes[0] < magnitude_limit

    stellar_mass = np.array([
        galaxy.masses["stellar"].to("Msun").value
        for galaxy in obj.galaxies
    ])[selected]

    stellar_metallicity = np.array([
        galaxy.metallicities["stellar"].to("Zsun").value
        for galaxy in obj.galaxies
    ], dtype=float)[selected]

    sfr = np.array([
        galaxy.sfr.to("Msun/yr").value
        for galaxy in obj.galaxies
    ])[selected]

    av = np.array([
        galaxy.absmag["v"] - galaxy.absmag_nodust["v"]
        for galaxy in obj.galaxies
    ])[selected]

    beta = Calbeta(magnitudes, wavelengths)[selected]

    valid = (
        np.isfinite(stellar_mass)
        & np.isfinite(beta)
        & (stellar_mass > 0)
    )

    log_stellar_metallicity = np.full(len(stellar_mass), np.nan)
    positive_metallicity = (
        np.isfinite(stellar_metallicity) & (stellar_metallicity > 0)
    )
    log_stellar_metallicity[positive_metallicity] = np.log10(
        stellar_metallicity[positive_metallicity]
    )

    log_ssfr = np.full(len(stellar_mass), np.nan)
    positive_ssfr = np.isfinite(sfr) & (sfr > 0)
    log_ssfr[positive_ssfr] = np.log10(
        sfr[positive_ssfr] / stellar_mass[positive_ssfr]
    )

    return {
        "log_stellar_mass": np.log10(stellar_mass[valid]),
        "beta": beta[valid],
        "log_stellar_metallicity": log_stellar_metallicity[valid],
        "av": av[valid],
        "log_ssfr": log_ssfr[valid],
    }


def combine_samples(samples):
    """Concatenate a list of identically structured sample dictionaries."""
    return {
        key: np.concatenate([sample[key] for sample in samples])
        for key in samples[0]
    }


def percentile_summary(values):
    """Return median and lower/upper percentile values."""
    return np.percentile(
        values,
        [lower_percentile, 50, upper_percentile],
    )


def find_mass_balanced_beta_split(mass_values, beta_values):
    """Find a 50/50 sloping beta split with matched median stellar mass.

    For every trial slope, the intercept lies between the two central beta
    residuals, placing half the objects above and half below the line. The
    selected slope minimizes the difference between the median stellar masses
    of those groups. A coarse-to-fine grid avoids a SciPy dependency.
    """
    mass_pivot = np.median(mass_values)
    centred_mass = mass_values - mass_pivot
    initial_slope = np.polyfit(centred_mass, beta_values, 1)[0]

    search_centre = initial_slope
    search_half_width = balance_slope_half_width
    best_result = None

    for _ in range(balance_refinements):
        trial_slopes = np.linspace(
            search_centre - search_half_width,
            search_centre + search_half_width,
            balance_grid_size,
        )

        for slope in trial_slopes:
            residual = beta_values - slope * centred_mass
            number_below = len(residual) // 2
            residual_order = np.argpartition(residual, number_below)

            below = np.zeros(len(residual), dtype=bool)
            below[residual_order[:number_below]] = True
            above = ~below

            # Put the line halfway between the two central residuals. This
            # makes the division exactly 50/50 (or different by one galaxy
            # when the sample size is odd).
            lower_edge = np.max(residual[below])
            upper_edge = np.min(residual[above])
            intercept = 0.5 * (lower_edge + upper_edge)

            median_mass_difference = abs(
                np.median(mass_values[above])
                - np.median(mass_values[below])
            )
            score = median_mass_difference

            if best_result is None or score < best_result[0]:
                best_result = (
                    score,
                    slope,
                    intercept,
                    above,
                    below,
                    median_mass_difference,
                )

        search_centre = best_result[1]
        search_half_width /= 10.0

    _, slope, intercept, above, below, mass_difference = best_result
    return {
        "slope": slope,
        "intercept": intercept,
        "mass_pivot": mass_pivot,
        "above": above,
        "below": below,
        "median_mass_difference": mass_difference,
    }


# ================================================================
# Load both boxes at every redshift and combine them
# ================================================================
samples = []

for snapshot in snapshots:
    samples.append(load_selected_galaxies(
        template_m25.format(snapshot, dust_law),
        magnitude_limit_m25,
    ))
    samples.append(load_selected_galaxies(
        template_m50.format(snapshot, dust_law),
        magnitude_limit_m50,
    ))

sample = combine_samples(samples)

if secondary_property not in property_settings:
    raise ValueError(
        f"Unknown secondary_property={secondary_property!r}. "
        f"Choose from {list(property_settings)}."
    )

property_info = property_settings[secondary_property]
secondary_values = sample[property_info["data_key"]]

# Use only galaxies with a finite value of the selected secondary property.
analysis_valid = np.isfinite(secondary_values)
log_mass = sample["log_stellar_mass"][analysis_valid]
beta = sample["beta"][analysis_valid]
secondary_values = secondary_values[analysis_valid]


# ================================================================
# Define four equally populated regions
# ================================================================
# First split the complete sample exactly 50/50 in stellar mass.
mass_order = np.argsort(log_mass)
number_low_mass = len(log_mass) // 2
low_mass = np.zeros(len(log_mass), dtype=bool)
low_mass[mass_order[:number_low_mass]] = True
high_mass = ~low_mass
mass_split = 0.5 * (
    np.max(log_mass[low_mass]) + np.min(log_mass[high_mass])
)

# Within each mass half, find a sloping beta division that produces a 50/50
# split while matching the median stellar masses above and below the line.
low_mass_split = find_mass_balanced_beta_split(
    log_mass[low_mass],
    beta[low_mass],
)
high_mass_split = find_mass_balanced_beta_split(
    log_mass[high_mass],
    beta[high_mass],
)

# Map the two local results back onto masks for the complete combined sample.
low_mass_indices = np.flatnonzero(low_mass)
high_mass_indices = np.flatnonzero(high_mass)

low_mass_low_beta = np.zeros(len(log_mass), dtype=bool)
low_mass_high_beta = np.zeros(len(log_mass), dtype=bool)
high_mass_low_beta = np.zeros(len(log_mass), dtype=bool)
high_mass_high_beta = np.zeros(len(log_mass), dtype=bool)

low_mass_low_beta[low_mass_indices[low_mass_split["below"]]] = True
low_mass_high_beta[low_mass_indices[low_mass_split["above"]]] = True
high_mass_low_beta[high_mass_indices[high_mass_split["below"]]] = True
high_mass_high_beta[high_mass_indices[high_mass_split["above"]]] = True

regions = [
    {
        "name": "Low mass, low beta",
        "short_name": r"low $M_\star$, low $\beta$",
        "mask": low_mass_low_beta,
        "marker": "o",
    },
    {
        "name": "Low mass, high beta",
        "short_name": r"low $M_\star$, high $\beta$",
        "mask": low_mass_high_beta,
        "marker": "o",
    },
    {
        "name": "High mass, low beta",
        "short_name": r"high $M_\star$, low $\beta$",
        "mask": high_mass_low_beta,
        "marker": "s",
    },
    {
        "name": "High mass, high beta",
        "short_name": r"high $M_\star$, high $\beta$",
        "mask": high_mass_high_beta,
        "marker": "s",
    },
]


# Calculate the representative position and secondary property of each region.
for region in regions:
    mask = region["mask"]
    mass16, mass50, mass84 = percentile_summary(log_mass[mask])
    beta16, beta50, beta84 = percentile_summary(beta[mask])
    property16, property50, property84 = percentile_summary(
        secondary_values[mask]
    )

    region.update({
        "count": np.count_nonzero(mask),
        "mass16": mass16,
        "mass50": mass50,
        "mass84": mass84,
        "beta16": beta16,
        "beta50": beta50,
        "beta84": beta84,
        "property16": property16,
        "property50": property50,
        "property84": property84,
    })

    print(
        f"{region['name']}: N={region['count']}, "
        f"median log10(M*/Msun)={mass50:.3f}, "
        f"median beta={beta50:.3f}, "
        f"median {property_info['description']}={property50:.3f} "
        f"(-{property50 - property16:.3f}, "
        f"+{property84 - property50:.3f})"
    )

print(
    "\nMass-balanced beta divisions:"
    f"\n  Lower-mass slope = {low_mass_split['slope']:.4f}, "
    f"median-mass difference = "
    f"{low_mass_split['median_mass_difference']:.5f} dex"
    f"\n  Higher-mass slope = {high_mass_split['slope']:.4f}, "
    f"median-mass difference = "
    f"{high_mass_split['median_mass_difference']:.5f} dex"
)


# ================================================================
# Single-panel figure
# ================================================================
# MNRAS single-column figure: approximately 3.3--3.5 inches wide.
# A near-square main panel is easier to read than a tall, narrow panel.
fig, ax = plt.subplots(
    figsize=(3.5, 4.2),
    constrained_layout=True,
)

# A light subsample shows the underlying distribution without dominating it.
rng = np.random.default_rng(random_seed)
if len(log_mass) > max_background_points:
    background_indices = rng.choice(
        len(log_mass),
        size=max_background_points,
        replace=False,
    )
else:
    background_indices = np.arange(len(log_mass))

ax.scatter(
    log_mass[background_indices],
    beta[background_indices],
    s=3,
    color="0.72",
    alpha=0.16,
    linewidths=0,
    rasterized=True,
    zorder=1,
)

# Show the exact boundaries used to construct the four populations.
ax.axvline(
    mass_split,
    color="0.35",
    linestyle="--",
    linewidth=0.9,
    zorder=2,
)
line_x_lower = np.linspace(
    np.nanpercentile(log_mass, 0.5),
    mass_split,
    200,
)
line_y_lower = (
    low_mass_split["slope"]
    * (line_x_lower - low_mass_split["mass_pivot"])
    + low_mass_split["intercept"]
)
ax.plot(
    line_x_lower,
    line_y_lower,
    color="0.35",
    linestyle="--",
    linewidth=0.9,
    zorder=2,
)

line_x_upper = np.linspace(
    mass_split,
    np.nanpercentile(log_mass, 99.5),
    200,
)
line_y_upper = (
    high_mass_split["slope"]
    * (line_x_upper - high_mass_split["mass_pivot"])
    + high_mass_split["intercept"]
)
ax.plot(
    line_x_upper,
    line_y_upper,
    color="0.35",
    linestyle="--",
    linewidth=0.9,
    zorder=2,
)

# Use a common colour scale for the four secondary-property medians.
property_medians = np.array([region["property50"] for region in regions])
colour_padding = max(0.03, 0.08 * np.ptp(property_medians))
normalisation = Normalize(
    vmin=np.min(property_medians) - colour_padding,
    vmax=np.max(property_medians) + colour_padding,
)
cmap = plt.get_cmap("viridis")

# Join the low-beta and high-beta representative points at fixed mass half.
# These guides make the within-mass secondary-property comparison easy to follow.
for first, second in ((regions[0], regions[1]), (regions[2], regions[3])):
    ax.plot(
        [first["mass50"], second["mass50"]],
        [first["beta50"], second["beta50"]],
        color="0.30",
        linestyle=":",
        linewidth=1.0,
        zorder=3,
    )

for region in regions:
    colour = cmap(normalisation(region["property50"]))

    # x/y error bars show the 16th--84th percentile extent of each region.
    ax.errorbar(
        region["mass50"],
        region["beta50"],
        xerr=[[region["mass50"] - region["mass16"]],
              [region["mass84"] - region["mass50"]]],
        yerr=[[region["beta50"] - region["beta16"]],
              [region["beta84"] - region["beta50"]]],
        fmt="none",
        ecolor=colour,
        elinewidth=1.3,
        capsize=2.5,
        capthick=1.0,
        zorder=4,
    )

    ax.scatter(
        region["mass50"],
        region["beta50"],
        s=105,
        marker=region["marker"],
        facecolor=colour,
        edgecolor="white",
        linewidth=1.2,
        zorder=5,
    )

    # Report only the numerical median and spread beside each point. The
    # colour-bar label identifies the property, which avoids repeating a long
    # equation four times in a narrow single-column figure.
    property_label = (
        rf"${region['property50']:.2f}"
        rf"^{{+{region['property84'] - region['property50']:.2f}}}"
        rf"_{{-{region['property50'] - region['property16']:.2f}}}$"
    )

    if "Low mass" in region["name"]:
        x_offset = -10
        horizontal_alignment = "right"
    else:
        x_offset = 10
        horizontal_alignment = "left"

    y_offset = 11 if "high beta" in region["name"].lower() else -13
    vertical_alignment = "bottom" if y_offset > 0 else "top"

    ax.annotate(
        property_label,
        (region["mass50"], region["beta50"]),
        xytext=(x_offset, y_offset),
        textcoords="offset points",
        ha=horizontal_alignment,
        va=vertical_alignment,
        fontsize=9.5,
        color="0.12",
        zorder=6,
    )


# Data-driven limits suppress extreme outliers while preserving the population.
x_lower, x_upper = np.nanpercentile(log_mass, [0.5, 99.5])
y_lower, y_upper = np.nanpercentile(beta, [0.5, 99.5])
x_padding = 0.04 * (x_upper - x_lower)
y_padding = 0.08 * (y_upper - y_lower)
ax.set_xlim(x_lower - x_padding, x_upper + x_padding)
ax.set_ylim(y_lower - y_padding, y_upper + y_padding)

ax.set_xlabel(r"Stellar mass, $\log_{10}(M_\star/\mathrm{M}_\odot)$")
ax.set_ylabel(r"UV slope, $\beta$")
ax.minorticks_on()
ax.tick_params(which="major", length=4.0, width=0.8)
ax.tick_params(which="minor", length=2.2, width=0.6)

# Marker shape identifies the stellar-mass half; colour carries the selected
# secondary property.
mass_legend = [
    Line2D(
        [0], [0], marker="o", linestyle="none", markersize=7.5,
        markerfacecolor="0.45", markeredgecolor="white",
        label=r"Lower-$M_\star$",
    ),
    Line2D(
        [0], [0], marker="s", linestyle="none", markersize=7.5,
        markerfacecolor="0.45", markeredgecolor="white",
        label=r"Higher-$M_\star$",
    ),
]
ax.legend(
    handles=mass_legend,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.42, 1.0),
    ncol=2,
    columnspacing=1.2,
    handletextpad=0.35,
    borderaxespad=0.6,
    fontsize=9.5,
)

scalar_mappable = plt.cm.ScalarMappable(norm=normalisation, cmap=cmap)
scalar_mappable.set_array([])
colour_bar = fig.colorbar(
    scalar_mappable,
    ax=ax,
    orientation="horizontal",
    pad=0.08,
    fraction=0.055,
    aspect=28,
)
colour_bar.set_label(
    property_info["colourbar_label"],
    fontsize=10.5,
    labelpad=3,
)
colour_bar.ax.tick_params(
    direction="in",
    length=3.5,
    width=0.8,
    labelsize=9.5,
)

fig.savefig(
    f"Beta_vs_StellarMass_FourRegions_{property_info['file_tag']}.png",
    dpi=300,
    bbox_inches="tight",
)
fig.savefig(
    f"Beta_vs_StellarMass_FourRegions_{property_info['file_tag']}.pdf",
    bbox_inches="tight",
)

plt.show()
