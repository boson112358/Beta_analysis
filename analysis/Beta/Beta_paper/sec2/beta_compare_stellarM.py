import numpy as np
import matplotlib.pyplot as plt
import caesar

from utils.beta_utils import Calbeta


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "axes.grid": True,
})


# ============================================================
# Files and parameters
# ============================================================

redshifts = ['016', '019', '022', '026', '030', '036']

dust_law = "calzetti"

bands = ["i1500", "i2300", "i2800"]
wavelengths = np.array([1500, 2300, 2800])


template_m25 = (
    "/cosma8/data/dp376/dc-xian3/simba-eor/"
    "EoRData/FullSpectra_Fit/m25n1024/"
    "caesar_m25n1024_{}_{}.hdf5"
)

template_m50 = (
    "/cosma8/data/dp376/dc-xian3/simba-eor/"
    "EoRData/FullSpectra_Fit/m50n1024/"
    "caesar_m50n1024_{}_{}.hdf5"
)


# ============================================================
# Storage
# ============================================================

zvals = []

# Global beta distributions
beta_spectral_all = []
beta_3point_all = []

# Stellar mass for every galaxy
mass_all = []

# Redshift for every galaxy
z_all = []


# ============================================================
# Global median beta evolution
# ============================================================

median_spectral = []
lower_spectral = []
upper_spectral = []

median_3point = []
lower_3point = []
upper_3point = []

median_delta = []


# ============================================================
# Loop over snapshots
# ============================================================

for z_str in redshifts:

    print("\n" + "=" * 60)
    print(f"Processing z = {z_str}")
    print("=" * 60)


    # --------------------------------------------------------
    # Load catalogues
    # --------------------------------------------------------

    obj_m25 = caesar.load(
        template_m25.format(z_str, dust_law)
    )

    obj_m50 = caesar.load(
        template_m50.format(z_str, dust_law)
    )

    z_sim = obj_m25.simulation.redshift


    # ========================================================
    # M25
    # ========================================================

    mags_m25 = np.array([
        [g.absmag[band] for g in obj_m25.galaxies]
        for band in bands
    ])

    beta_spectral_m25 = np.array([
        g.photometry["beta"]
        for g in obj_m25.galaxies
    ])

    stellar_mass_m25 = np.array([
        g.masses["stellar"]
        for g in obj_m25.galaxies
    ])


    # ========================================================
    # M50
    # ========================================================

    mags_m50 = np.array([
        [g.absmag[band] for g in obj_m50.galaxies]
        for band in bands
    ])

    beta_spectral_m50 = np.array([
        g.photometry["beta"]
        for g in obj_m50.galaxies
    ])

    stellar_mass_m50 = np.array([
        g.masses["stellar"]
        for g in obj_m50.galaxies
    ])


    # ========================================================
    # Magnitude selection
    # ========================================================

    mask_m25 = mags_m25[0] < -16
    mask_m50 = mags_m50[0] < -17.5


    # ========================================================
    # Combine M25 + M50
    # ========================================================

    mags = np.concatenate(
        [
            mags_m25[:, mask_m25],
            mags_m50[:, mask_m50]
        ],
        axis=1
    )

    beta_spectral = np.concatenate(
        [
            beta_spectral_m25[mask_m25],
            beta_spectral_m50[mask_m50]
        ]
    )

    stellar_mass = np.concatenate(
        [
            stellar_mass_m25[mask_m25],
            stellar_mass_m50[mask_m50]
        ]
    )


    # ========================================================
    # Three-point beta
    # ========================================================

    beta_3point = Calbeta(
        mags,
        wavelengths
    )


    # ========================================================
    # Remove invalid values
    # ========================================================

    valid = (
        np.isfinite(beta_spectral)
        & np.isfinite(beta_3point)
        & np.isfinite(stellar_mass)
        & (stellar_mass > 0)
    )

    beta_spectral = beta_spectral[valid]
    beta_3point = beta_3point[valid]
    stellar_mass = stellar_mass[valid]


    if len(beta_spectral) == 0:
        continue


    # ========================================================
    # Difference
    # ========================================================

    delta_beta = beta_3point - beta_spectral


    # ========================================================
    # Store individual galaxies
    # ========================================================

    beta_spectral_all.append(beta_spectral)
    beta_3point_all.append(beta_3point)
    mass_all.append(stellar_mass)
    z_all.append(
        np.full(len(beta_spectral), z_sim)
    )


    # ========================================================
    # Global beta statistics
    # ========================================================

    med_spec = np.median(beta_spectral)
    p16_spec = np.percentile(beta_spectral, 16)
    p84_spec = np.percentile(beta_spectral, 84)

    med_3pt = np.median(beta_3point)
    p16_3pt = np.percentile(beta_3point, 16)
    p84_3pt = np.percentile(beta_3point, 84)

    med_delta = np.median(delta_beta)


    median_spectral.append(med_spec)
    lower_spectral.append(med_spec - p16_spec)
    upper_spectral.append(p84_spec - med_spec)

    median_3point.append(med_3pt)
    lower_3point.append(med_3pt - p16_3pt)
    upper_3point.append(p84_3pt - med_3pt)

    median_delta.append(med_delta)

    zvals.append(z_sim)


    # ========================================================
    # Print results
    # ========================================================

    print(f"z = {z_sim:.2f}")
    print(f"N = {len(beta_spectral)}")

    print(
        f"Spectral:  median = {med_spec:.3f}, "
        f"16-84 = [{p16_spec:.3f}, {p84_spec:.3f}]"
    )

    print(
        f"3-point:   median = {med_3pt:.3f}, "
        f"16-84 = [{p16_3pt:.3f}, {p84_3pt:.3f}]"
    )

    print(
        f"Delta beta (3-point - spectral) = "
        f"{med_delta:.4f}"
    )


# ============================================================
# Convert to arrays
# ============================================================

zvals = np.array(zvals)

median_spectral = np.array(median_spectral)
lower_spectral = np.array(lower_spectral)
upper_spectral = np.array(upper_spectral)

median_3point = np.array(median_3point)
lower_3point = np.array(lower_3point)
upper_3point = np.array(upper_3point)

median_delta = np.array(median_delta)


beta_spectral_all = np.concatenate(beta_spectral_all)
beta_3point_all = np.concatenate(beta_3point_all)

mass_all = np.concatenate(mass_all)
z_all = np.concatenate(z_all)

logMstar_all = np.log10(mass_all)

delta_beta_all = beta_3point_all - beta_spectral_all


# ============================================================
# Print global comparison
# ============================================================

print("\n")
print("=" * 60)
print("GLOBAL COMPARISON")
print("=" * 60)

print(
    f"Overall median Delta beta = "
    f"{np.median(delta_beta_all):.4f}"
)

print(
    f"Overall mean Delta beta = "
    f"{np.mean(delta_beta_all):.4f}"
)

print(
    f"Overall std Delta beta = "
    f"{np.std(delta_beta_all):.4f}"
)


# ============================================================
# Figure 1
# Beta vs stellar mass
#
# Six panels, one for each redshift.
# Spectral and three-point methods are compared.
# ============================================================

fig, axes = plt.subplots(
    2, 3,
    figsize=(15, 9),
    sharex=True,
    sharey=True
)

axes = axes.flatten()


# ------------------------------------------------------------
# Stellar-mass bins
# ------------------------------------------------------------

mass_bins = np.arange(
    7.0,
    10.51,
    0.25
)

mass_centres = 0.5 * (
    mass_bins[:-1] + mass_bins[1:]
)


for ax, z in zip(axes, zvals):

    # --------------------------------------------------------
    # Select this redshift
    # --------------------------------------------------------

    zmask = np.isclose(
        z_all,
        z,
        atol=0.01
    )

    logM = logMstar_all[zmask]

    beta_spec = beta_spectral_all[zmask]
    beta_3pt = beta_3point_all[zmask]


    # --------------------------------------------------------
    # Binned statistics
    # --------------------------------------------------------

    med_spec = []
    low_spec = []
    high_spec = []

    med_3pt = []
    low_3pt = []
    high_3pt = []

    valid_mass = []


    for i in range(len(mass_bins) - 1):

        binmask = (
            (logM >= mass_bins[i])
            & (logM < mass_bins[i + 1])
        )

        if np.sum(binmask) < 10:
            continue

        spec_bin = beta_spec[binmask]
        three_bin = beta_3pt[binmask]

        valid_mass.append(
            mass_centres[i]
        )

        # Spectral
        med_spec.append(
            np.median(spec_bin)
        )

        low_spec.append(
            np.percentile(spec_bin, 16)
        )

        high_spec.append(
            np.percentile(spec_bin, 84)
        )

        # Three point
        med_3pt.append(
            np.median(three_bin)
        )

        low_3pt.append(
            np.percentile(three_bin, 16)
        )

        high_3pt.append(
            np.percentile(three_bin, 84)
        )


    valid_mass = np.array(valid_mass)

    med_spec = np.array(med_spec)
    low_spec = np.array(low_spec)
    high_spec = np.array(high_spec)

    med_3pt = np.array(med_3pt)
    low_3pt = np.array(low_3pt)
    high_3pt = np.array(high_3pt)


    # --------------------------------------------------------
    # Plot spectral
    # --------------------------------------------------------

    ax.plot(
        valid_mass,
        med_spec,
        'o-',
        linewidth=1.5,
        markersize=4,
        label="Spectral β"
    )

    ax.fill_between(
        valid_mass,
        low_spec,
        high_spec,
        alpha=0.15
    )


    # --------------------------------------------------------
    # Plot three-point
    # --------------------------------------------------------

    ax.plot(
        valid_mass,
        med_3pt,
        's--',
        linewidth=1.5,
        markersize=4,
        label="Three-point β"
    )

    ax.fill_between(
        valid_mass,
        low_3pt,
        high_3pt,
        alpha=0.15
    )


    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_title(
        rf"$z={z:.2f}$"
    )

    ax.grid(True, alpha=0.3)


axes[0].legend()

axes[3].set_xlabel(
    r"$\log_{10}(M_\star/M_\odot)$"
)

axes[4].set_xlabel(
    r"$\log_{10}(M_\star/M_\odot)$"
)

axes[5].set_xlabel(
    r"$\log_{10}(M_\star/M_\odot)$"
)

axes[0].set_ylabel(
    r"UV slope $\beta$"
)

axes[3].set_ylabel(
    r"UV slope $\beta$"
)


fig.suptitle(
    r"Comparison of $\beta$--$M_\star$ relations",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    "beta_vs_stellarmass_spectral_vs_3point.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# Figure 2
# Delta beta vs stellar mass
#
# This is the most direct test of systematic differences.
# ============================================================

fig, axes = plt.subplots(
    2, 3,
    figsize=(15, 9),
    sharex=True,
    sharey=True
)

axes = axes.flatten()


for ax, z in zip(axes, zvals):

    # --------------------------------------------------------
    # Select redshift
    # --------------------------------------------------------

    zmask = np.isclose(
        z_all,
        z,
        atol=0.01
    )

    logM = logMstar_all[zmask]
    delta = delta_beta_all[zmask]


    # --------------------------------------------------------
    # Bin delta beta
    # --------------------------------------------------------

    med_delta = []
    low_delta = []
    high_delta = []
    valid_mass = []


    for i in range(len(mass_bins) - 1):

        binmask = (
            (logM >= mass_bins[i])
            & (logM < mass_bins[i + 1])
        )

        if np.sum(binmask) < 10:
            continue

        delta_bin = delta[binmask]

        valid_mass.append(
            mass_centres[i]
        )

        med_delta.append(
            np.median(delta_bin)
        )

        low_delta.append(
            np.percentile(delta_bin, 16)
        )

        high_delta.append(
            np.percentile(delta_bin, 84)
        )


    valid_mass = np.array(valid_mass)

    med_delta = np.array(med_delta)
    low_delta = np.array(low_delta)
    high_delta = np.array(high_delta)


    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    ax.plot(
        valid_mass,
        med_delta,
        'o-',
        linewidth=1.5,
        markersize=4
    )

    ax.fill_between(
        valid_mass,
        low_delta,
        high_delta,
        alpha=0.2
    )


    # Zero difference
    ax.axhline(
        0,
        linestyle='--',
        linewidth=1.2
    )


    ax.set_title(
        rf"$z={z:.2f}$"
    )

    ax.grid(True, alpha=0.3)


axes[3].set_xlabel(
    r"$\log_{10}(M_\star/M_\odot)$"
)

axes[4].set_xlabel(
    r"$\log_{10}(M_\star/M_\odot)$"
)

axes[5].set_xlabel(
    r"$\log_{10}(M_\star/M_\odot)$"
)

axes[0].set_ylabel(
    r"$\Delta\beta = \beta_{\rm 3pt}-\beta_{\rm spec}$"
)

axes[3].set_ylabel(
    r"$\Delta\beta = \beta_{\rm 3pt}-\beta_{\rm spec}$"
)


fig.suptitle(
    r"Difference between the two $\beta$ estimators",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    "delta_beta_vs_stellarmass.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# Figure 3
# Median beta evolution
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 5.5)
)

ax.errorbar(
    zvals,
    median_spectral,
    yerr=[
        lower_spectral,
        upper_spectral
    ],
    fmt='o-',
    capsize=4,
    linewidth=1.5,
    markersize=6,
    label="Spectral β"
)

ax.errorbar(
    zvals,
    median_3point,
    yerr=[
        lower_3point,
        upper_3point
    ],
    fmt='s--',
    capsize=4,
    linewidth=1.5,
    markersize=5,
    label="Three-point β"
)

ax.set_xlabel("Redshift")
ax.set_ylabel(r"UV slope $\beta$")

ax.legend()

plt.tight_layout()

plt.savefig(
    "beta_evolution_spectral_vs_3point.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# Figure 4
# Galaxy-by-galaxy comparison
# ============================================================

fig, ax = plt.subplots(
    figsize=(6, 6)
)

ax.scatter(
    beta_spectral_all,
    beta_3point_all,
    s=4,
    alpha=0.15
)

xmin = min(
    beta_spectral_all.min(),
    beta_3point_all.min()
)

xmax = max(
    beta_spectral_all.max(),
    beta_3point_all.max()
)

ax.plot(
    [xmin, xmax],
    [xmin, xmax],
    '--',
    linewidth=1.5,
    label="1:1"
)

ax.set_xlabel(
    r"Spectral-fit $\beta$"
)

ax.set_ylabel(
    r"Three-point $\beta$"
)

ax.legend()

ax.set_title(
    r"Galaxy-by-galaxy $\beta$ comparison"
)

plt.tight_layout()

plt.savefig(
    "beta_spectral_vs_3point_scatter.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
