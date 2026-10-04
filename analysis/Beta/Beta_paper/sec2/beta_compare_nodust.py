"""Compare dust-free three-point and spectral UV slopes.

Run from the same environment as beta_compare.py (requires CAESAR and
utils.beta_utils.Calbeta). The default observed-M1500 selection preserves
the original comparison sample. Set selection_nodust=True to select using
intrinsic M1500 instead. Dust-free photometry can still include nebular emission.
"""

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

# Keep the original observed-magnitude selection by default.
selection_nodust = False


def nodust_mag(galaxy, band):
    """Read intrinsic AB magnitudes; never substitute attenuated values."""
    key = band
    if key not in galaxy.absmag_nodust:
        raise KeyError(
            f"Missing dust-free magnitude {key!r}. "
            f"Available absmag_nodust keys: {list(galaxy.absmag_nodust.keys())}"
        )
    return galaxy.absmag_nodust[key]


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

beta_spectral_all = []
beta_3point_all = []

median_spectral = []
median_3point = []

lower_spectral = []
upper_spectral = []

lower_3point = []
upper_3point = []


# ============================================================
# Loop over snapshots
# ============================================================

for z_str in redshifts:

    print(f"Processing z = {z_str}")

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
        [nodust_mag(g, band) for g in obj_m25.galaxies]
        for band in bands
    ])

    beta_spectral_m25 = np.array([
        g.photometry["beta_nodust"]
        for g in obj_m25.galaxies
    ])


    # ========================================================
    # M50
    # ========================================================

    mags_m50 = np.array([
        [nodust_mag(g, band) for g in obj_m50.galaxies]
        for band in bands
    ])

    beta_spectral_m50 = np.array([
        g.photometry["beta_nodust"]
        for g in obj_m50.galaxies
    ])


    # ========================================================
    # Magnitude selection
    # ========================================================

    selection_m25 = (mags_m25[0] if selection_nodust else np.array([
        g.absmag["i1500"] for g in obj_m25.galaxies
    ]))
    selection_m50 = (mags_m50[0] if selection_nodust else np.array([
        g.absmag["i1500"] for g in obj_m50.galaxies
    ]))
    mask_m25 = (selection_m25 < -16) & np.all(np.isfinite(mags_m25), axis=0)
    mask_m50 = (selection_m50 < -17.5) & np.all(np.isfinite(mags_m50), axis=0)


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
    )

    beta_spectral = beta_spectral[valid]
    beta_3point = beta_3point[valid]


    if len(beta_spectral) == 0:
        continue


    # ========================================================
    # Store individual galaxies
    # ========================================================

    beta_spectral_all.append(beta_spectral)
    beta_3point_all.append(beta_3point)


    # ========================================================
    # Median + 16-84 percentile
    # ========================================================

    med_spec = np.median(beta_spectral)
    p16_spec = np.percentile(beta_spectral, 16)
    p84_spec = np.percentile(beta_spectral, 84)

    med_3pt = np.median(beta_3point)
    p16_3pt = np.percentile(beta_3point, 16)
    p84_3pt = np.percentile(beta_3point, 84)


    median_spectral.append(med_spec)
    lower_spectral.append(med_spec - p16_spec)
    upper_spectral.append(p84_spec - med_spec)

    median_3point.append(med_3pt)
    lower_3point.append(med_3pt - p16_3pt)
    upper_3point.append(p84_3pt - med_3pt)

    zvals.append(z_sim)


    # ========================================================
    # Print results
    # ========================================================

    delta_beta = beta_3point - beta_spectral

    print(
        f"  z = {z_sim:.2f}, N = {len(beta_spectral)}"
    )

    print(
        f"  Spectral:  median = {med_spec:.3f}, "
        f"16-84 = [{p16_spec:.3f}, {p84_spec:.3f}]"
    )

    print(
        f"  3-point:   median = {med_3pt:.3f}, "
        f"16-84 = [{p16_3pt:.3f}, {p84_3pt:.3f}]"
    )

    print(
        f"  Δβ (3-point - spectral): "
        f"median = {np.median(delta_beta):.3f}"
    )


# ============================================================
# Convert to arrays
# ============================================================

if not beta_spectral_all:
    raise RuntimeError("No valid dust-free beta pairs remain after selection.")

zvals = np.array(zvals)

median_spectral = np.array(median_spectral)
median_3point = np.array(median_3point)

lower_spectral = np.array(lower_spectral)
upper_spectral = np.array(upper_spectral)

lower_3point = np.array(lower_3point)
upper_3point = np.array(upper_3point)

beta_spectral_all = np.concatenate(beta_spectral_all)
beta_3point_all = np.concatenate(beta_3point_all)


# ============================================================
# Figure
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(13, 5.5)
)


# ============================================================
# LEFT: galaxy-by-galaxy comparison
# ============================================================

ax1.scatter(
    beta_spectral_all,
    beta_3point_all,
    s=5,
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

ax1.plot(
    [xmin, xmax],
    [xmin, xmax],
    '--',
    linewidth=1.5,
    label="1:1"
)

ax1.set_xlabel(r"Spectral-fit $\beta_{\mathrm{nodust}}$")
ax1.set_ylabel(r"Three-point $\beta_{\mathrm{nodust}}$")
ax1.set_title(r"Dust-free galaxy-by-galaxy comparison")

ax1.legend()


# ============================================================
# RIGHT: redshift evolution
# ============================================================

ax2.errorbar(
    zvals,
    median_spectral,
    yerr=[lower_spectral, upper_spectral],
    fmt='o-',
    capsize=4,
    linewidth=1.5,
    markersize=6,
    label="Spectral fit"
)

ax2.errorbar(
    zvals,
    median_3point,
    yerr=[lower_3point, upper_3point],
    fmt='s--',
    capsize=4,
    linewidth=1.5,
    markersize=5,
    label="Three-point"
)

ax2.set_xlabel("Redshift")
ax2.set_ylabel(r"UV slope $\beta_{\mathrm{nodust}}$")
ax2.set_title(r"Evolution of $\beta_{\mathrm{nodust}}$")

ax2.legend()


plt.tight_layout()

plt.savefig(
    f"Beta_nodust_spectral_vs_3point_{dust_law}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    f"Beta_nodust_spectral_vs_3point_{dust_law}.pdf",
    bbox_inches="tight"
)

plt.show()
