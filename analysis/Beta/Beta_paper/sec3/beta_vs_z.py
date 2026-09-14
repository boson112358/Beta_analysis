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
    "EoRData/Dust_extin/m25n1024/"
    "caesar_m25n1024_{}_{}.hdf5"
)

template_m50 = (
    "/cosma8/data/dp376/dc-xian3/simba-eor/"
    "EoRData/Dust_extin/m50n1024/"
    "caesar_m50n1024_{}_{}.hdf5"
)


# ============================================================
# Storage
# ============================================================

zvals = []

median_beta = []
beta_lower = []
beta_upper = []

# Store beta distribution at each redshift
beta_distributions = []


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


    # --------------------------------------------------------
    # Magnitudes
    # --------------------------------------------------------

    mags_m25 = np.array([
        [g.absmag[band] for g in obj_m25.galaxies]
        for band in bands
    ])

    mags_m50 = np.array([
        [g.absmag[band] for g in obj_m50.galaxies]
        for band in bands
    ])


    # --------------------------------------------------------
    # Stellar mass
    # --------------------------------------------------------

    stellar_mass_m25 = np.array([
        g.masses["stellar"]
        for g in obj_m25.galaxies
    ])

    stellar_mass_m50 = np.array([
        g.masses["stellar"]
        for g in obj_m50.galaxies
    ])


    # --------------------------------------------------------
    # Magnitude cuts
    # --------------------------------------------------------

    mask_m25 = mags_m25[0] < -16
    mask_m50 = mags_m50[0] < -17.5


    # --------------------------------------------------------
    # Combine M25 + M50
    # --------------------------------------------------------

    mags = np.concatenate(
        [
            mags_m25[:, mask_m25],
            mags_m50[:, mask_m50]
        ],
        axis=1
    )


    if mags.shape[1] == 0:
        continue


    # --------------------------------------------------------
    # Calculate beta
    # --------------------------------------------------------

    beta = Calbeta(
        mags,
        wavelengths
    )


    # --------------------------------------------------------
    # Median + 16-84 percentile
    # --------------------------------------------------------

    median = np.median(beta)

    p16 = np.percentile(beta, 16)
    p84 = np.percentile(beta, 84)

    median_beta.append(median)

    beta_lower.append(median - p16)
    beta_upper.append(p84 - median)

    zvals.append(z_sim)


    # --------------------------------------------------------
    # Store beta distribution
    # --------------------------------------------------------

    beta_distributions.append(beta)


    print(
        f"  z = {z_sim:.2f}, "
        f"N = {len(beta)}, "
        f"median beta = {median:.3f}, "
        f"16-84 = [{p16:.3f}, {p84:.3f}]"
    )


# ============================================================
# Convert to arrays
# ============================================================

zvals = np.array(zvals)

median_beta = np.array(median_beta)
beta_lower = np.array(beta_lower)
beta_upper = np.array(beta_upper)


# ============================================================
# Create figure
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(13, 5.5)
)


# ============================================================
# LEFT PANEL
# Median beta evolution
# ============================================================

ax1.errorbar(
    zvals,
    median_beta,
    yerr=[beta_lower, beta_upper],
    fmt='o-',
    capsize=4,
    linewidth=1.5,
    markersize=6,
    label="Median β"
)

ax1.set_xlabel("Redshift")
ax1.set_ylabel(r"UV slope $\beta$")
ax1.set_title("β Evolution")

ax1.legend()

# Optional: reverse x-axis so cosmic time goes left -> right
# (high z on left, low z on right)
# Do not invert
#ax1.invert_xaxis()


# ============================================================
# RIGHT PANEL
# Beta distributions
# ============================================================

# Use common bins for all redshifts
all_beta = np.concatenate(beta_distributions)

bins = np.linspace(
    np.percentile(all_beta, 0.5),
    np.percentile(all_beta, 99.5),
    30
)


for z, beta in zip(zvals, beta_distributions):

    ax2.hist(
        beta,
        bins=bins,
        histtype='step',
        linewidth=1.5,
        density=False,
        label=f"z = {z:.1f}"
    )


ax2.set_xlabel(r"UV slope $\beta$")
ax2.set_ylabel("Number of galaxies")
ax2.set_title("β Distribution")

ax2.legend(
    fontsize=8,
    ncol=2
)


# ============================================================
# Final formatting
# ============================================================

plt.tight_layout()

plt.savefig(
    "Beta_evolution_and_distribution_calzetti.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
