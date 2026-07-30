import numpy as np
import caesar
import matplotlib.pyplot as plt
from utils.beta_utils import *

# ------------------------------------------------
# Dust laws
# ------------------------------------------------
dust_laws = ["calzetti", "lmc", "smc", "mw"]

colors = {
    "calzetti": "tab:blue",
    "lmc": "tab:orange",
    "smc": "tab:green",
    "mw": "tab:red",
}


# ------------------------------------------------
# Snapshots
# ------------------------------------------------
snapshots = ["016", "019", "022", "026", "030", "036"]


# ------------------------------------------------
# File templates
# ------------------------------------------------
template_m25 = (
    "/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/Dust_extin/"
    "m25n1024/caesar_m25n1024_{}_{}.hdf5"
)

template_m50 = (
    "/cosma8/data/dp376/dc-xian3/simba-eor/EoRData/Dust_extin/"
    "m50n1024/caesar_m50n1024_{}_{}.hdf5"
)


bands = ["i1500", "i2300", "i2800"]
wavelengths = np.array([1500, 2300, 2800])


# ------------------------------------------------
# Figure
# ------------------------------------------------
fig, axes = plt.subplots(
    3,
    2,
    figsize=(10, 12),
    sharex=True,
    sharey=True
)

axes = axes.flatten()


# ------------------------------------------------
# Loop snapshots
# ------------------------------------------------
for i, snap in enumerate(snapshots):

    ax = axes[i]

    for dust in dust_laws:

        # -------------------------
        # Load files
        # -------------------------
        f25 = template_m25.format(snap, dust)
        f50 = template_m50.format(snap, dust)

        obj = caesar.load(f25)
        obj_m50 = caesar.load(f50)

        Z = obj.simulation.redshift

        # -------------------------
        # SFR
        # -------------------------
        sfr_m25 = np.array([
            g.sfr.to("Msun/yr").value
            for g in obj.galaxies
        ])

        sfr_m50 = np.array([
            g.sfr.to("Msun/yr").value
            for g in obj_m50.galaxies
        ])

        # -------------------------
        # Magnitude cuts
        # -------------------------
        mask_m25 = np.array([
            g.absmag["i1500"]
            for g in obj.galaxies
        ]) < -16

        mask_m50 = np.array([
            g.absmag["i1500"]
            for g in obj_m50.galaxies
        ]) < -17.5

        sfr_combined = np.concatenate([
            sfr_m25[mask_m25],
            sfr_m50[mask_m50]
        ])

        # -------------------------
        # Magnitudes
        # -------------------------
        mags_m25 = np.array([
            [g.absmag[band] for g in obj.galaxies]
            for band in bands
        ])

        mags_m50 = np.array([
            [g.absmag[band] for g in obj_m50.galaxies]
            for band in bands
        ])

        mags_combined = np.concatenate([
            mags_m25[:, mask_m25],
            mags_m50[:, mask_m50]
        ], axis=1)

        # -------------------------
        # UV slope
        # -------------------------
        beta_combined = Calbeta(
            mags_combined,
            wavelengths
        )

        print(
            "Total galaxies:",
            len(sfr_combined),
            "zero SFR:",
            np.sum(sfr_combined == 0)
        )

        # -------------------------
        # Remove bad values
        # -------------------------
        valid = (
            np.isfinite(sfr_combined)
            & (sfr_combined > 0)
            & (sfr_combined < 1e3)
        )

        sfr_combined = sfr_combined[valid]
        beta_combined = beta_combined[valid]

        log_sfr = np.log10(sfr_combined)

        print(
            "z =", Z,
            "log SFR range:",
            np.min(log_sfr),
            np.max(log_sfr)
        )

        # -------------------------
        # Bin data
        # -------------------------
        bin_centers, beta_mean, beta_std, bin_count = bin_xy(
            x_values=log_sfr,
            y_values=beta_combined,
            mask_values=None,
            mask_cut=None,
            N_bins=10
        )

        # -------------------------
        # Plot
        # -------------------------
        ax.errorbar(
            bin_centers,
            beta_mean,
            yerr=beta_std,
            color=colors[dust],
            marker='o',
            linestyle='-',
            capsize=2,
            linewidth=2,
            markersize=5,
            label=dust if i == 0 else None
        )

    # =====================================================
    # No dust case
    # =====================================================

    f25 = template_m25.format(snap, "calzetti")
    f50 = template_m50.format(snap, "calzetti")

    obj = caesar.load(f25)
    obj_m50 = caesar.load(f50)

    # -------------------------
    # No dust magnitudes
    # -------------------------
    mags_m25_nodust = np.array([
        [g.absmag_nodust[band] for g in obj.galaxies]
        for band in bands
    ])

    mags_m50_nodust = np.array([
        [g.absmag_nodust[band] for g in obj_m50.galaxies]
        for band in bands
    ])

    # -------------------------
    # SFR
    # -------------------------
    sfr_m25 = np.array([
        g.sfr.to("Msun/yr").value
        for g in obj.galaxies
    ])

    sfr_m50 = np.array([
        g.sfr.to("Msun/yr").value
        for g in obj_m50.galaxies
    ])

    # -------------------------
    # Magnitude cuts
    # -------------------------
    mask_m25 = mags_m25_nodust[0] < -16
    mask_m50 = mags_m50_nodust[0] < -17.5

    sfr_combined = np.concatenate([
        sfr_m25[mask_m25],
        sfr_m50[mask_m50]
    ])

    mags_combined = np.concatenate([
        mags_m25_nodust[:, mask_m25],
        mags_m50_nodust[:, mask_m50]
    ], axis=1)

    # -------------------------
    # UV slope
    # -------------------------
    beta_nodust = Calbeta(
        mags_combined,
        wavelengths
    )

    # -------------------------
    # Remove bad values
    # -------------------------
    valid = (
        np.isfinite(sfr_combined)
        & (sfr_combined > 0)
        & (sfr_combined < 1e3)
    )

    sfr_combined = sfr_combined[valid]
    beta_nodust = beta_nodust[valid]

    log_sfr = np.log10(sfr_combined)

    # -------------------------
    # Bin data
    # -------------------------
    bin_centers_nodust, beta_mean_nodust, beta_std_nodust, _ = bin_xy(
        x_values=log_sfr,
        y_values=beta_nodust,
        mask_values=None,
        mask_cut=None,
        N_bins=10
    )

    # -------------------------
    # Plot
    # -------------------------
    ax.errorbar(
        bin_centers_nodust,
        beta_mean_nodust,
        yerr=beta_std_nodust,
        color="black",
        marker='s',
        linestyle='--',
        capsize=2,
        linewidth=2,
        markersize=5,
        label="no dust" if i == 0 else None
    )

    # -------------------------
    # Formatting
    # -------------------------
    ax.set_xlim(-2, 3)
    ax.set_ylim(-2.6, -0.8)

    ax.text(
        -1.8,
        -1.0,
        f"z = {round(Z)}",
        fontsize=11
    )

    if i >= 4:
        ax.set_xlabel(
            r"log$_{10}$(SFR / M$_\odot$ yr$^{-1}$)"
        )

    if i % 2 == 0:
        ax.set_ylabel(r"$\beta$")


# ------------------------------------------------
# Legend
# ------------------------------------------------
axes[0].legend(
    frameon=False,
    fontsize=10
)

plt.tight_layout()

plt.savefig(
    "Beta_vs_SFR_DustComparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
